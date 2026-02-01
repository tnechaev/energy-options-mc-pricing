import sdmx
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

class EnergyOptionPricer:
    
    def __init__(self, seed=42):
        self.client = sdmx.Client("ECB")
        self.dataflow_id = "YC"
        self._crn_cache = {}  # cache CRNs keyed by (T, n_paths)
        self.seed = seed
        
    @staticmethod
    def black76_call(F0, K, T, r, sigma):
        if T <= 0:
            return max(F0 - K, 0.0)
        d1 = (np.log(F0/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return np.exp(-r*T) * (F0*norm.cdf(d1) - K*norm.cdf(d2))

    # -------------------------
    # CRN + Antithetic generator
    # -------------------------
    def _generate_crn(self, T, n_paths, n_steps, antithetic=True):
        key = (T, n_paths, n_steps, antithetic)
        if key not in self._crn_cache:
            rng = np.random.default_rng(self.seed + int(1000*T))
            Z1 = rng.normal(size=(n_paths, n_steps))
            Z2 = rng.normal(size=(n_paths, n_steps))
            UJ = rng.uniform(size=(n_paths, n_steps))  # jump triggers
            JY = rng.normal(size=(n_paths, n_steps))   # jump sizes

            if antithetic:
                Z1 = np.vstack([Z1, -Z1])
                Z2 = np.vstack([Z2, -Z2])
                UJ = np.vstack([UJ, UJ])
                JY = np.vstack([JY, -JY])

            self._crn_cache[key] = (Z1, Z2, UJ, JY)
        return self._crn_cache[key]
    
    # -------------------------
    # Fetch ECB yield curve
    # -------------------------
    
    def get_yield_curve(self, maturities, start_period=None, end_period=None):
        keys = {
            "DATA_TYPE_FM": maturities,
            "INSTRUMENT_FM": ["G_N_A"],
            "FREQ": "B"
        }
        data_msg = self.client.data(self.dataflow_id, key=keys,
                                    params={"startPeriod": start_period, "endPeriod": end_period})
        df = sdmx.to_pandas(data_msg, datetime={"dim":"TIME_PERIOD"})
        return df

    # -------------------------
    # Convert to discount factors
    # -------------------------
    
    def discount_factors_from_latest(self, yc_df, desired_maturities=None):
        latest = yc_df.dropna(how="all").iloc[-1]
        if desired_maturities is None:
            desired_maturities = ["SR_3M","SR_6M","SR_1Y","SR_2Y","SR_3Y","SR_5Y","SR_10Y","SR_20Y","SR_30Y"]

        tenor_cols = [c for c in latest.index if c[-1] in desired_maturities]
        rates = np.array([latest[c] for c in tenor_cols]) / 100

        def mat_to_years(m):
            if "M" in m:
                return float(m.replace("SR_","").replace("M","")) / 12
            elif "Y" in m:
                return float(m.replace("SR_","").replace("Y",""))
            else:
                return None

        tenors = np.array([mat_to_years(c[-1]) for c in tenor_cols])
        DF = np.exp(-rates * tenors)
        df_disc = pd.DataFrame({"T": tenors, "ZeroRate": rates, "DF": DF})
        return df_disc

    # -------------------------
    # Process power price data, set to Germany here
    # -------------------------
    
    def process_power_data(self, file_path, country_iso="DEU", rolling_days=30):
        df_power = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        df_power_country = df_power[df_power["ISO3 Code"] == country_iso].copy()
        df_power_country = df_power_country.rename(columns={"Price (EUR/MWhe)": "Price"})
        prices = df_power_country["Price"].astype(float)
        fwd_proxy = prices.rolling(rolling_days).mean().dropna()
        return prices, fwd_proxy

    # -------------------------
    # Monte Carlo option pricing
    # -------------------------
    
    def price_option_mc(self, F0, K, T, df_disc,
                        sigma0, kappa, a, b, eta, rho,
                        jump_lambda, jump_mean, jump_std,
                        theta_func=None,
                        n_paths=100_000,
                        n_steps=251):
        dt = T / n_steps
        df_interp = interp1d(df_disc["T"], df_disc["DF"], kind="linear", fill_value="extrapolate")
        discount_factor = float(df_interp(T))

        if theta_func is None:
            theta_func = lambda t: F0

        Z1, Z2, UJ, JY = self._generate_crn(T, n_paths, n_steps, antithetic=True)
        n_paths_full = Z1.shape[0]

        F = np.zeros((n_paths_full, n_steps+1))
        sigma = np.zeros_like(F)
        F[:,0] = F0
        sigma[:,0] = sigma0

        for i in range(n_steps):
            t = i*dt
            Z2c = rho*Z1[:,i] + np.sqrt(1-rho**2)*Z2[:,i]

            sigma[:,i+1] = sigma[:,i] + a*(b - sigma[:,i])*dt + eta*np.sqrt(dt)*Z2c
            sigma[:,i+1] = np.maximum(sigma[:,i+1], 1e-4)

            jump_mask = UJ[:,i] < jump_lambda*dt
            jumps = (np.exp(jump_mean + jump_std*JY[:,i]) - 1) * jump_mask

            F[:,i+1] = F[:,i] \
                       + kappa*(theta_func(t) - F[:,i])*dt \
                       + sigma[:,i]*F[:,i]*np.sqrt(dt)*Z1[:,i] \
                       + F[:,i]*jumps

        F_T = F[:,-1]
        Y = discount_factor * np.maximum(F_T - K, 0)

        # Black-76 control variate
        F_bs = F0 * np.exp(-0.5*sigma0**2*T + sigma0*np.sqrt(T)*Z1[:,-1])
        X = discount_factor * np.maximum(F_bs - K, 0)

        beta = np.cov(Y, X)[0,1] / np.var(X)
        r = -np.log(discount_factor)/T
        C_black = self.black76_call(F0, K, T, r, sigma0)

        price = Y.mean() - beta*(X.mean() - C_black)
        stderr = Y.std()/np.sqrt(n_paths_full)

        return price, stderr
        
    # -------------------------
    # Compute jump caps from quantiles for normal-crisis regime separation; these are for normal regimes
    # -------------------------
    
    def compute_jump_caps(self, log_returns, window=252,
                          q_std=0.95, q_mean=0.90, q_lambda=0.95):
        """
        Compute regime caps for jump parameters using rolling quantiles.
        Returns caps for normal regime.
        """
    
        sigma0 = log_returns.std()
    
        # Extreme moves
        jump_thresh = 3 * sigma0
        jump_events = log_returns[np.abs(log_returns) > jump_thresh]
    
        if len(jump_events) < 10:
            return {
                "jump_mean_cap": 0.2,
                "jump_std_cap": 0.6,
                "jump_lambda_cap": 0.02
            }
    
        # --- caps ---
        jump_std_cap = np.quantile(np.abs(jump_events), q_std)
        jump_mean_cap = np.quantile(np.abs(jump_events), q_mean)
    
        # rolling jump intensity
        jump_flags = (np.abs(log_returns) > jump_thresh).astype(int)
        rolling_lambda = jump_flags.rolling(window).mean().dropna()
        jump_lambda_cap = np.quantile(rolling_lambda, q_lambda)
    
        return {
            "jump_mean_cap": float(jump_mean_cap),
            "jump_std_cap": float(jump_std_cap),
            "jump_lambda_cap": float(jump_lambda_cap)
        }


    def calibrate_mc_params(self, prices, fwd_proxy, rolling_vol_days=30):
        """
        Calibrate MC parameters:
        - prices: raw daily electricity prices (pd.Series)
        - fwd_proxy: rolling average forward price (for F0)
        """
        # -------------------------
        # Compute log returns on raw prices
        # -------------------------
        prices = prices[prices > 0]
        log_ret_raw = np.log(prices / prices.shift(1)).dropna()
    
        # -------------------------
        # Volatility
        # -------------------------
        sigma0 = log_ret_raw.std()  # initial volatility (daily)
        b = log_ret_raw.rolling(rolling_vol_days).std().mean()  # long-run volatility
    
        # -------------------------
        # Vol-of-vol
        # -------------------------
        vol_series = log_ret_raw.rolling(rolling_vol_days).std().dropna()
        vol_diff = vol_series.diff().dropna()
        vol_diff_aligned, log_ret_aligned = vol_diff.align(log_ret_raw[vol_diff.index], join="inner")
        eta = vol_diff_aligned.std() if len(vol_diff_aligned) > 1 else 0.1
    
        # -------------------------
        # Correlation between price and vol change
        # -------------------------
        rho = np.corrcoef(log_ret_aligned, vol_diff_aligned)[0,1] if len(vol_diff_aligned) > 1 else 0.0
    
        # -------------------------
        # Mean-reversion
        # -------------------------
        phi = log_ret_raw.autocorr(lag=1)
        kappa = -np.log(phi) if phi > 0 else 1.0
        a = -np.log(vol_series.autocorr(lag=1)) if vol_series.autocorr(lag=1) > 0 else 3.0
    
        # -------------------------
        # Jump calibration (extreme moves)
        # -------------------------
        
        jump_threshold = 3 * sigma0
        jump_events = log_ret_raw[np.abs(log_ret_raw) > jump_threshold]
        jump_lambda = len(jump_events) / len(log_ret_raw)
        jump_mean = jump_events.mean() if len(jump_events) > 0 else 0.0
        jump_std = jump_events.std() if len(jump_events) > 0 else 0.1

        # -------------------------
        # Quantile-capped jumps
        # -------------------------
        caps = self.compute_jump_caps(log_ret_raw)
    
        jump_lambda = min(jump_lambda, caps["jump_lambda_cap"])
        jump_mean = np.clip(jump_mean, -caps["jump_mean_cap"], caps["jump_mean_cap"])
        jump_std = min(jump_std, caps["jump_std_cap"])


        # -------------------------
        # Return calibrated parameters
        # -------------------------
        return {
            "sigma0": sigma0,
            "b": b,
            "eta": eta,
            "rho": rho,
            "kappa": kappa,
            "a": a,
            "jump_lambda": jump_lambda,
            "jump_mean": jump_mean,
            "jump_std": jump_std
        }


    # -------------------------
    # Compute seasonal mean-reversion function
    # -------------------------
    
    def compute_theta(self, prices, fwd_proxy, rolling_days=30):
        """
        Returns a function theta(t) in years, capturing weekly + annual seasonality.
        - prices: pd.Series with Date index
        """
        df = prices.copy().to_frame("Price")
        df["DOY"] = df.index.dayofyear
        df["DOW"] = df.index.dayofweek

        # Weekly seasonality (Monday=0 ... Sunday=6)
        weekly = df.groupby("DOW")["Price"].mean()
        # Annual seasonality: smooth daily average over year
        annual = df.groupby("DOY")["Price"].mean().rolling(7, center=True, min_periods=1).mean()

        # Function: combine weekly + annual, scale to F0
        F0 = fwd_proxy.iloc[-1]

        def theta(t):
            """
            t in years
            """
            # map t to day of year (assuming 365 days)
            day_idx = int(t * 365) % 365 + 1  # 1..365
            dow_idx = int(t * 365) % 7
            seasonal_price = annual.get(day_idx, F0) * weekly.get(dow_idx, 1.0) / weekly.mean()
            return seasonal_price

        return theta

    # -------------------------
    # Generate MC paths once per maturity
    # -------------------------
    def generate_paths(self, F0, T, df_disc, params, theta_func=None,
                       n_paths=100_000, n_steps=251):
        dt = T / n_steps
        df_interp = interp1d(df_disc["T"], df_disc["DF"], kind="linear", fill_value="extrapolate")
        discount_factor = float(df_interp(T))
    
        if theta_func is None:
            theta_func = lambda t: F0
    
        Z1, Z2, UJ, JY = self._generate_crn(T, n_paths, n_steps, antithetic=True)
        n_paths_full = Z1.shape[0]
    
        F = np.zeros((n_paths_full, n_steps+1))
        sigma = np.zeros_like(F)
    
        # pathwise derivatives
        dF_dF0 = np.ones((n_paths_full, n_steps+1))
        dF_dsig0 = np.zeros((n_paths_full, n_steps+1))
    
        F[:,0] = F0
        sigma[:,0] = params["sigma0"]
    
        for i in range(n_steps):
            t = i*dt
            Z2c = params["rho"]*Z1[:,i] + np.sqrt(1-params["rho"]**2)*Z2[:,i]
    
            sigma[:,i+1] = sigma[:,i] + params["a"]*(params["b"] - sigma[:,i])*dt \
                           + params["eta"]*np.sqrt(dt)*Z2c
            sigma[:,i+1] = np.maximum(sigma[:,i+1], 1e-4)
    
            jump_mask = UJ[:,i] < params["jump_lambda"]*dt
            jumps = (np.exp(params["jump_mean"] + params["jump_std"]*JY[:,i]) - 1) * jump_mask
    
            drift = params["kappa"]*(theta_func(t) - F[:,i])*dt
            diff = sigma[:,i]*F[:,i]*np.sqrt(dt)*Z1[:,i]
    
            F[:,i+1] = F[:,i] + drift + diff + F[:,i]*jumps
    
            # ---------- Pathwise derivatives ----------
            mult = (1 + sigma[:,i]*np.sqrt(dt)*Z1[:,i] + jumps)
            dF_dF0[:,i+1] = dF_dF0[:,i] * mult
            dF_dsig0[:,i+1] = dF_dsig0[:,i] + F[:,i]*np.sqrt(dt)*Z1[:,i]
    
        return F[:,-1], dF_dF0[:,-1], dF_dsig0[:,-1], discount_factor


    # -------------------------
    # Price options using paths (all strikes from the same paths)
    # -------------------------
    def price_options_from_paths(self, F_T, dF_dF0, dF_dsig0,
                                 discount_factor, strikes):
    
        prices, deltas, vegas = [], [], []
    
        for K in strikes:
            payoff = np.maximum(F_T - K, 0)
            ind = (F_T > K).astype(float)
    
            prices.append(discount_factor * np.mean(payoff))
            deltas.append(discount_factor * np.mean(ind * dF_dF0))
            vegas.append(discount_factor * np.mean(ind * dF_dsig0))
    
        return np.array(prices), np.array(deltas), np.array(vegas)


    # -------------------------
    # Hedged PnL simulation, simple 1st order delta-vega hedge
    # -------------------------

    def hedged_pnl_simulation(
        self, F0, K, T, df_disc, params, theta_func=None,
        n_paths=100_000, n_steps=252, eps=1e-4, include_vega=True
    ):
        """
        Monte Carlo hedged P&L simulation using finite-difference Greeks.
        Uses the current generate_paths which returns terminal F, dF_dF0, dF_dsig0, discount factor.
        
        Returns:
            pnl: np.array of hedged P&L across MC paths
        """
        # Generate terminal forward prices and pathwise derivatives ---
        F_T, dF_dF0, dF_dsig0, discount_factor = self.generate_paths(
            F0, T, df_disc, params, theta_func, n_paths=n_paths, n_steps=n_steps
        )
    
        # Compute option payoff and pathwise Delta/Vega ---
        payoff = np.maximum(F_T - K, 0)
    
        # Pathwise Delta
        delta_pathwise = dF_dF0 * ((F_T - K) > 0)
    
        delta = discount_factor * np.mean(delta_pathwise)
        
        # Pathwise Vega (optional)
        vega = 0.0
        if include_vega:
            vega_pathwise = dF_dsig0 * ((F_T - K) > 0)
            vega = discount_factor * np.mean(vega_pathwise)
    
        # Hedged P&L ---
        cash = discount_factor * np.mean(payoff)
        pnl = cash + (-delta) * F_T  # delta hedge
    
        if include_vega:
            pnl += (-vega) * params["sigma0"]  # simple vega hedge
    
        return pnl
        
    # -------------------------
    # Compute VaR / CVaR, quantiled
    # -------------------------

    def compute_var_cvar(self, pnl, alpha=0.99, trim_extreme=True, trim_quantile=0.0005):
        """
        Compute VaR and CVaR from simulated P&L.
        Optional trimming of extreme tails to avoid single-path explosions.
        """
        pnl_clean = pnl.copy()
        if trim_extreme:
            lower = np.quantile(pnl, trim_quantile)
            upper = np.quantile(pnl, 1 - trim_quantile)
            pnl_clean = pnl[(pnl >= lower) & (pnl <= upper)]
    
        var = np.percentile(pnl_clean, 100*(1-alpha))
        cvar = pnl_clean[pnl_clean <= var].mean()
        return var, cvar
        
    # -------------------------
    # Very simple stress scenario, adding shock factor to forward prices
    # -------------------------

    def stress_scenario(self, F0, K, shock_pct, option_price):
        """
        Deterministic shock: e.g. +200% power spike
        """
        F_shocked = F0 * (1 + shock_pct)
        payoff = max(F_shocked - K, 0)
        pnl = payoff - option_price
        return pnl

    # -------------------------
    # Regime filtering, using Median Absolute Deviation and rolling factor
    # -------------------------

    def filter_normal_crisis(self, fwd_proxy, k_normal=2, k_crisis=3, rolling_window=None, rolling_factor=2.5):
        """
        Robust regime split: normal vs crisis periods.
    
        Combines:
        - Median + MAD (global) to define normal vs extreme
        - Optional rolling median + MAD to catch local spikes
    
        Args:
            fwd_proxy: pd.Series of forward prices
            k_normal: # of MADs above median to consider normal
            k_crisis: # of MADs above median to consider crisis
            rolling_window: int or None, window in days for rolling MAD filter
            rolling_factor: multiplier for rolling MAD (if rolling_window is set)
    
        Returns:
            normal_periods, crisis_periods: pd.DatetimeIndex
        """
        fwd_clean = fwd_proxy.dropna()
        fwd_clean = fwd_clean[fwd_clean > 0]
    
        # ----- Global median + MAD -----
        median = fwd_clean.median()
        mad = (fwd_clean - median).abs().median()
    
        normal_mask = fwd_clean <= median + k_normal*mad
        crisis_mask = fwd_clean >= median + k_crisis*mad
    
        # ----- Optional rolling MAD filter -----
        if rolling_window is not None:
            rolling_median = fwd_clean.rolling(rolling_window).median()
            rolling_mad = fwd_clean.rolling(rolling_window).apply(
                lambda x: np.median(np.abs(x - np.median(x))), raw=True
            )
            rolling_normal = fwd_clean <= rolling_median + rolling_factor*rolling_mad
            rolling_crisis = fwd_clean > rolling_median + rolling_factor*rolling_mad
    
            normal_mask &= rolling_normal
            crisis_mask |= rolling_crisis
    
        normal_periods = fwd_clean[normal_mask].index
        crisis_periods = fwd_clean[crisis_mask].index
    
        return normal_periods, crisis_periods

    # -------------------------
    # Set prices according to regime and compute corresponding forward proxies
    # -------------------------
    
    def compute_forward_proxies_by_regime(self, prices, rolling_days=30,
                                          k_normal=2, k_crisis=3,
                                          rolling_window=30, rolling_factor=2.5):
        """
        Split price series into normal and crisis regimes and compute rolling forward proxies.
        
        Returns:
            fwd_normal, fwd_crisis: pd.Series of forward proxies
            prices_normal, prices_crisis: pd.Series of prices in each regime
        """
        fwd_proxy = prices.rolling(rolling_days).mean().dropna()
        
        # Apply robust filter
        normal_idx, crisis_idx = self.filter_normal_crisis(
            fwd_proxy,
            k_normal=k_normal,
            k_crisis=k_crisis,
            rolling_window=rolling_window,
            rolling_factor=rolling_factor
        )
        
        prices_normal = prices.loc[normal_idx]
        prices_crisis = prices.loc[crisis_idx]
        
        fwd_normal = prices_normal.rolling(rolling_days).mean().dropna()
        fwd_crisis = prices_crisis.rolling(rolling_days).mean().dropna()
        
        return fwd_normal, fwd_crisis, prices_normal, prices_crisis
