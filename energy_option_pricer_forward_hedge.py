from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend, cpu_count
from numpy.random import SeedSequence, default_rng
from scipy.interpolate import interp1d
from scipy.stats import norm, qmc


@dataclass
class CalibrationResult:
    regime: str
    theta_mean: float
    kappa: float
    sigma0: float
    sigma_bar: float
    a: float
    eta: float
    rho: float
    jump_lambda: float
    jump_mu: float
    jump_sigma: float
    # negative-jump parameters (downward spikes)
    neg_jump_lambda: float
    neg_jump_mu: float
    neg_jump_sigma: float
    neg_jump_cap: float
    jump_threshold_z: float
    jump_count: int
    obs_frequency_years: float
    obs_per_day: float
    obs_per_year: float
    spot_bump_scale: float
    sigma_bump_scale: float
    jump_cap: float
    # declared here so dataclass tracks them properly
    sigma_floor: float = 0.0
    sigma_ceiling: float = 1e30


class ElectricitySpotAsianPricer:
    def __init__(self, seed: int = 42, day_count_basis: float = 365.25):
        self.seed = int(seed)
        self.day_count_basis = float(day_count_basis)
        self.rng = default_rng(self.seed)

        self.hourly_: pd.DataFrame | None = None
        self.daily_: pd.DataFrame | None = None
        self.discount_curve_ = None
        self.seasonality_: dict | None = None
        self.calibration_: dict[str, CalibrationResult] = {}
        self.jump_samples_: dict[str, np.ndarray] = {}
        self.futures_: pd.DataFrame | None = None
        self.forward_proxy_: dict | None = None

    # -------------------- utils --------------------
    def _log(self, msg: str, verbose: int | bool = 0):
        if verbose:
            print(msg, flush=True)

    @staticmethod
    def _detect_datetime_col(df: pd.DataFrame, candidates: list[str]) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f"Could not find any datetime column among: {candidates}")

    @staticmethod
    def _drop_duplicate_index(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.DatetimeIndex):
            return df[~df.index.duplicated(keep="last")].sort_index()
        return df

    @staticmethod
    def _parse_volume(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip().upper().replace(",", "")
        if s.endswith("K"):
            return float(s[:-1]) * 1_000.0
        if s.endswith("M"):
            return float(s[:-1]) * 1_000_000.0
        if s.endswith("B"):
            return float(s[:-1]) * 1_000_000_000.0
        return float(s)

    @staticmethod
    def _parse_change_pct(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x).strip().replace("%", "").replace("+", ""))

    @staticmethod
    def _parse_datetime_heuristic(series: pd.Series, dayfirst: bool | None = None) -> pd.DatetimeIndex:
        s = pd.Series(series).astype(str)
        if dayfirst is not None:
            return pd.to_datetime(s, errors="coerce", dayfirst=bool(dayfirst))
        cand_false = pd.to_datetime(s, errors="coerce", dayfirst=False)
        cand_true = pd.to_datetime(s, errors="coerce", dayfirst=True)
        n_false = int(cand_false.notna().sum())
        n_true = int(cand_true.notna().sum())
        if n_true > n_false:
            return cand_true
        if n_false > n_true:
            return cand_false
        # Tie-breaker: prefer the parse with the wider non-null span.
        span_false = cand_false.max() - cand_false.min() if n_false > 0 else pd.Timedelta(0)
        span_true = cand_true.max() - cand_true.min() if n_true > 0 else pd.Timedelta(0)
        return cand_true if span_true > span_false else cand_false

    @staticmethod
    def _normalize_market_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        rename_map = {}
        for c in out.columns:
            cl = c.strip().lower()
            if cl in {"vol.", "vol", "volume"}:
                rename_map[c] = "Volume"
            elif cl in {"change %", "change%", "change_pct", "changepct"}:
                rename_map[c] = "ChangePct"
            elif cl in {"datetime", "date", "timestamp"}:
                rename_map[c] = c
        if rename_map:
            out = out.rename(columns=rename_map)
        return out

    def _resolve_n_jobs(self, n_jobs: int | None) -> int:
        if n_jobs is None:
            return 1
        n_jobs = int(n_jobs)
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        if n_jobs == -1:
            return max(cpu_count(), 1)
        if n_jobs < -1:
            return max(cpu_count() + 1 + n_jobs, 1)
        return max(n_jobs, 1)

    def _observed_dt_years(self, index: pd.DatetimeIndex) -> pd.Series:
        dt_days = index.to_series().diff().dt.total_seconds().div(86400.0)
        return (dt_days / self.day_count_basis).dropna()

    def _default_n_paths(self, use_hourly: bool = False) -> int:
        s = self.price_series_hourly() if (use_hourly and self.hourly_ is not None) else self.price_series_daily()
        dt = self._observed_dt_years(s.index)
        if len(dt) == 0:
            return 1000
        obs_per_day = 1.0 / (float(dt.median()) * self.day_count_basis)
        return max(256, int(np.ceil(len(s) / max(obs_per_day, np.finfo(float).tiny))))

    def _draw_jump(self, p: dict, u: np.ndarray) -> np.ndarray:
        """
        Two-sided jump draw in price units (positive spikes + negative spikes).

        Positive jumps: empirical inverse-CDF from calibrated positive residuals,
        falling back to a lognormal parametric draw.
        Negative jumps: empirical inverse-CDF from calibrated negative residuals
        (stored as positive magnitudes, negated on draw), falling back to a
        normal parametric draw scaled by neg_jump_sigma.

        Each simulated jump event is assigned a sign based on the relative
        intensities lambda_pos / (lambda_pos + lambda_neg).  u is used both
        for the sign draw and for the size quantile, keeping correlation with
        the Poisson indicator uniform u already drawn in the simulation loop.
        """
        regime = p["regime"]
        lam_pos = float(p.get("jump_lambda", 0.0))
        lam_neg = float(p.get("neg_jump_lambda", 0.0))
        lam_tot = lam_pos + lam_neg
        if lam_tot <= 0.0:
            return np.zeros_like(u, dtype=float)

        # Probability that a jump event is a positive spike
        p_pos = lam_pos / lam_tot

        u = np.asarray(u, dtype=float)
        is_pos = u < p_pos
        # Re-scale u within each half to [0,1] for size sampling
        u_pos_size = np.where(is_pos, u / max(p_pos, np.finfo(float).eps), 0.0)
        u_neg_size = np.where(~is_pos, (u - p_pos) / max(1.0 - p_pos, np.finfo(float).eps), 0.0)

        # --- positive jump sizes ---
        pos_samples = self.jump_samples_.get(regime + "_pos", None)
        if pos_samples is None:
            pos_samples = self.jump_samples_.get(regime, None)
        if pos_samples is not None:
            s = np.asarray(pos_samples, dtype=float)
            s = s[np.isfinite(s) & (s > 0)]
        else:
            s = np.array([], dtype=float)

        if len(s) > 0:
            s = np.sort(s)
            idx = np.minimum((u_pos_size * len(s)).astype(int), len(s) - 1)
            pos_jump = s[idx]
            jump_cap = float(p.get("jump_cap", 0.0))
            if np.isfinite(jump_cap) and jump_cap > 0:
                pos_jump = np.clip(pos_jump, 0.0, jump_cap)
        else:
            mu = max(float(p.get("jump_mu", 0.0)), 0.0)
            sigma = max(float(p.get("jump_sigma", 0.0)), 0.0)
            if mu > 0.0 and sigma > 0.0:
                var = sigma ** 2
                sigma_ln = np.sqrt(np.log1p(var / max(mu ** 2, np.finfo(float).tiny)))
                mu_ln = np.log(mu) - 0.5 * sigma_ln ** 2
                z = norm.ppf(np.clip(u_pos_size, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
                pos_jump = np.exp(mu_ln + sigma_ln * z)
            else:
                pos_jump = np.zeros_like(u, dtype=float)
            jump_cap = float(p.get("jump_cap", 0.0))
            if np.isfinite(jump_cap) and jump_cap > 0:
                pos_jump = np.clip(pos_jump, 0.0, jump_cap)
            pos_jump = np.maximum(pos_jump, 0.0)

        # --- negative jump sizes (stored as positive magnitudes) ---
        neg_samples = self.jump_samples_.get(regime + "_neg", None)
        if neg_samples is not None:
            ns = np.asarray(neg_samples, dtype=float)
            ns = ns[np.isfinite(ns) & (ns > 0)]
        else:
            ns = np.array([], dtype=float)

        if len(ns) > 0:
            ns = np.sort(ns)
            idx_n = np.minimum((u_neg_size * len(ns)).astype(int), len(ns) - 1)
            neg_jump = -ns[idx_n]
            neg_cap = float(p.get("neg_jump_cap", 0.0))
            if np.isfinite(neg_cap) and neg_cap > 0:
                neg_jump = np.clip(neg_jump, -neg_cap, 0.0)
        else:
            neg_mu = max(float(p.get("neg_jump_mu", 0.0)), 0.0)
            neg_sigma = max(float(p.get("neg_jump_sigma", 0.0)), 0.0)
            if neg_mu > 0.0 and neg_sigma > 0.0:
                z_n = norm.ppf(np.clip(u_neg_size, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
                neg_jump = -(neg_mu + neg_sigma * z_n)
                neg_jump = np.minimum(neg_jump, 0.0)
            else:
                neg_jump = np.zeros_like(u, dtype=float)

        return np.where(is_pos, pos_jump, neg_jump)

    @staticmethod
    def _robust_mad(x: pd.Series | np.ndarray) -> float:
        arr = np.asarray(x, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return 0.0
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return 1.4826 * mad

    @staticmethod
    def _delivery_mask_from_grid(tgrid: np.ndarray, delivery_start: float, delivery_end: float) -> np.ndarray:
        tgrid = np.asarray(tgrid, dtype=float)
        return (tgrid >= float(delivery_start)) & (tgrid <= float(delivery_end))

    @staticmethod
    def _future_delivery_window(now_t: float, horizon_T: float, delivery_start: float, delivery_end: float) -> tuple[float, float]:
        now_t = float(now_t)
        horizon_T = float(horizon_T)
        d0 = max(float(delivery_start) - now_t, 0.0)
        d1 = min(float(delivery_end), horizon_T) - now_t
        d1 = max(d1, 0.0)
        return d0, d1

    def _effective_window(self, phi: float, min_window: int = 5) -> int:
        phi = float(np.clip(phi, 1e-8, 1.0 - 1e-8))
        eff = 1.0 / max(1.0 - phi, np.finfo(float).tiny)
        return max(int(np.ceil(eff)), int(min_window))

    def _expected_average_price(
        self,
        T: float,
        regime: str = "all",
        use_hourly: bool = False,
        S0: float | None = None,
        sigma0_override: float | None = None,
        delivery_start: float = 0.0,
        delivery_end: float | None = None,
        n_steps: int | None = None,
    ) -> float:
        if delivery_end is None:
            delivery_end = T
        if S0 is None:
            S0 = self.current_price
        p = self.params(regime=regime, use_hourly=use_hourly)
        n_steps = self._infer_n_steps(T, use_hourly=use_hourly, n_steps=n_steps)
        dt = T / n_steps
        step_days = T * self.day_count_basis / n_steps
        ts = pd.date_range(start=self.current_timestamp, periods=n_steps + 1, freq=pd.to_timedelta(step_days, unit="D"))
        theta_grid = self.theta(ts)

        jump_mean = float(p["jump_mu"])
        samples = self.jump_samples_.get(regime, None)
        if samples is not None:
            s = np.asarray(samples, dtype=float)
            s = s[np.isfinite(s)]
            s = s[s > 0]
            if len(s) > 0:
                jump_mean = float(np.mean(s))

        S = float(S0)
        exp_path = np.zeros(n_steps + 1, dtype=float)
        exp_path[0] = S
        for i in range(n_steps):
            theta_i = float(theta_grid[i])
            if p["kappa"] > 0:
                S = theta_i + (S - theta_i) * np.exp(-p["kappa"] * dt) + p["jump_lambda"] * dt * jump_mean
            else:
                S = S + p["kappa"] * (theta_i - S) * dt + p["jump_lambda"] * dt * jump_mean
            exp_path[i + 1] = S

        tgrid = np.linspace(0.0, T, n_steps + 1)
        mask = self._delivery_mask_from_grid(tgrid[1:], delivery_start, delivery_end)
        if not np.any(mask):
            raise ValueError("Delivery window is empty on the simulation grid.")
        return float(exp_path[1:][mask].mean())

    # -------------------- loading --------------------
    def load_hourly_csv(
        self,
        csv_path: str,
        country_iso: str | None = None,
        timezone_col_priority: tuple[str, ...] = ("Datetime (Local)", "Datetime (UTC)"),
        country_col: str = "Country",
        iso_col: str = "ISO3 Code",
        price_col: str = "Price (EUR/MWhe)",
        parse_extra_cols: bool = True,
    ) -> pd.DataFrame:
        df = pd.read_csv(csv_path, sep=None, engine="python")
        dt_col = self._detect_datetime_col(df, list(timezone_col_priority))
        if price_col not in df.columns:
            raise ValueError(f"Missing required price column '{price_col}'.")

        if country_iso is not None and iso_col in df.columns:
            df = df[df[iso_col] == country_iso].copy()
        elif country_iso is not None and country_col in df.columns and iso_col not in df.columns:
            df = df[df[country_col].astype(str).str.upper() == str(country_iso).upper()].copy()

        df = df.copy()
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col])
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        df = df.dropna(subset=[price_col])

        if parse_extra_cols:
            if "Vol" in df.columns:
                df["Vol"] = df["Vol"].apply(self._parse_volume)
            if "Change %" in df.columns:
                df["ChangePct"] = df["Change %"].apply(self._parse_change_pct)
            elif "ChangePct" in df.columns:
                df["ChangePct"] = df["ChangePct"].apply(self._parse_change_pct)

        df = df.rename(columns={dt_col: "Datetime", price_col: "Price"}).set_index("Datetime")
        df = self._drop_duplicate_index(df)
        self.hourly_ = df
        self.daily_ = None
        self.seasonality_ = None
        self.calibration_.clear()
        return df

    def load_daily_csv(
        self,
        csv_path: str,
        country_iso: str | None = None,
        country_col: str = "Country",
        iso_col: str = "ISO3 Code",
        date_col: str = "Date",
        price_col: str = "Price (EUR/MWhe)",
    ) -> pd.DataFrame:
        df = pd.read_csv(csv_path, sep=None, engine="python")
        if date_col not in df.columns or price_col not in df.columns:
            raise ValueError("Missing required date or price column.")

        if country_iso is not None and iso_col in df.columns:
            df = df[df[iso_col] == country_iso].copy()
        elif country_iso is not None and country_col in df.columns and iso_col not in df.columns:
            df = df[df[country_col].astype(str).str.upper() == str(country_iso).upper()].copy()

        df = df.copy()
        df[date_col] = self._parse_datetime_heuristic(df[date_col])
        df = df.dropna(subset=[date_col])
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        df = df.dropna(subset=[price_col])
        df = df.rename(columns={date_col: "Date", price_col: "Price"}).set_index("Date")
        df = self._drop_duplicate_index(df)
        self.daily_ = df
        return df

    def load_futures_csv(
        self,
        csv_path: str,
        date_col: str = "Date",
        price_col: str = "Price",
        open_col: str = "Open",
        high_col: str = "High",
        low_col: str = "Low",
        volume_col: str = "Vol.",
        change_pct_col: str = "Change %",
        dayfirst: bool | None = None,
        keep_raw_columns: bool = True,
    ) -> pd.DataFrame:
        """Load a daily futures / forward CSV with columns like:
        Date, Price, Open, High, Low, Vol., Change %
        """
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df = self._normalize_market_dataframe(df)

        if date_col not in df.columns or price_col not in df.columns:
            raise ValueError(f"Missing required futures columns '{date_col}' or '{price_col}'.")

        out = df.copy()
        out[date_col] = self._parse_datetime_heuristic(out[date_col], dayfirst=dayfirst)
        out = out.dropna(subset=[date_col])
        out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
        out = out.dropna(subset=[price_col])

        for c in [open_col, high_col, low_col]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        if "Volume" in out.columns:
            out["Volume"] = out["Volume"].apply(self._parse_volume)
        elif volume_col in out.columns:
            out["Volume"] = out[volume_col].apply(self._parse_volume)

        if "ChangePct" in out.columns:
            out["ChangePct"] = out["ChangePct"].apply(self._parse_change_pct)
        elif change_pct_col in out.columns:
            out["ChangePct"] = out[change_pct_col].apply(self._parse_change_pct)

        cols = [c for c in [price_col, open_col, high_col, low_col, "Volume", "ChangePct"] if c in out.columns]
        if keep_raw_columns:
            # keep all original columns as well, but ensure the standard ones exist
            standard = out[[date_col] + cols].copy()
            out = out.copy()
            out = out.rename(columns={date_col: "Date", price_col: "Price"})
            out = out.set_index("Date")
            if not keep_raw_columns:
                out = out[[c for c in ["Price", open_col, high_col, low_col, "Volume", "ChangePct"] if c in out.columns]]
        else:
            out = out.rename(columns={date_col: "Date", price_col: "Price"})
            out = out.set_index("Date")
            keep = [c for c in ["Price", open_col, high_col, low_col, "Volume", "ChangePct"] if c in out.columns]
            out = out[keep]

        out = self._drop_duplicate_index(out)
        out = out.sort_index()
        self.futures_ = out
        self.forward_proxy_ = None
        return out

    def load_forward_csv(self, *args, **kwargs) -> pd.DataFrame:
        """Alias for load_futures_csv() for terminology consistency."""
        return self.load_futures_csv(*args, **kwargs)

    # -------------------- accessors --------------------
    @property
    def hourly(self) -> pd.DataFrame:
        if self.hourly_ is None:
            raise ValueError("No hourly data loaded.")
        return self.hourly_

    @property
    def daily(self) -> pd.DataFrame:
        if self.daily_ is None:
            if self.hourly_ is None:
                raise ValueError("No data loaded.")
            self.daily_ = self.aggregate_hourly_to_daily(self.hourly_)
        return self.daily_

    @property
    def current_price(self) -> float:
        if self.hourly_ is not None:
            return float(self.hourly_["Price"].iloc[-1])
        if self.daily_ is not None:
            return float(self.daily_["Price"].iloc[-1])
        raise ValueError("No data loaded.")

    @property
    def current_timestamp(self) -> pd.Timestamp:
        if self.hourly_ is not None:
            return pd.Timestamp(self.hourly_.index[-1])
        if self.daily_ is not None:
            return pd.Timestamp(self.daily_.index[-1])
        raise ValueError("No data loaded.")

    def price_series_hourly(self) -> pd.Series:
        return self.hourly["Price"].astype(float).copy()

    def price_series_daily(self) -> pd.Series:
        return self.daily["Price"].astype(float).copy()

    def aggregate_hourly_to_daily(self, hourly_df: pd.DataFrame | None = None) -> pd.DataFrame:
        df = self.hourly if hourly_df is None else hourly_df
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Hourly data must be indexed by DatetimeIndex.")
        out = df.copy()
        out["Date"] = out.index.floor("D")
        out = out.groupby("Date", sort=True).agg({"Price": "mean"})
        return self._drop_duplicate_index(out)

    def diagnostics(self, use_hourly: bool = True) -> pd.DataFrame:
        s = self.price_series_hourly() if (use_hourly and self.hourly_ is not None) else self.price_series_daily()
        if len(s) < 2:
            raise ValueError("Need at least two observations.")
        if (s > 0).all():
            r = np.log(s / s.shift(1)).dropna()
        else:
            scale = float(np.abs(s).median()) if float(np.abs(s).median()) > 0 else 1.0
            r = (s.diff() / scale).dropna()
        out = {
            "obs_count": len(s),
            "start": s.index.min(),
            "end": s.index.max(),
            "min_price": float(s.min()),
            "max_price": float(s.max()),
            "mean_price": float(s.mean()),
            "median_price": float(s.median()),
            "realized_vol_annualized": float(self.realized_volatility(use_hourly=use_hourly)),
            "mean_log_return": float(r.mean()) if len(r) else np.nan,
            "std_log_return": float(r.std(ddof=1)) if len(r) > 1 else np.nan,
        }
        if self.calibration_:
            out.update(list(self.calibration_.values())[-1].__dict__)
        return pd.DataFrame([out])

    def realized_volatility(self, use_hourly: bool = True, annualize: bool = True) -> float:
        s = self.price_series_hourly() if (use_hourly and self.hourly_ is not None) else self.price_series_daily()
        if (s > 0).all():
            x = np.log(s / s.shift(1)).dropna()
        else:
            scale = float(np.abs(s).median()) if float(np.abs(s).median()) > 0 else 1.0
            x = (s.diff() / scale).dropna()
        dt = self._observed_dt_years(s.index).loc[x.index]
        if len(x) < 2:
            raise ValueError("Not enough observations to compute volatility.")
        sigma_sq = np.sum((x - x.mean()).to_numpy() ** 2) / np.sum(dt.to_numpy())
        sigma = float(np.sqrt(max(sigma_sq, 0.0)))
        return sigma if annualize else sigma * np.sqrt(float(dt.median()))

    # -------------------- seasonality --------------------
    def fit_seasonality(self, use_hourly: bool = True):
        if use_hourly:
            s = self.price_series_hourly()
            idx = s.index
            overall = float(s.mean())
            hour_means = s.groupby(idx.hour).mean().reindex(range(24)).interpolate(limit_direction="both").fillna(overall)
            dow_means = s.groupby(idx.dayofweek).mean().reindex(range(7)).interpolate(limit_direction="both").fillna(overall)
            doy_means = s.groupby(idx.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both").fillna(overall)
            self.seasonality_ = {
                "mode": "hourly",
                "overall": overall,
                "hour_dev": (hour_means - overall).to_numpy(),
                "dow_dev": (dow_means - overall).to_numpy(),
                "doy_dev": (doy_means - overall).to_numpy(),
            }
        else:
            s = self.price_series_daily() if self.daily_ is not None else self.aggregate_hourly_to_daily()["Price"]
            idx = s.index
            overall = float(s.mean())
            dow_means = s.groupby(idx.dayofweek).mean().reindex(range(7)).interpolate(limit_direction="both").fillna(overall)
            doy_means = s.groupby(idx.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both").fillna(overall)
            self.seasonality_ = {
                "mode": "daily",
                "overall": overall,
                "dow_dev": (dow_means - overall).to_numpy(),
                "doy_dev": (doy_means - overall).to_numpy(),
            }
        return self.seasonality_

    def theta(self, timestamps) -> np.ndarray:
        if self.seasonality_ is None:
            self.fit_seasonality(use_hourly=self.hourly_ is not None)
        ts = pd.DatetimeIndex(timestamps)
        overall = self.seasonality_["overall"]
        dow = ts.dayofweek.to_numpy()
        doy = ((ts.dayofyear.to_numpy() - 1) % 366).astype(int)
        if self.seasonality_["mode"] == "hourly":
            hour = ts.hour.to_numpy()
            return overall + self.seasonality_["hour_dev"][hour] + self.seasonality_["dow_dev"][dow] + self.seasonality_["doy_dev"][doy]
        return overall + self.seasonality_["dow_dev"][dow] + self.seasonality_["doy_dev"][doy]

    def deseasonalized_series(self, use_hourly: bool = True) -> pd.Series:
        s = self.price_series_hourly() if use_hourly else self.price_series_daily()
        return s - pd.Series(self.theta(s.index), index=s.index)

    # -------------------- calibration --------------------
    def _fit_ar1_residual_model(self, x: pd.Series):
        x = pd.Series(x).dropna().copy()
        x = x[~x.index.duplicated(keep="last")].sort_index()
        if len(x) < 50:
            raise ValueError("Not enough observations for AR(1) calibration.")

        dt_seconds = x.index.to_series().diff().dt.total_seconds().dropna()
        if len(dt_seconds) == 0:
            raise ValueError("Cannot infer observation frequency from timestamps.")
        median_dt_years = float((dt_seconds.median() / 86400.0) / self.day_count_basis)
        if median_dt_years <= 0:
            raise ValueError("Invalid inferred time step.")

        df = pd.concat([x.rename("x_now"), x.shift(1).rename("x_lag")], axis=1).dropna()
        if len(df) < 30:
            raise ValueError("Not enough contiguous observations for AR(1) calibration.")
        y = df["x_now"].to_numpy()
        X = np.column_stack([np.ones(len(df)), df["x_lag"].to_numpy()])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        c_ar = float(coef[0])
        phi = float(np.clip(float(coef[1]), 1e-8, 1 - 1e-8))
        kappa = float(-np.log(phi) / median_dt_years)
        resid = df["x_now"] - (c_ar + phi * df["x_lag"])
        return {"c_ar": c_ar, "phi": phi, "kappa": kappa, "resid": resid, "median_dt_years": median_dt_years}

    def split_regimes(self, use_hourly: bool = True):
        x = self.deseasonalized_series(use_hourly=use_hourly).dropna().sort_index()
        ar = self._fit_ar1_residual_model(x)
        resid = ar["resid"].dropna().sort_index()
        phi = float(ar["phi"])
        window = self._effective_window(phi)

        local_sigma = resid.rolling(window, min_periods=max(3, window // 3)).apply(self._robust_mad, raw=False)
        local_sigma = local_sigma.replace([np.inf, -np.inf], np.nan).reindex(resid.index)
        local_sigma = local_sigma.interpolate(limit_direction="both").bfill().ffill()
        z = (resid / local_sigma).replace([np.inf, -np.inf], np.nan).dropna()
        z_cut = float(np.sqrt(2.0 * np.log(max(len(z), 2))))

        return {
            "normal_idx": z.index[np.abs(z) <= z_cut],
            "crisis_idx": z.index[np.abs(z) > z_cut],
            "z_cut": z_cut,
            "window": int(window),
        }

    def calibrate(self, regime: str = "all", use_hourly: bool = True, verbose: int = 0) -> CalibrationResult:
        if regime not in {"all", "normal", "crisis"}:
            raise ValueError("regime must be 'all', 'normal', or 'crisis'.")
        self._log(f"[CAL] started | regime={regime}", verbose)

        self.fit_seasonality(use_hourly=use_hourly and self.hourly_ is not None)
        s = self.price_series_hourly() if (use_hourly and self.hourly_ is not None) else self.price_series_daily()

        if regime == "all":
            sample = s
        else:
            split = self.split_regimes(use_hourly=use_hourly)
            sample = s.loc[split["normal_idx"] if regime == "normal" else split["crisis_idx"]]

        sample = sample.dropna().sort_index()
        if len(sample) < 50:
            raise ValueError("Not enough data for calibration.")

        # ----------------------------------------------------------------
        # Kappa estimation: always use daily aggregated data.
        # AR(1) on hourly data measures intra-day autocorrelation, not the
        # multi-day OU mean-reversion speed.  Aggregating to daily before
        # fitting kappa gives an economically meaningful half-life.
        # ----------------------------------------------------------------
        if use_hourly and self.hourly_ is not None:
            s_daily = self.aggregate_hourly_to_daily(self.hourly_)["Price"].astype(float).dropna()
        else:
            s_daily = sample.copy()

        theta_daily = pd.Series(self.theta(s_daily.index), index=s_daily.index)
        x_daily = (s_daily - theta_daily).dropna()

        # Pass 1 on daily: rough jump detection to clean kappa estimate
        ar_rough = self._fit_ar1_residual_model(x_daily)
        resid_rough = ar_rough["resid"].dropna().sort_index()
        window_rough = self._effective_window(ar_rough["phi"])
        local_scale_rough = resid_rough.rolling(window_rough, min_periods=max(3, window_rough // 3)).apply(
            self._robust_mad, raw=False
        )
        local_scale_rough = local_scale_rough.replace([np.inf, -np.inf], np.nan).reindex(resid_rough.index)
        local_scale_rough = local_scale_rough.interpolate(limit_direction="both").bfill().ffill()
        z_rough = (resid_rough / local_scale_rough).replace([np.inf, -np.inf], np.nan).dropna()
        z_cut_rough = float(np.sqrt(2.0 * np.log(max(len(z_rough), 2))))
        clean_idx_rough = z_rough.index[np.abs(z_rough) <= z_cut_rough]
        x_daily_clean = x_daily.loc[x_daily.index.intersection(clean_idx_rough)] if len(clean_idx_rough) >= 50 else x_daily

        # Pass 2 on daily: unbiased kappa from jump-cleaned data
        ar_daily = self._fit_ar1_residual_model(x_daily_clean)
        kappa_final = float(ar_daily["kappa"])

        # med_dt for obs_frequency_years stays on the original (hourly) sample
        theta_sample = pd.Series(self.theta(sample.index), index=sample.index)
        x = sample - theta_sample

        ar_hourly = self._fit_ar1_residual_model(x)
        med_dt = ar_hourly["median_dt_years"]
        obs_per_year = 1.0 / med_dt
        obs_per_day = obs_per_year / self.day_count_basis

        # ----------------------------------------------------------------
        # Sigma / vol-of-vol / jump calibration: use full-frequency sample
        # Re-compute residuals using daily kappa phi translated to hourly dt
        # ----------------------------------------------------------------
        phi_hourly = float(np.exp(-kappa_final * med_dt))
        phi_hourly = float(np.clip(phi_hourly, 1e-10, 1.0 - 1e-10))
        # Intercept: theta absorption at hourly scale
        c_ar_hourly = float(ar_hourly["c_ar"])  # use hourly-fitted intercept for residuals

        df_full = pd.concat([x.rename("x_now"), x.shift(1).rename("x_lag")], axis=1).dropna()
        resid_full = df_full["x_now"] - (c_ar_hourly + phi_hourly * df_full["x_lag"])

        window = self._effective_window(phi_hourly)
        local_scale = resid_full.rolling(window, min_periods=max(3, window // 3)).apply(self._robust_mad, raw=False)
        local_scale = local_scale.replace([np.inf, -np.inf], np.nan).reindex(resid_full.index)
        local_scale = local_scale.interpolate(limit_direction="both").bfill().ffill()
        sigma_series = local_scale / np.sqrt(med_dt)
        if len(sigma_series) < 3:
            raise ValueError("Not enough data to calibrate stochastic volatility.")

        z = (resid_full / local_scale).replace([np.inf, -np.inf], np.nan).dropna()
        z_cut = float(np.sqrt(2.0 * np.log(max(len(z), 2))))
        clean_idx = z.index[np.abs(z) <= z_cut]
        if len(clean_idx) < 10:
            clean_idx = sigma_series.index

        sigma_clean = sigma_series.loc[clean_idx].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sigma_clean) < 3:
            raise ValueError("Not enough clean observations to calibrate stochastic volatility.")

        sigma0 = float(sigma_clean.iloc[-1])
        sigma_bar = float(sigma_clean.median())
        sigma_bump_scale = float(sigma_clean.diff().abs().dropna().median())
        if not np.isfinite(sigma_bump_scale) or sigma_bump_scale <= 0:
            sigma_bump_scale = float(sigma_clean.std(ddof=1))

        log_sigma = np.log(np.maximum(sigma_clean, np.finfo(float).tiny))
        log_lag = log_sigma.shift(1).dropna()
        log_now = log_sigma.loc[log_lag.index]
        B = np.column_stack([np.ones(len(log_lag)), log_lag.to_numpy()])
        coef_s, *_ = np.linalg.lstsq(B, log_now.to_numpy(), rcond=None)
        c_s, phi_s = float(coef_s[0]), float(np.clip(float(coef_s[1]), 1e-8, 1 - 1e-8))
        a = float(-np.log(phi_s) / med_dt)
        m_sigma = float(c_s / (1.0 - phi_s))
        sigma_bar = float(np.exp(m_sigma))
        vol_resid = log_now - (c_s + phi_s * log_lag)
        eta = float(vol_resid.std(ddof=1) / np.sqrt(med_dt)) if len(vol_resid) > 1 else 0.0

        common_idx = resid_full.index.intersection(vol_resid.index)
        rho = float(np.corrcoef(resid_full.loc[common_idx], vol_resid.loc[common_idx])[0, 1]) if len(common_idx) > 2 else 0.0
        rho = float(np.clip(rho, -0.999, 0.999))

        # ---- Two-sided jump calibration ----
        pos_jump_idx = z.index[z > z_cut]
        neg_jump_idx = z.index[z < -z_cut]
        pos_jump_events = resid_full.loc[pos_jump_idx].clip(lower=0.0)
        neg_jump_events = resid_full.loc[neg_jump_idx].clip(upper=0.0).abs()

        total_years = float((sample.index.max() - sample.index.min()).total_seconds() / 86400.0 / self.day_count_basis)
        if total_years <= 0:
            total_years = float(len(sample) * med_dt)

        if len(pos_jump_events) > 0:
            jump_lambda = float(len(pos_jump_events) / total_years)
            jump_mu = float(max(pos_jump_events.mean(), 0.0))
            jump_sigma = float(pos_jump_events.std(ddof=1)) if len(pos_jump_events) > 1 else 0.0
            if not np.isfinite(jump_sigma) or jump_sigma <= 0:
                jump_sigma = float(self._robust_mad(pos_jump_events))
            jump_cap = float(pos_jump_events.quantile(0.995))
            self.jump_samples_[regime + "_pos"] = pos_jump_events.to_numpy(dtype=float)
            self.jump_samples_[regime] = pos_jump_events.to_numpy(dtype=float)  # backward-compat
        else:
            jump_lambda = 0.0
            jump_mu = 0.0
            jump_sigma = 0.0
            jump_cap = 0.0
            self.jump_samples_[regime + "_pos"] = np.array([], dtype=float)
            self.jump_samples_[regime] = np.array([], dtype=float)

        if len(neg_jump_events) > 0:
            neg_jump_lambda = float(len(neg_jump_events) / total_years)
            neg_jump_mu = float(max(neg_jump_events.mean(), 0.0))
            neg_jump_sigma = float(neg_jump_events.std(ddof=1)) if len(neg_jump_events) > 1 else 0.0
            if not np.isfinite(neg_jump_sigma) or neg_jump_sigma <= 0:
                neg_jump_sigma = float(self._robust_mad(neg_jump_events))
            neg_jump_cap = float(neg_jump_events.quantile(0.995))
            self.jump_samples_[regime + "_neg"] = neg_jump_events.to_numpy(dtype=float)
        else:
            neg_jump_lambda = 0.0
            neg_jump_mu = 0.0
            neg_jump_sigma = 0.0
            neg_jump_cap = 0.0
            self.jump_samples_[regime + "_neg"] = np.array([], dtype=float)

        jump_count = int(len(pos_jump_events) + len(neg_jump_events))

        spot_bump_scale = float(sample.diff().abs().median())
        if not np.isfinite(spot_bump_scale) or spot_bump_scale <= 0:
            spot_bump_scale = float(sample.diff().abs().std(ddof=1))
        if not np.isfinite(spot_bump_scale) or spot_bump_scale <= 0:
            spot_bump_scale = float(sample.std(ddof=1))
        if not np.isfinite(spot_bump_scale) or spot_bump_scale <= 0:
            spot_bump_scale = max(float(np.abs(sample).median()), 1e-8)

        result = CalibrationResult(
            regime=regime,
            theta_mean=float(self.seasonality_["overall"]),
            kappa=kappa_final,
            sigma0=float(sigma0),
            sigma_bar=float(sigma_bar),
            a=float(a),
            eta=float(eta),
            rho=float(rho),
            jump_lambda=float(jump_lambda),
            jump_mu=float(jump_mu),
            jump_sigma=float(jump_sigma),
            neg_jump_lambda=float(neg_jump_lambda),
            neg_jump_mu=float(neg_jump_mu),
            neg_jump_sigma=float(neg_jump_sigma),
            neg_jump_cap=float(neg_jump_cap),
            jump_threshold_z=float(z_cut),
            jump_count=int(jump_count),
            obs_frequency_years=float(med_dt),
            obs_per_day=float(obs_per_day),
            obs_per_year=float(obs_per_year),
            spot_bump_scale=float(spot_bump_scale),
            sigma_bump_scale=float(sigma_bump_scale),
            jump_cap=float(jump_cap),
            sigma_floor=float(max(float(sigma_clean.min()), np.finfo(float).tiny)),
            sigma_ceiling=float(max(float(sigma_clean.max()), float(max(float(sigma_clean.min()), np.finfo(float).tiny)))),
        )
        self.calibration_[regime] = result
        self._log(
            f"[CAL] finished | kappa={result.kappa:.4f} | sigma0={result.sigma0:.4f} | z_cut={z_cut:.3f} | jump_count={result.jump_count}",
            verbose,
        )
        return result

    def params(self, regime: str = "all", use_hourly: bool = True, verbose: int = 0) -> dict:
        if regime in self.calibration_:
            return self.calibration_[regime].__dict__
        return self.calibrate(regime=regime, use_hourly=use_hourly, verbose=verbose).__dict__

    # -------------------- discount curve --------------------
    def set_discount_curve(
        self,
        curve: pd.DataFrame,
        maturity_col: str = "T",
        df_col: str = "DF",
        zero_rate_col: str | None = None,
    ):
        c = curve.copy().sort_values(maturity_col)
        t = c[maturity_col].astype(float).to_numpy()
        if zero_rate_col is not None:
            if zero_rate_col not in c.columns:
                raise ValueError(f"Missing zero-rate column '{zero_rate_col}'.")
            dfs = np.exp(-c[zero_rate_col].astype(float).to_numpy() * t)
        else:
            if df_col not in c.columns:
                raise ValueError(f"Missing discount-factor column '{df_col}'.")
            dfs = c[df_col].astype(float).to_numpy()
        self.discount_curve_ = interp1d(t, dfs, kind="linear", fill_value="extrapolate", assume_sorted=True)
        return self

    def discount_factor(self, T: float) -> float:
        if self.discount_curve_ is None:
            return 1.0
        return float(self.discount_curve_(float(T)))

    # -------------------- simulation --------------------
    def _random_bundle(self, n_paths: int, n_steps: int, antithetic: bool = True, seed: int | None = None):
        rng = default_rng(self.seed if seed is None else seed)
        z1 = rng.standard_normal((n_paths, n_steps))
        z2 = rng.standard_normal((n_paths, n_steps))
        uj = rng.uniform(size=(n_paths, n_steps))
        zj = rng.standard_normal((n_paths, n_steps))
        if antithetic:
            z1 = np.vstack([z1, -z1])
            z2 = np.vstack([z2, -z2])
            uj = np.vstack([uj, 1.0 - uj])
            zj = np.vstack([zj, -zj])
        return {"z1": z1, "z2": z2, "uj": uj, "zj": zj}

    def _infer_n_steps(self, T: float, use_hourly: bool = False, n_steps: int | None = None):
        if n_steps is not None:
            return int(n_steps)
        s = self.price_series_hourly() if (use_hourly and self.hourly_ is not None) else self.price_series_daily()
        dt = self._observed_dt_years(s.index)
        if len(dt) == 0:
            raise ValueError("Need loaded data to infer time steps.")
        return max(1, int(np.ceil(T / float(dt.median()))))

    def _simulate_paths_core(
        self,
        T: float,
        regime: str = "all",
        use_hourly: bool = False,
        n_paths: int | None = None,
        n_steps: int | None = None,
        antithetic: bool = True,
        S0: float | None = None,
        sigma0_override: float | None = None,
        seed: int | None = None,
    ):
        p = self.params(regime=regime, use_hourly=use_hourly)
        if S0 is None:
            S0 = self.current_price
        if n_paths is None:
            n_paths = self._default_n_paths(use_hourly=use_hourly)
        n_steps = self._infer_n_steps(T, use_hourly=use_hourly, n_steps=n_steps)
        dt = T / n_steps

        bundle = self._random_bundle(n_paths, n_steps, antithetic=antithetic, seed=seed)
        z1, z2, uj, zj = bundle["z1"], bundle["z2"], bundle["uj"], bundle["zj"]
        zsigma = p["rho"] * z1 + np.sqrt(1.0 - p["rho"] ** 2) * z2

        step_days = T * self.day_count_basis / n_steps
        ts = pd.date_range(start=self.current_timestamp, periods=n_steps + 1, freq=pd.to_timedelta(step_days, unit="D"))
        theta_grid = self.theta(ts)

        paths = np.zeros((z1.shape[0], n_steps + 1), dtype=float)
        sigmas = np.zeros_like(paths)
        paths[:, 0] = S0
        sigmas[:, 0] = p["sigma0"] if sigma0_override is None else float(sigma0_override)
        jump_prob = 1.0 - np.exp(-max(p["jump_lambda"] + p.get("neg_jump_lambda", 0.0), 0.0) * dt)
        kappa = float(max(p["kappa"], 0.0))
        exp_kdt = np.exp(-kappa * dt) if kappa > 0 else 1.0
        ou_scale = np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)) if kappa > 0 else np.sqrt(dt)
        sigma_floor = float(p.get("sigma_floor", np.finfo(float).tiny))
        sigma_ceiling = float(p.get("sigma_ceiling", np.finfo(float).max))
        log_floor = np.log(max(sigma_floor, np.finfo(float).tiny))
        log_ceiling = np.log(max(sigma_ceiling, sigma_floor * (1.0 + np.finfo(float).eps)))

        for i in range(n_steps):
            sigma_prev = np.clip(sigmas[:, i], sigma_floor, sigma_ceiling)
            log_sigma_next = (
                np.log(sigma_prev)
                + p["a"] * (np.log(np.maximum(p["sigma_bar"], np.finfo(float).tiny)) - np.log(sigma_prev)) * dt
                + p["eta"] * np.sqrt(dt) * zsigma[:, i]
            )
            log_sigma_next = np.clip(log_sigma_next, log_floor, log_ceiling)
            sigmas[:, i + 1] = np.clip(np.exp(log_sigma_next), sigma_floor, sigma_ceiling)
            u_jump = norm.cdf(zj[:, i])
            jump = (uj[:, i] < jump_prob).astype(float) * self._draw_jump(p, u_jump)
            if kappa > 0:
                paths[:, i + 1] = theta_grid[i] + (paths[:, i] - theta_grid[i]) * exp_kdt + sigma_prev * ou_scale * z1[:, i] + jump
            else:
                paths[:, i + 1] = paths[:, i] + p["kappa"] * (theta_grid[i] - paths[:, i]) * dt + sigma_prev * ou_scale * z1[:, i] + jump
        return ts, paths, sigmas, p

    def _payoff_batch(
        self,
        batch_paths: int,
        T: float,
        K: float,
        regime: str,
        use_hourly: bool,
        n_steps: int,
        antithetic: bool,
        S0: float,
        sigma0_override: float | None,
        delivery_start: float,
        delivery_end: float,
        seed: int,
    ):
        p = self.params(regime=regime, use_hourly=use_hourly)
        dt = T / n_steps
        bundle = self._random_bundle(batch_paths, n_steps, antithetic=antithetic, seed=seed)
        z1, z2, uj, zj = bundle["z1"], bundle["z2"], bundle["uj"], bundle["zj"]
        zsigma = p["rho"] * z1 + np.sqrt(1.0 - p["rho"] ** 2) * z2

        step_days = T * self.day_count_basis / n_steps
        ts = pd.date_range(start=self.current_timestamp, periods=n_steps + 1, freq=pd.to_timedelta(step_days, unit="D"))
        theta_grid = self.theta(ts)

        S = np.full(z1.shape[0], S0, dtype=float)
        sigma = np.full(z1.shape[0], p["sigma0"] if sigma0_override is None else float(sigma0_override), dtype=float)
        tgrid = np.linspace(0.0, T, n_steps + 1)
        in_window = (tgrid[1:] >= delivery_start) & (tgrid[1:] <= delivery_end)
        if int(in_window.sum()) < 1:
            raise ValueError("Delivery window is empty on the simulation grid.")
        window_sum = np.zeros_like(S)
        jump_prob = 1.0 - np.exp(-max(p["jump_lambda"] + p.get("neg_jump_lambda", 0.0), 0.0) * dt)
        kappa = float(max(p["kappa"], 0.0))
        exp_kdt = np.exp(-kappa * dt) if kappa > 0 else 1.0
        ou_scale = np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)) if kappa > 0 else np.sqrt(dt)
        sigma_floor = float(p.get("sigma_floor", np.finfo(float).tiny))
        sigma_ceiling = float(p.get("sigma_ceiling", np.finfo(float).max))
        log_floor = np.log(max(sigma_floor, np.finfo(float).tiny))
        log_ceiling = np.log(max(sigma_ceiling, sigma_floor * (1.0 + np.finfo(float).eps)))

        for i in range(n_steps):
            sigma_prev = np.clip(sigma, sigma_floor, sigma_ceiling)
            log_sigma_next = (
                np.log(sigma_prev)
                + p["a"] * (np.log(np.maximum(p["sigma_bar"], np.finfo(float).tiny)) - np.log(sigma_prev)) * dt
                + p["eta"] * np.sqrt(dt) * zsigma[:, i]
            )
            log_sigma_next = np.clip(log_sigma_next, log_floor, log_ceiling)
            sigma = np.clip(np.exp(log_sigma_next), sigma_floor, sigma_ceiling)
            u_jump = norm.cdf(zj[:, i])
            jump = (uj[:, i] < jump_prob).astype(float) * self._draw_jump(p, u_jump)
            if kappa > 0:
                S = theta_grid[i] + (S - theta_grid[i]) * exp_kdt + sigma_prev * ou_scale * z1[:, i] + jump
            else:
                S = S + p["kappa"] * (theta_grid[i] - S) * dt + sigma_prev * ou_scale * z1[:, i] + jump
            if in_window[i]:
                window_sum += S

        avg_price = window_sum / float(in_window.sum())
        payoff = np.maximum(avg_price - K, 0.0)
        return {
            "count": int(len(payoff)),
            "sum_payoff": float(payoff.sum()),
            "sumsq_payoff": float(np.square(payoff).sum()),
            "payoff": payoff,
            "avg_price": avg_price,
        }

    # -------------------- pricing --------------------
    def price_asian_call_mc(
        self,
        K: float,
        T: float,
        regime: str = "all",
        use_hourly: bool = False,
        n_paths: int | None = None,
        n_steps: int | None = None,
        antithetic: bool = True,
        S0: float | None = None,
        sigma0_override: float | None = None,
        delivery_start: float = 0.0,
        delivery_end: float | None = None,
        seed: int | None = None,
        n_jobs: int = 1,
        verbose: int = 10,
        return_paths: bool = False,
    ):
        """
        Plain Monte Carlo without control variate.
        """
        return self.price_asian_call_cv_mc(
            K=K,
            T=T,
            regime=regime,
            use_hourly=use_hourly,
            n_paths=n_paths,
            n_steps=n_steps,
            antithetic=antithetic,
            S0=S0,
            sigma0_override=sigma0_override,
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            seed=seed,
            n_jobs=n_jobs,
            verbose=verbose,
            sampling="mc",
            control_variate="none",
            return_paths=return_paths,
        )

    def price_asian_call_cv_mc(
        self,
        K: float,
        T: float,
        regime: str = "all",
        use_hourly: bool = False,
        n_paths: int | None = None,
        n_steps: int | None = None,
        antithetic: bool = True,
        S0: float | None = None,
        sigma0_override: float | None = None,
        delivery_start: float = 0.0,
        delivery_end: float | None = None,
        seed: int | None = None,
        n_jobs: int = 1,
        verbose: int = 10,
        sampling: str = "mc",
        control_variate: str = "average",
        return_paths: bool = False,
    ):
        """
        Control-variate pricing engine.

        sampling:
          - "mc": standard Monte Carlo
          - "sobol": scrambled Sobol quasi-MC

        control_variate:
          - "average": use arithmetic average over the delivery window as control variate
          - "none": no control variate
        """
        if delivery_end is None:
            delivery_end = T
        if S0 is None:
            S0 = self.current_price
        if n_paths is None:
            n_paths = self._default_n_paths(use_hourly=use_hourly)
        n_steps = self._infer_n_steps(T, use_hourly=use_hourly, n_steps=n_steps)
        p = self.params(regime=regime, use_hourly=use_hourly)
        base_seed = self.seed if seed is None else int(seed)
        jobs = self._resolve_n_jobs(n_jobs)

        self._log(
            f"[PRICE-CV] started | regime={regime} | T={T} | paths={n_paths} | steps={n_steps} | "
            f"sampling={sampling} | jobs={jobs}",
            verbose,
        )

        if return_paths:
            ts, paths, sigmas, _ = self._simulate_paths_core(
                T=T,
                regime=regime,
                use_hourly=use_hourly,
                n_paths=n_paths,
                n_steps=n_steps,
                antithetic=antithetic,
                S0=S0,
                sigma0_override=sigma0_override,
                seed=base_seed,
            )
            tgrid = np.linspace(0.0, T, paths.shape[1])
            mask = (tgrid[1:] >= delivery_start) & (tgrid[1:] <= delivery_end)
            if int(mask.sum()) < 1:
                raise ValueError("Delivery window is empty on the simulation grid.")
            avg_price = paths[:, 1:][:, mask].mean(axis=1)
            payoff = np.maximum(avg_price - K, 0.0)
            out = {
                "price": float(self.discount_factor(T) * payoff.mean()),
                "stderr": float(self.discount_factor(T) * payoff.std(ddof=1) / np.sqrt(len(payoff))),
                "discount_factor": float(self.discount_factor(T)),
                "avg_price": avg_price,
                "paths": paths,
                "sigmas": sigmas,
                "dates": ts,
                "tgrid": tgrid,
            }
            self._log(f"[PRICE-CV] finished | price={out['price']:.6f} | stderr={out['stderr']:.6f}", verbose)
            return out

        def make_randoms(batch_n: int, batch_seed: int):
            if sampling.lower() == "sobol":
                d = 4 * n_steps
                eng = qmc.Sobol(d=d, scramble=True, seed=batch_seed)
                u = eng.random(batch_n)
                if antithetic:
                    u = np.vstack([u, 1.0 - u])
                eps = np.finfo(float).eps
                z = norm.ppf(np.clip(u, eps, 1.0 - eps))
                z1 = z[:, :n_steps]
                z2 = z[:, n_steps:2 * n_steps]
                uj = u[:, 2 * n_steps:3 * n_steps]
                zj = z[:, 3 * n_steps:4 * n_steps]
                return z1, z2, uj, zj
            rng = default_rng(batch_seed)
            z1 = rng.standard_normal((batch_n, n_steps))
            z2 = rng.standard_normal((batch_n, n_steps))
            uj = rng.uniform(size=(batch_n, n_steps))
            zj = rng.standard_normal((batch_n, n_steps))
            if antithetic:
                z1 = np.vstack([z1, -z1])
                z2 = np.vstack([z2, -z2])
                uj = np.vstack([uj, 1.0 - uj])
                zj = np.vstack([zj, -zj])
            return z1, z2, uj, zj

        def run_batch(batch_n: int, batch_seed: int):
            dt = T / n_steps
            z1, z2, uj, zj = make_randoms(batch_n, batch_seed)
            zsigma = p["rho"] * z1 + np.sqrt(1.0 - p["rho"] ** 2) * z2
            step_days = T * self.day_count_basis / n_steps
            ts = pd.date_range(start=self.current_timestamp, periods=n_steps + 1, freq=pd.to_timedelta(step_days, unit="D"))
            theta_grid = self.theta(ts)

            S = np.full(z1.shape[0], S0, dtype=float)
            sigma = np.full(z1.shape[0], p["sigma0"] if sigma0_override is None else float(sigma0_override), dtype=float)
            tgrid = np.linspace(0.0, T, n_steps + 1)
            mask = (tgrid[1:] >= delivery_start) & (tgrid[1:] <= delivery_end)
            if int(mask.sum()) < 1:
                raise ValueError("Delivery window is empty on the simulation grid.")

            window_sum = np.zeros_like(S)
            jump_prob = 1.0 - np.exp(-max(p["jump_lambda"] + p.get("neg_jump_lambda", 0.0), 0.0) * dt)
            kappa = float(max(p["kappa"], 0.0))
            exp_kdt = np.exp(-kappa * dt) if kappa > 0 else 1.0
            ou_scale = np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)) if kappa > 0 else np.sqrt(dt)
            sigma_floor = float(p.get("sigma_floor", np.finfo(float).tiny))
            sigma_ceiling = float(p.get("sigma_ceiling", np.finfo(float).max))
            log_floor = np.log(max(sigma_floor, np.finfo(float).tiny))
            log_ceiling = np.log(max(sigma_ceiling, sigma_floor * (1.0 + np.finfo(float).eps)))

            for i in range(n_steps):
                sigma_prev = np.clip(sigma, sigma_floor, sigma_ceiling)
                log_sigma_next = (
                    np.log(sigma_prev)
                    + p["a"] * (np.log(np.maximum(p["sigma_bar"], np.finfo(float).tiny)) - np.log(sigma_prev)) * dt
                    + p["eta"] * np.sqrt(dt) * zsigma[:, i]
                )
                log_sigma_next = np.clip(log_sigma_next, log_floor, log_ceiling)
                sigma = np.clip(np.exp(log_sigma_next), sigma_floor, sigma_ceiling)
                u_jump = norm.cdf(zj[:, i])
                jump = (uj[:, i] < jump_prob).astype(float) * self._draw_jump(p, u_jump)
                if kappa > 0:
                    S = theta_grid[i] + (S - theta_grid[i]) * exp_kdt + sigma_prev * ou_scale * z1[:, i] + jump
                else:
                    S = S + p["kappa"] * (theta_grid[i] - S) * dt + sigma_prev * ou_scale * z1[:, i] + jump
                if mask[i]:
                    window_sum += S

            avg_price = window_sum / float(mask.sum())
            payoff = np.maximum(avg_price - K, 0.0)
            return payoff, avg_price

        batch_sizes = np.full(jobs, n_paths // jobs, dtype=int)
        batch_sizes[: n_paths % jobs] += 1
        batch_sizes = batch_sizes[batch_sizes > 0]
        seeds = SeedSequence(base_seed).spawn(len(batch_sizes))

        self._log("[PRICE-CV] Monte Carlo / Sobol engine running", verbose)
        if len(batch_sizes) == 1:
            payoff, avg_price = run_batch(int(batch_sizes[0]), int(seeds[0].generate_state(1)[0]))
            payoff_all = payoff
            avg_all = avg_price
        else:
            with parallel_backend("threading"):
                results = Parallel(n_jobs=len(batch_sizes), verbose=verbose)(
                    delayed(run_batch)(int(batch_sizes[i]), int(seeds[i].generate_state(1)[0]))
                    for i in range(len(batch_sizes))
                )
            payoff_all = np.concatenate([r[0] for r in results])
            avg_all = np.concatenate([r[1] for r in results])

        if control_variate == "average":
            cv_expect = self._expected_average_price(
                T=T,
                regime=regime,
                use_hourly=use_hourly,
                S0=S0,
                sigma0_override=sigma0_override,
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                n_steps=n_steps,
            )
            var_c = float(np.var(avg_all, ddof=1))
            beta = float(np.cov(payoff_all, avg_all, ddof=1)[0, 1] / var_c) if var_c > 0 else 0.0
            adj = payoff_all - beta * (avg_all - cv_expect)
        else:
            beta = 0.0
            adj = payoff_all

        df = self.discount_factor(T)
        out = {
            "price": float(df * np.mean(adj)),
            "stderr": float(df * np.std(adj, ddof=1) / np.sqrt(len(adj))),
            "discount_factor": float(df),
            "beta": float(beta),
            "cv_expect": float(cv_expect) if control_variate == "average" else np.nan,
            "sampling": sampling,
            "control_variate": control_variate,
            "n_paths": int(len(adj)),
        }
        self._log(f"[PRICE-CV] finished | price={out['price']:.6f} | stderr={out['stderr']:.6f} | beta={beta:.4f}", verbose)
        return out

    def compare_sampling_methods(
        self,
        K: float,
        T: float,
        regime: str = "all",
        use_hourly: bool = False,
        n_paths: int | None = None,
        n_steps: int | None = None,
        antithetic: bool = True,
        seed: int | None = None,
        verbose: int = 10,
    ) -> pd.DataFrame:
        rows = []
        for sampling in ("mc", "sobol"):
            t0 = time.perf_counter()
            out = self.price_asian_call_cv_mc(
                K=K,
                T=T,
                regime=regime,
                use_hourly=use_hourly,
                n_paths=n_paths,
                n_steps=n_steps,
                antithetic=antithetic,
                seed=seed,
                verbose=verbose,
                sampling=sampling,
                control_variate="average",
            )
            rows.append({
                "method": sampling,
                **out,
                "elapsed_sec": time.perf_counter() - t0,
            })
        return pd.DataFrame(rows)

    # -------------------- Greeks / surface --------------------
    def finite_difference_greeks(
        self,
        K: float,
        T: float,
        regime: str = "all",
        use_hourly: bool = False,
        n_paths: int | None = None,
        n_steps: int | None = None,
        antithetic: bool = True,
        S0: float | None = None,
        sigma0_override: float | None = None,
        delivery_start: float = 0.0,
        delivery_end: float | None = None,
        seed: int | None = None,
        n_jobs: int = 1,
        verbose: int = 10,
        mc_n_jobs: int = 1,
    ):
        if S0 is None:
            S0 = self.current_price
        if delivery_end is None:
            delivery_end = T
        p = self.params(regime=regime, use_hourly=use_hourly)
        dS = float(p["spot_bump_scale"])
        dV = float(p["sigma_bump_scale"])
        if not np.isfinite(dS) or dS <= 0:
            dS = max(abs(S0) * 1e-4, np.finfo(float).eps)
        if not np.isfinite(dV) or dV <= 0:
            dV = max(abs(p["sigma0"]) * 1e-4, np.finfo(float).eps)
        base_seed = self.seed if seed is None else int(seed)
        self._log(f"[GREEKS] started | regime={regime} | T={T} | jobs={n_jobs}", verbose)

        tasks = [
            ("base", S0, sigma0_override),
            ("upS", S0 + dS, sigma0_override),
            ("dnS", max(S0 - dS, np.finfo(float).eps), sigma0_override),
            ("upV", S0, (p["sigma0"] if sigma0_override is None else sigma0_override) + dV),
            ("dnV", S0, max((p["sigma0"] if sigma0_override is None else sigma0_override) - dV, np.finfo(float).eps)),
        ]

        def eval_task(label, spot, sig0):
            return self.price_asian_call_cv_mc(
                K=K,
                T=T,
                regime=regime,
                use_hourly=use_hourly,
                n_paths=n_paths,
                n_steps=n_steps,
                antithetic=antithetic,
                S0=spot,
                sigma0_override=sig0,
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                seed=base_seed,
                n_jobs=mc_n_jobs,
                verbose=0,
                sampling="mc",
                control_variate="average",
            )["price"]

        jobs = self._resolve_n_jobs(n_jobs)
        self._log("[GREEKS] pricing base / bumps", verbose)
        if jobs <= 1:
            prices = [eval_task(*t) for t in tasks]
        else:
            with parallel_backend("threading"):
                prices = Parallel(n_jobs=min(jobs, len(tasks)), verbose=verbose)(
                    delayed(eval_task)(*t) for t in tasks
                )

        base, up_S, dn_S, up_V, dn_V = prices
        out = {
            "price": float(base),
            "delta": float((up_S - dn_S) / (2.0 * dS)),
            "vega": float((up_V - dn_V) / (2.0 * dV)),
            "dS": float(dS),
            "dV": float(dV),
        }
        self._log(f"[GREEKS] finished | price={out['price']:.6f} | delta={out['delta']:.6f} | vega={out['vega']:.6f}", verbose)
        return out

    def option_surface(
        self,
        strikes,
        maturities,
        regime: str = "all",
        use_hourly: bool = False,
        n_paths: int | None = None,
        n_steps: int | None = None,
        antithetic: bool = True,
        S0: float | None = None,
        sigma0_override: float | None = None,
        delivery_start: float = 0.0,
        delivery_end_func=None,
        n_jobs: int = 1,
        verbose: int = 10,
        mc_n_jobs: int = 1,
        seed: int | None = None,
    ) -> pd.DataFrame:
        strikes = np.asarray(strikes, dtype=float)
        maturities = np.asarray(maturities, dtype=float)
        if S0 is None:
            S0 = self.current_price
        self._log(f"[SURFACE] started | strikes={len(strikes)} | maturities={len(maturities)}", verbose)

        tasks = []
        for T in maturities:
            delivery_end = T if delivery_end_func is None else float(delivery_end_func(T))
            for K in strikes:
                tasks.append((T, K, delivery_end))
        seeds = SeedSequence(self.seed if seed is None else int(seed)).spawn(len(tasks))

        def worker(i, T, K, delivery_end):
            g = self.finite_difference_greeks(
                K=K,
                T=T,
                regime=regime,
                use_hourly=use_hourly,
                n_paths=n_paths,
                n_steps=n_steps,
                antithetic=antithetic,
                S0=S0,
                sigma0_override=sigma0_override,
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                seed=int(seeds[i].generate_state(1)[0]),
                n_jobs=1,
                verbose=0,
                mc_n_jobs=mc_n_jobs,
            )
            return {"T": float(T), "K": float(K), "Price": g["price"], "Delta": g["delta"], "Vega": g["vega"]}

        jobs = self._resolve_n_jobs(n_jobs)
        if jobs <= 1:
            rows = [worker(i, *tasks[i]) for i in range(len(tasks))]
        else:
            with parallel_backend("threading"):
                rows = Parallel(n_jobs=min(jobs, len(tasks)), verbose=verbose)(
                    delayed(worker)(i, *tasks[i]) for i in range(len(tasks))
                )
        out = pd.DataFrame(rows)
        self._log(f"[SURFACE] finished | rows={len(out)}", verbose)
        return out

    def build_greek_surface(self, strikes, maturities, **kwargs) -> pd.DataFrame:
        return self.option_surface(strikes=strikes, maturities=maturities, **kwargs).rename(
            columns={"T": "maturity", "K": "strike", "Price": "price", "Delta": "delta", "Vega": "vega"}
        )

    @staticmethod
    def pivot_surface(surface: pd.DataFrame, value_col: str):
        if "T" in surface.columns:
            return surface.pivot(index="T", columns="K", values=value_col)
        return surface.pivot(index="maturity", columns="strike", values=value_col)

    # -------------------- futures / forward support --------------------
    def spot_daily_series(self) -> pd.Series:
        if self.hourly_ is not None:
            return self.aggregate_hourly_to_daily(self.hourly_)["Price"].astype(float).copy()
        return self.price_series_daily()

    def forward_daily_series(self) -> pd.Series:
        if self.futures_ is None:
            raise ValueError("No futures/forward data loaded.")
        if "Price" not in self.futures_.columns:
            raise ValueError("Futures dataframe does not contain a standard 'Price' column.")
        return self.futures_["Price"].astype(float).copy()

    def align_spot_and_futures(self, use_hourly: bool = True) -> pd.DataFrame:
        if self.futures_ is None:
            raise ValueError("Load futures data first with load_futures_csv().")
        spot = self.spot_daily_series() if use_hourly else self.price_series_daily()
        fut = self.forward_daily_series()
        combined = pd.concat([spot.rename("Spot"), fut.rename("Forward")], axis=1, join="inner").dropna()
        idx = pd.DatetimeIndex(combined.index)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        combined.index = idx
        if len(combined) == 0:
            raise ValueError("No overlapping dates between spot and futures data.")
        return combined.sort_index()

    def fit_forward_proxy(self, use_hourly: bool = True, verbose: int = 0) -> dict:
        """Fit a data-driven spot->futures proxy on overlapping dates.

        The Cal baseload future moves slowly relative to the daily spot; first-
        differencing at daily frequency produces near-zero signal (the future
        barely moves day-to-day while spot is noisy).  We therefore resample
        both series to weekly means before differencing, which captures the
        economically meaningful co-movement between the spot level and where
        the Cal future settles over multi-day windows.

        beta is estimated by intercept-free OLS on weekly first differences
        (ΔF_weekly ≈ beta * ΔS_weekly).  The level intercept alpha is
        recovered as mean(F - beta*S) over the overlap for use in delta
        conversion during the hedge simulation.
        """
        overlap = self.align_spot_and_futures(use_hourly=use_hourly).copy()

        # Resample to weekly means (Monday-anchored ISO weeks)
        spot_weekly = overlap["Spot"].resample("W-MON").mean().dropna()
        fwd_weekly = overlap["Forward"].resample("W-MON").mean().dropna()
        weekly = pd.concat([spot_weekly.rename("Spot"), fwd_weekly.rename("Forward")], axis=1).dropna()

        if len(weekly) < 10:
            # Fall back to daily if not enough weekly observations
            weekly = overlap.copy()
            self._log("[FWD] warning: fewer than 10 weekly observations, falling back to daily", verbose)

        spot_w = weekly["Spot"].to_numpy(dtype=float)
        fwd_w = weekly["Forward"].to_numpy(dtype=float)

        d_spot = np.diff(spot_w)
        d_fwd = np.diff(fwd_w)
        finite_mask = np.isfinite(d_spot) & np.isfinite(d_fwd)
        d_spot_c = d_spot[finite_mask]
        d_fwd_c = d_fwd[finite_mask]

        if len(d_spot_c) < 5:
            raise ValueError("Not enough overlapping first-difference observations for forward proxy.")

        # Intercept-free OLS on weekly differences: beta = (ΔS'ΔF) / (ΔS'ΔS)
        beta = float(np.dot(d_spot_c, d_fwd_c) / max(np.dot(d_spot_c, d_spot_c), np.finfo(float).tiny))

        # Level intercept from daily overlap
        spot_lvl = overlap["Spot"].to_numpy(dtype=float)
        fwd_lvl = overlap["Forward"].to_numpy(dtype=float)
        alpha = float(np.mean(fwd_lvl - beta * spot_lvl))

        # R² on weekly first differences
        d_fwd_hat = beta * d_spot_c
        ss_res = float(np.sum((d_fwd_c - d_fwd_hat) ** 2))
        ss_tot = float(np.sum((d_fwd_c - d_fwd_c.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        proxy = {
            "alpha": alpha,
            "beta": beta,
            "r2": r2,
            "n_obs": int(len(overlap)),
            "n_weekly": int(len(d_spot_c)),
            "start": overlap.index.min(),
            "end": overlap.index.max(),
            "spot_mean": float(overlap["Spot"].mean()),
            "forward_mean": float(overlap["Forward"].mean()),
            "spot_std": float(overlap["Spot"].std(ddof=1)) if len(overlap) > 1 else np.nan,
            "forward_std": float(overlap["Forward"].std(ddof=1)) if len(overlap) > 1 else np.nan,
        }
        self.forward_proxy_ = proxy
        self._log(
            f"[FWD] fitted proxy | alpha={alpha:.6f} | beta={beta:.6f} | r2={r2:.4f} | n={len(overlap)} | n_weekly={len(d_spot_c)}",
            verbose,
        )
        return proxy

    def forward_proxy_price(self, spot_price: float) -> float:
        if self.forward_proxy_ is None:
            raise ValueError("Forward proxy not calibrated. Call fit_forward_proxy() first.")
        alpha = float(self.forward_proxy_["alpha"])
        beta = float(self.forward_proxy_["beta"])
        return float(alpha + beta * float(spot_price))

    # -------------------- hedge --------------------
    def _conditional_asian_price(
        self,
        S0: float,
        sigma0: float,
        past_sum: float,
        past_count: int,
        K: float,
        remaining_T: float,
        delivery_start: float,
        delivery_end: float,
        regime: str,
        use_hourly: bool,
        n_paths: int,
        n_steps: int,
        antithetic: bool,
        seed: int,
        current_time: float = 0.0,
    ) -> float:
        if remaining_T <= 0:
            return max((past_sum / max(past_count, 1)) - K, 0.0)

        future_start, future_end = self._future_delivery_window(current_time, current_time + remaining_T, delivery_start, delivery_end)
        if future_end <= 0:
            return max((past_sum / max(past_count, 1)) - K, 0.0)

        out = self.price_asian_call_cv_mc(
            K=K,
            T=remaining_T,
            regime=regime,
            use_hourly=use_hourly,
            n_paths=n_paths,
            n_steps=n_steps,
            antithetic=antithetic,
            S0=S0,
            sigma0_override=sigma0,
            delivery_start=future_start,
            delivery_end=future_end,
            seed=seed,
            n_jobs=1,
            verbose=0,
            sampling="mc",
            control_variate="average",
            return_paths=True,
        )
        paths = out["paths"]
        tgrid = out["tgrid"]
        mask = (tgrid[1:] >= future_start) & (tgrid[1:] <= future_end)
        future_sum = paths[:, 1:][:, mask].sum(axis=1) if int(mask.sum()) > 0 else np.zeros(paths.shape[0], dtype=float)
        total_count = past_count + int(mask.sum())
        avg = (past_sum + future_sum) / float(max(total_count, 1))
        return float(np.maximum(avg - K, 0.0).mean())

    def simulate_dynamic_forward_hedge(
        self,
        K: float,
        T: float,
        regime: str = "all",
        use_hourly: bool = True,
        n_outer_paths: int | None = None,
        n_inner_paths: int | None = None,
        n_steps: int | None = None,
        antithetic: bool = True,
        S0: float | None = None,
        sigma0_override: float | None = None,
        delivery_start: float = 0.0,
        delivery_end: float | None = None,
        rebalance_grid: np.ndarray | None = None,
        n_jobs: int = -1,
        verbose: int = 10,
        seed: int | None = None,
    ):
        """Dynamic hedge against the calibrated futures proxy.

        This is a proxy hedge, not a perfect replication hedge. It uses:
        1) a data-fitted linear spot->forward mapping on overlapping spot/futures dates,
        2) the model delta of the Asian option with respect to spot,
        3) conversion from spot delta to forward units via dF/dS ≈ beta.
        """
        if S0 is None:
            S0 = self.current_price
        if delivery_end is None:
            delivery_end = T

        p = self.params(regime=regime, use_hourly=use_hourly)

        if self.futures_ is None:
            raise ValueError("Load futures data first with load_futures_csv() before running the hedge.")
        if self.forward_proxy_ is None:
            proxy = self.fit_forward_proxy(use_hourly=use_hourly, verbose=verbose if verbose else 0)
        else:
            proxy = self.forward_proxy_

        beta = float(proxy["beta"])
        alpha = float(proxy["alpha"])
        if not np.isfinite(beta) or abs(beta) <= np.finfo(float).eps:
            raise ValueError("Estimated forward proxy slope beta is too small to hedge with.")

        if n_outer_paths is None:
            n_outer_paths = max(64, self._default_n_paths(use_hourly=use_hourly) // 4)
        if n_inner_paths is None:
            n_inner_paths = max(64, self._default_n_paths(use_hourly=use_hourly) // 4)
        if n_steps is None:
            if self.futures_ is not None:
                fut_dt = self._observed_dt_years(self.futures_.index)
                if len(fut_dt) > 0 and np.isfinite(float(fut_dt.median())) and float(fut_dt.median()) > 0:
                    n_steps = max(1, int(np.ceil(T / float(fut_dt.median()))))
                else:
                    n_steps = self._infer_n_steps(T, use_hourly=use_hourly, n_steps=None)
            else:
                n_steps = self._infer_n_steps(T, use_hourly=use_hourly, n_steps=None)

        if rebalance_grid is None:
            # Data-driven default: rebalance on the simulation time grid implied by the observed sampling frequency.
            rebalance_grid = np.linspace(0.0, T, n_steps + 1)
        else:
            rebalance_grid = np.asarray(rebalance_grid, dtype=float)
            if rebalance_grid[0] != 0.0 or rebalance_grid[-1] != T:
                raise ValueError("rebalance_grid must start at 0 and end at T.")
            rebalance_grid = np.unique(rebalance_grid)

        base_seed = self.seed if seed is None else int(seed)
        jobs = self._resolve_n_jobs(n_jobs)

        self._log(
            f"[HEDGE] started | outer={n_outer_paths} | inner={n_inner_paths} | rebals={len(rebalance_grid)} | jobs={jobs}",
            verbose,
        )
        self._log(f"[HEDGE] forward proxy | alpha={alpha:.6f} | beta={beta:.6f} | r2={proxy.get('r2', np.nan):.4f}", verbose)

        self._log("[HEDGE] Step 1/4: simulating outer paths", verbose)
        _, outer_paths, outer_sigmas, _ = self._simulate_paths_core(
            T=T,
            regime=regime,
            use_hourly=use_hourly,
            n_paths=n_outer_paths,
            n_steps=n_steps,
            antithetic=antithetic,
            S0=S0,
            sigma0_override=sigma0_override,
            seed=base_seed,
        )
        outer_tgrid = np.linspace(0.0, T, outer_paths.shape[1])
        reb_idx = np.searchsorted(outer_tgrid, rebalance_grid, side="left")
        reb_idx = np.clip(reb_idx, 0, len(outer_tgrid) - 1)

        # Remove duplicate rebalance indices to avoid zero-length intervals.
        uniq, first_pos = np.unique(reb_idx, return_index=True)
        reb_idx = uniq
        rebalance_grid = rebalance_grid[np.sort(first_pos)]

        dS = float(p["spot_bump_scale"])
        if not np.isfinite(dS) or dS <= 0:
            dS = max(abs(S0) * 1e-4, np.finfo(float).eps)

        df_T = self.discount_factor(T)
        r = 0.0 if (T <= 0 or df_T <= 0) else float(-np.log(df_T) / T)

        def hedge_one_path(pidx: int, path_seed: int):
            S_path = outer_paths[pidx]
            sig_path = outer_sigmas[pidx]
            cash = 0.0
            units_prev = 0.0
            t_prev = 0.0

            for j, t in enumerate(rebalance_grid):
                idx = int(reb_idx[j])
                S_t = float(S_path[idx])
                sig_t = float(sig_path[idx]) if sigma0_override is None else float(sigma0_override)
                rem_T = max(T - t, 0.0)
                rem_steps = max(1, self._infer_n_steps(rem_T, use_hourly=use_hourly, n_steps=None)) if rem_T > 0 else 1
                seed_j = int(SeedSequence([path_seed, pidx, j]).generate_state(1)[0])

                obs_times = outer_tgrid[1 : idx + 1]
                obs_vals = outer_paths[pidx, 1 : idx + 1]
                past_mask = (obs_times >= delivery_start) & (obs_times <= min(t, delivery_end))
                past_sum = float(obs_vals[past_mask].sum())
                past_count = int(past_mask.sum())

                future_start, future_end = self._future_delivery_window(t, T, delivery_start, delivery_end)

                price0 = self._conditional_asian_price(
                    S_t,
                    sig_t,
                    past_sum,
                    past_count,
                    K,
                    rem_T,
                    future_start,
                    future_end,
                    regime,
                    use_hourly,
                    n_inner_paths,
                    rem_steps,
                    antithetic,
                    seed_j,
                    current_time=t,
                )
                price_up = self._conditional_asian_price(
                    S_t + dS,
                    sig_t,
                    past_sum,
                    past_count,
                    K,
                    rem_T,
                    future_start,
                    future_end,
                    regime,
                    use_hourly,
                    n_inner_paths,
                    rem_steps,
                    antithetic,
                    seed_j,
                    current_time=t,
                )
                price_dn = self._conditional_asian_price(
                    max(S_t - dS, np.finfo(float).eps),
                    sig_t,
                    past_sum,
                    past_count,
                    K,
                    rem_T,
                    future_start,
                    future_end,
                    regime,
                    use_hourly,
                    n_inner_paths,
                    rem_steps,
                    antithetic,
                    seed_j,
                    current_time=t,
                )
                delta_spot = (price_up - price_dn) / (2.0 * dS)

                # Convert spot delta into futures units using dF/dS ≈ beta from the overlapping historical sample.
                units_new = delta_spot / beta

                F_t = max(alpha + beta * S_t, np.finfo(float).eps)
                if j == 0:
                    cash = price0 - units_new * F_t
                else:
                    cash *= np.exp(r * (t - t_prev))
                    cash -= (units_new - units_prev) * F_t

                units_prev = units_new
                t_prev = t

            obs_times = outer_tgrid[1:]
            obs_vals = outer_paths[pidx, 1:]
            mask = (obs_times >= delivery_start) & (obs_times <= delivery_end)
            payoff = max(float(obs_vals[mask].mean()) - K, 0.0) if mask.any() else 0.0
            F_T = max(alpha + beta * float(S_path[-1]), np.finfo(float).eps)
            return cash + units_prev * F_T - payoff

        seeds = SeedSequence(base_seed).spawn(n_outer_paths)
        self._log("[HEDGE] Step 3/4: dynamic rebalancing", verbose)
        if jobs <= 1:
            pnl = np.asarray([hedge_one_path(i, int(seeds[i].generate_state(1)[0])) for i in range(n_outer_paths)], dtype=float)
        else:
            with parallel_backend("threading"):
                pnl = np.asarray(
                    Parallel(n_jobs=jobs, verbose=verbose)(
                        delayed(hedge_one_path)(i, int(seeds[i].generate_state(1)[0]))
                        for i in range(n_outer_paths)
                    ),
                    dtype=float,
                )

        self._log("[HEDGE] Step 4/4: summarizing hedge PnL", verbose)
        option_price = self.price_asian_call_cv_mc(
            K=K,
            T=T,
            regime=regime,
            use_hourly=use_hourly,
            n_paths=max(256, n_inner_paths),
            n_steps=n_steps,
            antithetic=antithetic,
            S0=S0,
            sigma0_override=sigma0_override,
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            seed=base_seed,
            n_jobs=1,
            verbose=0,
            sampling="mc",
            control_variate="average",
        )["price"]

        result = {
            "pnl_paths": pnl,
            "mean_pnl": float(np.mean(pnl)),
            "std_pnl": float(np.std(pnl, ddof=1)),
            "var_99": float(np.quantile(pnl, 0.01)),
            "cvar_99": float(pnl[pnl <= np.quantile(pnl, 0.01)].mean()),
            "option_price": float(option_price),
            "forward_alpha": alpha,
            "forward_beta": beta,
            "forward_r2": float(proxy.get("r2", np.nan)),
            "forward_overlap_n": int(proxy.get("n_obs", 0)),
            "outer_paths": outer_paths,
            "outer_sigmas": outer_sigmas,
            "rebalance_grid": rebalance_grid,
            "hedge_type": "forward_proxy",
        }
        self._log(f"[HEDGE] finished | mean_pnl={result['mean_pnl']:.6f} | var99={result['var_99']:.6f}", verbose)
        return result

    def simulate_dynamic_delta_hedge(
        self,
        K: float,
        T: float,
        regime: str = "all",
        use_hourly: bool = True,
        n_outer_paths: int | None = None,
        n_inner_paths: int | None = None,
        n_steps: int | None = None,
        antithetic: bool = True,
        S0: float | None = None,
        sigma0_override: float | None = None,
        delivery_start: float = 0.0,
        delivery_end: float | None = None,
        rebalance_grid: np.ndarray | None = None,
        n_jobs: int = -1,
        verbose: int = 10,
        seed: int | None = None,
    ):
        """Compatibility wrapper. Uses the forward-proxy hedge implementation."""
        return self.simulate_dynamic_forward_hedge(
            K=K,
            T=T,
            regime=regime,
            use_hourly=use_hourly,
            n_outer_paths=n_outer_paths,
            n_inner_paths=n_inner_paths,
            n_steps=n_steps,
            antithetic=antithetic,
            S0=S0,
            sigma0_override=sigma0_override,
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            rebalance_grid=rebalance_grid,
            n_jobs=n_jobs,
            verbose=verbose,
            seed=seed,
        )

    def hedge_pnl_table(self, hedge_result: dict) -> pd.DataFrame:
        pnl = np.asarray(hedge_result["pnl_paths"], dtype=float)
        q = np.quantile(pnl, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        extra = {
            "ForwardAlpha": hedge_result.get("forward_alpha", np.nan),
            "ForwardBeta": hedge_result.get("forward_beta", np.nan),
            "ForwardR2": hedge_result.get("forward_r2", np.nan),
            "OverlapN": hedge_result.get("forward_overlap_n", np.nan),
        }
        return pd.DataFrame([
            {
                "Mean": float(np.mean(pnl)),
                "Std": float(np.std(pnl, ddof=1)),
                "Min": float(np.min(pnl)),
                "1%": float(q[0]),
                "5%": float(q[1]),
                "25%": float(q[2]),
                "Median": float(q[3]),
                "75%": float(q[4]),
                "95%": float(q[5]),
                "99%": float(q[6]),
                "Max": float(np.max(pnl)),
                "VaR99": float(hedge_result.get("var_99", np.nan)),
                "CVaR99": float(hedge_result.get("cvar_99", np.nan)),
                **extra,
            }
        ])

    # -------------------- extra diagnostics --------------------
    # -------------------- extra diagnostics --------------------
    def stress_test_table(
        self,
        K: float,
        T: float,
        regime: str = "all",
        use_hourly: bool = False,
        shock_grid: np.ndarray | None = None,
        vol_scale_grid: np.ndarray | None = None,
        n_paths: int | None = None,
        n_steps: int | None = None,
        antithetic: bool = True,
        seed: int | None = None,
        n_jobs: int = -1,
        verbose: int = 10,
    ) -> pd.DataFrame:
        if shock_grid is None:
            shock_grid = np.array([-0.5, -0.25, 0.0, 0.25, 0.5, 1.0], dtype=float)
        if vol_scale_grid is None:
            vol_scale_grid = np.array([0.5, 1.0, 1.5, 2.0], dtype=float)

        p = self.params(regime=regime, use_hourly=use_hourly)
        S0 = self.current_price
        sigma0 = p["sigma0"]
        base_seed = self.seed if seed is None else int(seed)

        base_price = self.price_asian_call_cv_mc(
            K=K,
            T=T,
            regime=regime,
            use_hourly=use_hourly,
            n_paths=n_paths,
            n_steps=n_steps,
            antithetic=antithetic,
            S0=S0,
            sigma0_override=sigma0,
            seed=base_seed,
            n_jobs=1,
            verbose=0,
            sampling="mc",
            control_variate="average",
        )["price"]

        tasks = [(float(sh), float(vs)) for sh in np.asarray(shock_grid, dtype=float) for vs in np.asarray(vol_scale_grid, dtype=float)]
        seeds = SeedSequence(base_seed).spawn(len(tasks))

        def worker(i, sh, vs):
            spot = max(S0 * (1.0 + sh), np.finfo(float).eps)
            sig = max(sigma0 * vs, np.finfo(float).eps)
            out = self.price_asian_call_cv_mc(
                K=K,
                T=T,
                regime=regime,
                use_hourly=use_hourly,
                n_paths=n_paths,
                n_steps=n_steps,
                antithetic=antithetic,
                S0=spot,
                sigma0_override=sig,
                seed=int(seeds[i].generate_state(1)[0]),
                n_jobs=1,
                verbose=0,
                sampling="mc",
                control_variate="average",
            )
            return {
                "SpotShockPct": sh,
                "VolScale": vs,
                "ShockedSpot": spot,
                "ShockedSigma0": sig,
                "Price": out["price"],
                "StdErr": out["stderr"],
                "PnL_vs_base": out["price"] - base_price,
            }

        jobs = self._resolve_n_jobs(n_jobs)
        if jobs <= 1:
            rows = [worker(i, *tasks[i]) for i in range(len(tasks))]
        else:
            with parallel_backend("threading"):
                rows = Parallel(n_jobs=min(jobs, len(tasks)), verbose=verbose)(
                    delayed(worker)(i, *tasks[i]) for i in range(len(tasks))
                )
        return pd.DataFrame(rows)

    def realism_report(
        self,
        regime: str = "all",
        use_hourly: bool = True,
        n_paths: int | None = None,
        n_steps: int | None = None,
        antithetic: bool = True,
        seed: int | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> pd.DataFrame:
        p = self.params(regime=regime, use_hourly=use_hourly)
        s = self.price_series_hourly() if (use_hourly and self.hourly_ is not None) else self.price_series_daily()
        if (s > 0).all():
            hist_inc = np.log(s / s.shift(1)).dropna()
        else:
            scale = float(np.abs(s).median()) if float(np.abs(s).median()) > 0 else 1.0
            hist_inc = (s.diff() / scale).dropna()

        hist = {
            "HistMean": float(hist_inc.mean()),
            "HistStd": float(hist_inc.std(ddof=1)),
            "HistSkew": float(hist_inc.skew()),
            "HistKurt": float(hist_inc.kurtosis()),
            "HistN": int(len(hist_inc)),
        }

        T = float(p["obs_frequency_years"])
        _, paths, _, _ = self._simulate_paths_core(
            T=T,
            regime=regime,
            use_hourly=use_hourly,
            n_paths=n_paths or max(1024, self._default_n_paths(use_hourly=use_hourly)),
            n_steps=1,
            antithetic=antithetic,
            S0=self.current_price,
            sigma0_override=p["sigma0"],
            seed=seed,
        )
        sim_inc = paths[:, -1] - paths[:, 0]
        sim = {
            "SimMean": float(np.mean(sim_inc)),
            "SimStd": float(np.std(sim_inc, ddof=1)),
            "SimSkew": float(pd.Series(sim_inc).skew()),
            "SimKurt": float(pd.Series(sim_inc).kurtosis()),
            "SimN": int(len(sim_inc)),
        }
        return pd.DataFrame([{**hist, **sim}])

    def rolling_vol_surface(self, use_hourly: bool = True, lookbacks: np.ndarray | None = None) -> pd.DataFrame:
        s = self.price_series_hourly() if (use_hourly and self.hourly_ is not None) else self.price_series_daily()
        s = s.dropna()
        if len(s) < 10:
            raise ValueError("Not enough data for rolling volatility surface.")
        if lookbacks is None:
            dt = self._observed_dt_years(s.index)
            obs_per_year = 1.0 / float(dt.median())
            lookbacks = np.unique(np.round(np.geomspace(max(2, obs_per_year / 365.0), max(5, obs_per_year / 4.0), num=8)).astype(int))
        else:
            lookbacks = np.asarray(lookbacks, dtype=int)
        base = np.log(s / s.shift(1)).dropna() if (s > 0).all() else (s.diff() / max(float(np.abs(s).median()), 1.0)).dropna()
        out = pd.DataFrame(index=base.index)
        ann_scale = np.sqrt(1.0 / float(self._observed_dt_years(s.index).median()))
        for w in lookbacks:
            out[f"LB_{int(w)}"] = base.rolling(int(w)).std(ddof=1) * ann_scale
        return out.dropna(how="all")

    def diagnostics_block(
        self,
        calibration=None,
        pricing=None,
        greeks=None,
        stress=None,
        hedge_summary=None,
        realism=None,
        rolling_vol_surface=None,
    ) -> dict:
        return {
            "calibration": calibration,
            "pricing": pricing,
            "greeks": greeks,
            "stress": stress,
            "hedge_summary": hedge_summary,
            "realism": realism,
            "rolling_vol_surface": rolling_vol_surface,
        }

    def print_overview(self, bundle: dict):
        for name in ["calibration", "pricing", "greeks", "stress", "hedge_summary", "realism"]:
            if bundle.get(name) is not None:
                print(f"\n=== {name.upper()} ===", flush=True)
                print(bundle[name])
