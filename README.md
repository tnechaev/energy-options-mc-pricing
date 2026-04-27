# Energy Options: Pricing & Risk Framework

## Contents
- [Project Objectives](#project-objectives)
- [Model Overview](#model-overview)
- [Main Class](#main-class-electricityspotasianpricer)
- [Data](#data)
- [Key Results](#key-results)
- [Hedging Logic](#hedging-logic)
- [Limitations & Open Issues](#limitations--open-issues)

---

A quant research project for pricing **Asian-style options on electricity spot prices** using a jump-diffusion model with mean-reversion and stochastic volatility, regime-aware. Parameters are calibrated from 10 years of hourly German day-ahead prices (ENTSO-e/EMBER). This is a research-grade demo, actively evolving.

**Current status:** Pricing and calibration are working. Number of paths for MC in hedge simulation (pricing part) is now chosen via convergence testing. However, hedge simulation still produces negative mean P&L — see [Hedging Logic](#hedging-logic) for possible reasons.

---

## Project Objectives

- Build a realistic stochastic model for electricity spot price dynamics with seasonality, mean-reversion, stochastic volatility and positive/negative jumps
- Calibrate to historical German hourly price data
- Price Asian call options via Monte Carlo with control variates
- Compute Greeks (Delta, Vega) via finite-difference bumping with CRN
- Simulate dynamic delta hedging and compute risk metrics (VaR, CVaR)
- Stress test across spot and volatility shocks

---

## Model Overview

The spot price is modeled as follows:

```
dS = κ(θ(t) − S) dt + σ_t dW_S + J dN
d log σ = a(log σ̄ − log σ) dt + η dW_σ
corr(dW_S, dW_σ) = ρ
```

where:
- `θ(t)` — deterministic seasonality (hour-of-day + day-of-week + day-of-year Fourier components)
- `κ` — mean-reversion speed (calibrated daily: ~20/year, half-life ~12 days)
- `σ_t` —  stochastic volatility
- `J` — jump size drawn from empirical two-sided distribution (positive spikes + negative spikes)
- `N` — Poisson arrival with intensity `λ_pos + λ_neg`

Jump compensation is applied so the jump process has zero net drift under the pricing measure.

---

## Main Class: `ElectricitySpotAsianPricer`

### Calibration
- Two-pass kappa estimation: jumps filtered on daily data first, then AR(1) fitted on jump-cleaned residuals → unbiased kappa
- Sigma / stochastic vol parameters calibrated on hourly data using the daily-fitted kappa
- Two-sided jump calibration: positive and negative extreme residuals separated, each with independent λ, μ, σ

### Monte Carlo Engine
- Antithetic variates 
- Parallel execution via `joblib`
- Control variate: model-implied expected average `E[S̄]` with zero jump drift

### Pricing
- Asian call on arithmetic average spot over a delivery window
- Control variate variance reduction (CV β ≈ 0.60 at T=0.25)
- Standard error reported on all price outputs
- Stress test grid across spot shocks (−50% to +100%) and vol scaling (0.5× to 2×)

### Greeks
- Finite-difference Delta and Vega with CRN
- Vega bump = 5% of σ₀ (relative, not absolute, to avoid over-bumping)

### Forward Proxy
- Synthetic model-implied forwards computed from expected path for any delivery window

### Hedging
- Multi-strip synthetic forward hedge — see [Hedging Logic](#hedging-logic)
- Hedge P&L simulation with mean, std, VaR99, CVaR99

---

## Data

| Dataset | Source | Period | Frequency |
|---|---|---|---|
| German day-ahead spot (EPEX) | EMBER | Jan 2015–Feb 2026 | Hourly |
| German yearly baseload futures (**no longer used**)| investing.com | 2017–2025 | Daily |

---

## Key Results

All results use regime=`all`, T=0.25y (3-month option), K=80 EUR/MWh, S₀=93.24 EUR/MWh.

### Calibration

| Parameter | Value | Interpretation |
|---|---|---|
| κ (kappa) | 20.33 /year | Half-life ≈ 12.5 days |
| σ₀ | 397.7 EUR/MWh/√year | Current annualised diffusion vol |
| σ̄ | 582.0 | Long-run vol level |
| θ_mean | 73.4 EUR/MWh | Overall seasonal mean |
| λ_pos | 56.2 /year | ~1 positive spike per week |
| μ_pos | 55.1 EUR/MWh | Average spike size |
| Jump count | 1167 over 10y | Both positive and negative |

### Pricing (T=0.25, K=80, paths=12000, steps=365)

| Metric | Value |
|---|---|
| Option price | **18.72 EUR/MWh** |
| Monte Carlo stderr | 0.121 |
| CV β | 0.60 |
| Expected average E[S̄] | 70.6 EUR/MWh |
| Delta | 0.085 |
| Vega | 0.018 |

The expected average (70.6) is below strike (80) due to fast mean-reversion pulling paths toward θ=73. The option is slightly out-of-the-money on the model's expected path but still gets the value from jump and vol dispersion.


### Stress Test Summary (base price ≈ 18.3)

| Spot shock | Vol × 1.0 | Vol × 2.0 |
|---|---|---|
| −50% (S=46.6) | +14.97 (−3.8) | +23.40 (+5.1) |
| 0% (S=93.2) | +18.52 (base) | +27.44 (+8.9) |
| +50% (S=139.9) | +23.06 (+4.7) | +31.82 (+13.5) |
| +100% (S=186.5) | +27.86 (+9.6) | +36.67 (+18.3) |

PnL vs base in parentheses. The option price seems to be more sensitive to vol scaling than to spot shocks, possibly due to fast mean-reversion dampening spot sensitivity.

### Hedge Simulation (outer=1015, inner=1015, rebals=15, strips=3)

| Metric | Value |
|---|---|
| Mean P&L | −14.96 EUR |
| Std | — |
| VaR 99% | −121.4 EUR |
| CVaR 99% | — |
| Hedge type | Multi-strip synthetic forward |
| Strips | 3 monthly |
| avg dF/dS | 0.174 |

Mean P&L is negative — see [Hedging Logic](#hedging-logic) for why and what remains to be fixed.

---

## Hedging Logic

### Approach

The hedge uses **model-implied synthetic forwards**  due to lack of freely available historic data. At each rebalancing step t, the remaining delivery window `[max(d₀, t), d₁]` is split into monthly sub-strips. For each strip i:

```
F_i(t)  = E_t[ S̄_{strip_i} ]           (expected path from current S_t)
dF_i/dS ≈ exp(−κ × mid_i)               (analytic sensitivity)
h_i     = w_i × min(1/dF_i, 1/dF_near) × Δ_spot   (leverage-capped units)
```

where `w_i = |strip_i| / |delivery window|` is the strip weight and leverage is capped at the near-dated strip's inverse sensitivity to prevent exploding positions on far-dated strips.

Each strip maintains its own cash account. At terminal settlement each strip closes at its realised average.

### Why P&L is negative

The negative mean P&L is likely a consequence of fast mean-reversion:

**1. Near-zero instrument sensitivity.** With κ=20, the model implies:
- Month 1 forward: `dF/dS = exp(−20×0.042) ≈ 0.43` — hedgeable
- Month 2 forward: `dF/dS ≈ 0.08` — weakly hedgeable
- Month 3 forward: `dF/dS ≈ 0.016` — essentially unhedgeable

The **avg dF/dS = 0.17** confirms: the hedge instruments on average capture only 17% of the spot's sensitivity.

**2. Discrete delta is a poor approximation.** The option payoff is a nonlinear function of a path average. With κ=20 (fast reversion) and large jumps (λ=56, μ=55), the spot process is highly non-Gaussian over rebalancing intervals. First-order delta hedging misses the convexity (gamma) and jump discontinuities, creating rebalancing errors.

**3. Inner MC noise.** Delta is estimated with 1015 inner paths: stderr on delta ≈ `0.121/sqrt(1015)/3.52 ≈ 0.001`. Over 15 rebalancing steps this accumulates.

### What would improve hedge performance

1. **Reduce kappa** via a longer estimation window or a different model: a slower mean-reversion rate could make dF/dS larger for 1–3 month forwards
2. **Gamma hedging**: add a second-order correction using the option's gamma
3. **Shorter delivery windows**: a, e.g., 1-month Asian option should be more hedgeable than a 3-month one
4. **More inner paths**: would reduce delta noise at the cost of runtime

---

## Limitations & Open Issues

| Issue | Status |
|---|---|
| Hedge mean P&L negative | Potentially model-driven: fast κ might make forwards less sensitive to spot prices; partially mitigated by multi-strip |
| ECB discount curve | Old data access broken with new Python; currently using discount factor = 1; fixing ongoing |
| Calibration two-pass kappa | Daily aggregation is approximate; joint MLE would be better |
| No market-implied vol surface | Calibration is entirely historical |
| Runtime | 40 min for full hedge simulation, already parallelized; try more efficient way? |