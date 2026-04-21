# Energy Options: Pricing & Risk Framework  

## Contents
- [Project Objectives](#project-objectives)
- [Model Overview](#model-overview)
- [Main Class: ElectricitySpotAsianPricer](#main-class-electricityspotasianpricer)
- [Data](#data)
- [Notes on Hedging](#notes-on-hedging)
- [Limitations](#limitations)

---

This is an ongoing quant research project aimed at pricing **power options** on spot prices using a **Merton jump-diffusion-style model with mean-reversion, stochastic volatility, seasonality, and regime filtering** and a MC engine.

The framework is designed primarily for **Asian-style options on electricity spot**, where the payoff depends on the **average spot price over a delivery window**.  
Model parameters are calibrated from **historical German hourly electricity price data**. For the hedge proxy construction, **historical yearly baseload futures data**  was used.

## Attention!
This project is a **research-grade demo**.  
It is actively evolving:

- Calibration logic is refined further
- Hedge logic is improving, as it is still heavy and noisy
- More diagnostics and visualization tools are being added
- Overall computational efficiency is still being improved
- Result summaries are still messy and are being improved

---

## Project Objectives

- Build a **realistic stochastic model** for electricity spot prices, that includes **seasonality, mean reversion, stochastic volatility, and jumps**
- Calibrate parameters to **historical German power data**
- Explicitly model **normal vs crisis regimes**
- Price **Asian call options** on spot using calibrated model and MC engine
- Derive **Greeks (Delta, Vega)** via finite-difference MC
- Use **forward/futures proxy hedging**; simulate hedged P&L and compute risk metrics such as **VaR** and **CVaR**
- Include **stress testing** and model realism diagnostics
---

## Model Overview

The model is built around a **seasonal mean-reverting electricity spot process** with:

- deterministic seasonality `theta(t)`
- mean reversion in spot
- stochastic volatility
- positive jump spikes
- regime splitting between normal and crisis periods

For the model formulas please see the demo notebook.

The option payoff is modeled as:

\[
\max\left(\frac{1}{N}\sum_{i=1}^{N} S_{t_i} - K, 0\right)
\]

which is appropriate for an **Asian-style contract on average spot**, specifically for delivery-window products in power markets.

---

## Main Class: ElectricitySpotAsianPricer

The class provides:

### Data Loading & Alignment
- Load **hourly German spot price data**
- Load **futures / forward proxy data** 
- Automatically align spot and futures data on the **overlapping date range only**
- Filter calibration and hedge proxy logic to the common observation window

### Seasonality & Regime Handling
- Seasonal mean function `theta(t)`
- Deseasonalized spot series
- Robust regime split using standardized residuals
- Vol z-score-style regime detection

### Monte Carlo Engine
- Correlated Brownian motions
- Antithetic variates
- Common random numbers (CRN) where needed
- Parallel execution
- MC and quasi-MC / Sobol support

### Pricing
- Asian call pricing on average spot
- Control variates using the model-implied average
- Monte Carlo standard error reporting
- Multi-path pricing with detailed diagnostics
- Stress testing across spot and volatility shocks

### Greeks
- Finite-difference **Delta** and **Vega**
- Greek surface computation across strike and maturity
- Diagnostic reporting for bump sizes and model stability

### Hedging
- **Forward-proxy hedge** based on futures data
- Spot-to-futures regression proxy on overlapping dates
- Minimum-variance style hedge logic
- Hedged P&L simulation
- Risk metrics: mean P&L, standard deviation, VaR, CVaR
- Designed to be computationally lighter than nested spot-only hedge Monte Carlo

### Diagnostics
- Calibration summary tables
- Realism report comparing historical and simulated increments
- Rolling volatility surfaces
- Stress test tables
- Hedge P&L tables
- Greek surfaces and contour plots

---

## Data 

- Spot data: hourly German electricity spot prices, taken from EMBER
- Futures data: historical German yearly baseload at daily frequency, taken from investing.com ; used for rough hedge proxy; ideally need more tenors for basket construction

---

## Notes on Hedging

Currently the function:

- fits a **forward proxy** from overlapping spot/futures observations
- uses that proxy as a hedge instrument
- reports the quality of the fit via regression coefficients and \(R^2\)
- simulates hedge P&L to assess how much risk is actually reduced

This is still a rough approximation.  
A single yearly baseload future is ok as a first hedge proxy, but it is not enough for a tight hedge of a half-year Asian option. More futures tenors would improve the hedge substantially.

---

## Limitations

### 1) Hedge quality depends on proxy quality
If only a yearly futures series is available, the hedge is necessarily coarse and basis risk can remain large.

### 2) No full forward curve
The current setup works with available historical futures data, but it is not yet a full multi-tenor forward curve hedge.

### 3) No market-implied volatility surface
The framework is calibrated from historical data rather than option market quotes.

### 4) Monte Carlo runtime
- Large path counts can be memory intensive
- Nested pricing in hedge routines is expensive
- Parallelization helps, but one layer of parallelism is sometimes needed and is not the best

### 5) Hedging is approximate
The hedge is designed as a **risk-reduction test**, not a perfect replication.

---

