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

- Hedge: now using synthetic forwards derived from spot price data, but still negative PnL; needs further work
- ECB curve discounting: old way to access data no longer works with new Python version, needs complete rewrite
- Calibration logic still work in progress
- Overall computational efficiency is still being improved
- Result summaries are still messy and are being improved

---

## Project Objectives

- Build a **realistic stochastic model** for electricity spot price dynamics, that includes **seasonality, mean reversion, stochastic volatility and jumps**
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
- positive/negative jump spikes as possible in energy markets
- regime splitting between normal and crisis periods

For the model formulas please see the demo notebook.


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
- Multi-path pricing with diagnostics
- Stress testing across spot and volatility shocks

### Greeks
- Finite-difference **Delta** and **Vega**
- Greek surface computation across strike and maturity
- Diagnostic reporting for bump sizes and model stability

### Hedging
- **Forward-proxy hedge** based on synthetic forwards of matching tenors
- Minimum-variance style hedge logic
- Hedged P&L simulation
- Risk metrics: mean P&L, standard deviation, VaR, CVaR

### Diagnostics
- Calibration summary tables
- Realism report comparing historical and simulated increments
- Rolling volatility surfaces
- Stress test tables
- Hedge P&L tables
- Greek surfaces and contour plots

---

## Data 

- Spot data: hourly German electricity spot prices, taken from EMBER, from 2015 until Feb. 2026
- Futures data: historical German yearly baseload at daily frequency, from 2017 to 2025; taken from investing.com; currently **not used** because not suitable for hedging

---

## Notes on Hedging

**Section outdated, is in rewrite**

Currently the function:

- fits a **forward proxy** from overlapping spot/futures observations
- uses that proxy as a hedge instrument
- reports the quality of the fit via regression coefficients and \(R^2\)
- simulates hedge P&L to assess how much risk is actually reduced

This is still a very rough approximation.  
A single yearly baseload future is not sufficient for hedging. This needs to be improved by adding more tenors, potentially synthesized from spot data, for a simulation.

---

## Limitations

- **Hedge quality depends on proxy quality**
Need forwards of matching tenor to the maturity of the option, and hedge with the weighted combination of those. Might be still not enough, need to check for other limitations.

- **No historical forward curve available**
Using synthesized forwards. 

- **No market-implied volatility surface**
The calibration is done on historical spot price data, not from option market quotes.

- **MC runtime**
- Large path counts can be memory intensive
- Nested pricing in hedge routines is expensive
- Parallelization helps, but one layer of parallelism is sometimes needed and is not the best

---

