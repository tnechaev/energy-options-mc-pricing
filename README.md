# Energy Options: Pricing & Risk Framework  

This is an ongoing quant research project aimed at
pricing energy options (with a focus on German power) using Monte Carlo simulation 
and a jump-diffusion mean-reversion model, including additionally stochastic volatility, 
seasonality and regime filtering. 
Model parameters are calibrated using daily German electricity price data sourced from ENTSO-e.
Discount factors are derived from yield curves published by European Central Bank.

# Attention!
This project is a **research-grade** demo, not production-grade. 
Its limitations are listed below. It is also actively evolving:

- Calibration logic is updated
- Risk and hedging functions are refined
- New diagnostics and filters are added
---

## Project Objectives

- Build a **realistic stochastic model** for electricity prices
- Calibrate parameters to **historical German power data**
- Explicitly model **crisis vs normal regimes**
- Price European call options using **calibrated model and MC simulation**
- Derive **Greeks (Delta, Vega)** via pathwise / finite-difference methods
- Simulate **hedged P&L** and compute risk metrics (VaR, CVaR)
- Stress test and analyse model weaknesses

---

## Model Overview

Formulas are given in the demo jupyter notebook.

---

## Main Class: `EnergyOptionPricer`

The class provides:

### Data & Regime Handling
- Rolling forward proxy construction from historical electricity price data
- Retrieval of ECB yield curves and discount factor derivation
- Seasonal mean-reversion function `theta(t)`
- **Normal / crisis regime split**
  - Uses rolling median + MAD 
  - Correctly isolates 2022–23 energy crisis

### Calibration
- Historical log-return and volatility estimation
- Stochastic volatility calibration
- Jump intensity, mean and variance estimation
- Crisis and normal regimes calibrated separately

### Monte Carlo Engine
- Correlated Brownian motions
- Antithetic variates
- Common Random Numbers (CRN)
- Jump diffusion
- Stochastic volatility
- Mean reversion to seasonal

### Pricing
- European call pricing
- Control variates
- Multi-strike pricing from the same path set
- Monte Carlo standard error reporting

### Greeks
- Delta & Vega via pathwise derivatives
- Finite-difference fallback
- Surface computation (strike × maturity)

### Risk & Hedging
- Static delta/vega hedging
- Monte Carlo hedged P&L simulation
- VaR & CVaR estimation
- Stress scenarios on forward level

---

## Limitations

### 1) No actual forward prices, using a 30-day rolling proxy on electricity prices instead. 

### 2) No Market Implied Volatility Surface
There is no calibration to real option prices (those are not freely available).
Volatility is inferred from historical forward prices only.

### 3) Monte Carlo Accuracy
- Path counts are limited by runtime and memory.
- Tail risk (CVaR) is especially noisy.

### 4) Imperfect Hedging
The hedge is **first-order only (Delta/Vega)** and static.
