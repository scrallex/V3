# V3 Deployment Guide

## Strategy Profile
- **Target:** "Safe Regimes" (Structural Stability + Low Hazard).
- **Models:** 7 Pair-Specific XGBoost Models (`models/model_*.json`).
- **Input Data:** S5 (5-Second) Granularity.
- **Features:** Regime Stability, Entropy, Coherence, RSI, Volatility, EMA Distance.

## Trade Mechanics
- **Entry Trigger:** Model Probability > 0.55 (Confirmed Signal).
- **Exit Logic:**
  - **Primary:** Signal Decay (Probability drops < 0.55).
  - **Horizon:** Models are trained to predict **5-Minute** profitability window.
  - **Average Duration:** Typically 5-15 minutes (as long as the regime remains stable).
- **Frequency:** ~142 trades/day (across 7 pairs).
- **Sizing:** Recommended **14% NAV per trade** (Unleveraged).

## Execution Components
- **`regime_agent.py`:** Runs inference and publishes `gate:last:{instrument}`.
- **Executor:** Should listen to `is_open: True/False` derived from the Gate Signal.

## Status
- **Models:** Trained & Verified (Jan 2026).
- **Code:** Synced.
- **Binary:** Configured.

## Launch
1. `tar -xzf v3_deployment_ready.tar.gz`
2. Restart `regime_agent` service.
