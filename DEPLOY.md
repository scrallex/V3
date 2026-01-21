# SEP V3 Regime System Deployment Guide
**Version:** 3.0.0-RegimeAdaptive
**Date:** January 2026

## 1. Overview
This package contains the fully integrated **Regime Adaptive Trading System**. It replaces the legacy "Stability" logic with a dual-model Machine Learning core (XGBoost) driven by the **SFI Hazard** metric.

**Logic:**
- **Input:** Live Market Data (S2/S5 Updates) -> Aggregated M1.
- **Pipeline:** `manifold_generator` (C++ Binary) -> `regime_agent.py` (Feature Eng) -> Model Inference.
- **Switch:**
  - If `structure_hazard` (Normalized) < 0: Use **Low Volatility Model** (>55 Log Ret in Test).
  - If `structure_hazard` (Normalized) > 0: Use **High Volatility Model** (>20 Log Ret in Test).
- **Signal:** Long (>60%), Short (<40%).

### Execution Logic (New)
The `execution_agent.py` runs alongside the regime agent to handle orders and sizing:
- **NAV Sizing:** Dynamically calculates position size based on **Net Asset Value**.
- **Target Allocation:** **14% Notional Value per Pair** (approx 100% Total Exposure if all 7 active).
  - Example: $300 Account -> $42 Notional per trade.
  - As NAV grows to $3000 -> $420 Notional per trade.
- **Overlap:** Supports simultaneous positions in all 7 pairs.
- **Risk Control:** Sticky signals (re-sizes only on large drifts or entry/exit).

## 2. Directory Structure
```
V3/
├── bin/
│   └── manifold_generator       # Validated SFI Binary (Step=1 High Fidelity)
├── models/
│   ├── model_low_vol.json       # XGBoost Low Vol Model
│   ├── model_high_vol.json      # XGBoost High Vol Model
│   └── feature_builder_meta.json# Normalization Stats
├── scripts/
│   └── trading/
│       ├── regime_agent.py      # Signal Agent
│       └── execution_agent.py   # Execution Agent (NAV Sizing)
├── start_regime_system.sh       # Launcher (Runs both agents)
├── requirements.txt             # Python Deps
└── OANDA.env                    # Config
```

## 3. Deployment Instructions

### Step 1: Transfer
Copy the `sep_regime_v3.tar.gz` to the droplet.
```bash
scp sep_regime_v3.tar.gz user@droplet:/path/to/sep/
```

### Step 2: Extract
```bash
cd /path/to/sep
tar -xzvf sep_regime_v3.tar.gz
```

### Step 3: Install & Config
```bash
cd V3
pip install -r requirements.txt
# Check OANDA.env has correct Key and Account ID
```

### Step 4: Run
```bash
./start_regime_system.sh
```
This starts both agents.
- **Regime Agent:** Logs signals to stdout.
- **Execution Agent:** Logs orders (`Filled EUR_USD 100`) and Sizing (`Target Allocation...`).

## 4. Monitoring
Check logs for "Filled" orders.
Redis Keys:
- Signal: `gate:last:{INSTRUMENT}`

## 5. Live Execution Architecture & Alignment
To resolve the "Disconnect" between Backtest (Minute-Level) and Live Market (Seconds-Level), the system implements a **Strict Alignment Protocol**:

### The Disconnect
- **Backtest:** Trades on **Completed Candles** (Close Price).
- **Live Stream:** Receives updates every second (Forming Candle).
- **Risk:** Trading on forming candles creates noise ("Regime Flicker") and invalidates backtest stats.

### The Wiring Fix
The `regime_agent.py` is wired to filter this noise:
1.  **Wait for Completion:** It monitors the stream but **ignores** updates until the Minute Timestamp changes (e.g., 12:00 -> 12:01).
2.  **Trade Previous:** Immediately upon the new minute (12:01:00), it processes the **Completed 12:00 Candle**.
3.  **Result:** Live execution matches Backtest logic 1:1. Zero intra-minute flicker.

**Example Payload:**
```json
{
  "instrument": "EUR_USD",
  "signal": "LONG",
  "prob": 0.72,
  "regime": "LowVol",
  "hazard_norm": -0.85,
  "admit": true
}
```

## 6. Validation Results
This system was validated on **Jan 7-21, 2026** (2 Weeks) across 7 Pairs with **Zero Retraining**.
- **GBP_USD:** +77 Log Return (Sharpe 24.38)
- **USD_CHF:** +56 Log Return (Sharpe 16.25)
- **EUR_USD:** +31 Log Return (Sharpe 15.51)
