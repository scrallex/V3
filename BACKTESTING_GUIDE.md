# SEP System Backtesting Guide

This guide explains how to set up the SEP Trading System on a fresh instance to run backtests using real historical data.

## 1. Setup

### 1.1 Unpack the Archive
Copy the `sep_system_archive.tar.gz` to your new machine and unpack it:
```bash
mkdir sep
tar -xzvf sep_system_archive.tar.gz -C sep
cd sep
```

### 1.2 Start Database
You need a running Redis (Valkey) instance to store historical data. The easiest way is to use Docker Compose:
```bash
# Start only the database
docker compose -f docker-compose.hotband.yml up -d valkey
```

### 1.3 Configure Environment
Ensure your `.env` and `OANDA.env` files are present and contain valid credentials.
The backfill tool needs these to fetch data from OANDA.

## 2. Fetch Historical Data (3 Months)

Use the provided backfill tool to fetch real candle data from OANDA into your local database.

**Command:**
```bash
# Backfill last 3 months (approx 2200 hours)
# Using M5 granularity (standard for this system)
python3 scripts/tools/backfill_candles.py \
  --instruments "EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CHF,USD_CAD,NZD_USD" \
  --granularity M5 \
  --lookback-hours 2200
```
*Note: This may take a few minutes depending on your connection speed.*

## 3. Run Backtest

Once data is in Redis, run the unified backtest simulator.

**Command:**
```bash
# Run simulation over the backfilled period
# Adjust --start and --end to match your testing window
python3 scripts/run_unified_backtest.py \
  --start "$(date -u -d '3 months ago' +%Y-%m-%dT%H:%M:%SZ)" \
  --end "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --nav 100000 \
  --granularity M5 \
  --output my_backtest_results.json
```

### 3.1 Analyzing Results
The results will be saved to `my_backtest_results.json`. You can inspect this file for performance metrics (CAGR, Drawdown, Trades).
