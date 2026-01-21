# SEP Minimal Trading Stack

This repository contains a lean, production-focused trading loop:

- **Native metrics** (`src/core`) transform candles into coherence/stability/entropy fingerprints.
- **Portfolio manager** (`scripts/trading/portfolio_manager.py`) loads gate payloads, applies session rules, and requests fills through the broker.
- **Risk planner** (`scripts/trading/risk_planner.py`) tracks exposure and builds trade stacks.
- **Broker bridge** (`scripts/trading/oanda.py`) wraps the handful of OANDA REST calls we rely on.
- **HTTP API** (`scripts/trading/api.py`) exposes a tiny monitoring surface.
- **Service runner** (`scripts/trading_service.py`) glues everything together.

Everything else—adaptive controls, research tooling, and heavyweight documentation—has been removed so the stack focuses on one job: **Candles → Gate → Trade**.

## Quick Start

1. **Install dependencies**
   ```bash
   make install
   ```
2. **(Optional) install frontend deps**
   ```bash
   make frontend-install
   ```
3. **Configure OANDA** – export `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, and (if you want to trade live) set `READ_ONLY=0`.
4. **Run the service**
   ```bash
   make start
   ```
5. **Watch the dashboard** – the frontend (`apps/frontend`) is a single monitoring view. Build it with `make frontend-build` and serve the static bundle with any web server.

## Signal Analytics

The entire roadmap is signal-first: understand what the gate stream is doing, correlate it with forward returns, then decide what (if anything) deserves a backtest.

1. **Spot-check the live stream**
   ```bash
   python scripts/tools/signal_analytics.py \
     --redis ${VALKEY_URL:-redis://localhost:6379/0} \
     --instruments EUR_USD,USD_JPY \
     --lookback-minutes 180 \
     --top-count 5
   ```
   This surfaces admit rates, structural averages, and the most common rejection reasons for the current buffer.

2. **Publish the weekly study**
   ```bash
   # Rebuild gates for the target window (e.g., past 7 days)
   python scripts/tools/backfill_gate_history.py --lookback-days 7 --redis ${VALKEY_URL}

   # Correlate gates with OANDA prices across multiple horizons
   python scripts/tools/signal_outcome_study.py \
     --redis ${VALKEY_URL} \
     --horizons "5,15,30,60,240" \
     --cost-model median_spread_plus_half_slip \
     --export-json docs/evidence/outcome_weekly_costs.json
   ```
   The resulting JSON (or whatever path you set via `SIGNAL_EVIDENCE_PATH`) is what the dashboard renders in the “Weekly Signal Analytics” panel. Archive every snapshot before regenerating the next one.

Until the evidence from step 2 clearly supports a hypothesis, don’t touch any backtest or live configuration knobs.

## Research Toolkit

Create a virtualenv and run exports/backtests:

```bash
cd research
make env
make export-gates OUTPUT=../output/gates.csv REDIS=redis://valkey:6379/0
make simulate-day INPUT=../output/gates.csv NAV=10000
../scripts/research/fetch_history.py EUR_USD 2024-01-01T00:00:00Z 2024-03-01T00:00:00Z --granularity M5 --output ../data/EUR_USD_M5.csv
```

Extend `scripts/research/simulate_day.py` to compute full trade/PnL metrics as strategy rules evolve.

## Architecture Snapshot

```
Candles → gate:last:{instrument} → PortfolioManager → OANDA
```

- `PortfolioManager` polls Valkey for the latest gate payload, applies the strategy profile, honours session windows, and asks the risk planner for target units.
- `RiskPlanner` returns net targets while guarding against exposure and duplication.
- `TradingService` submits the delta via the OANDA connector (unless the kill switch or read-only mode blocks it).

## Repository Layout

```
config/                  # echo_strategy.yaml and runtime defaults
scripts/trading/
  api.py                 # Minimal JSON HTTP server
  oanda.py               # OANDA connector + helper functions
  portfolio_manager.py   # Execution loop, sessions, gate reader
  risk_planner.py        # Risk sizing + trade planning primitives
scripts/trading_service.py
apps/frontend/           # Lightweight monitoring dashboard
src/core/                # C++ metrics engine (unchanged)
docs/                    # Snapshot overview + operations runbook
```

## Docs

- System overview: `docs/01_System_Concepts.md`
- Operations runbook: `docs/02_Operations_Runbook.md`
- Signal analytics playbook: `docs/03_Signal_Analytics.md`

Keep these references aligned with the code—if behaviour changes, update the docs or delete the stale sections.
