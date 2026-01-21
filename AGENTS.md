# SEP Codex Guide

This doc is the single source of truth for how the simplified SEP stack now operates. It reflects the signal-first plan: **understand the gate stream, correlate it with history, then (and only then) stage backtests / execution changes.**

## Architecture Snapshot
- Flow: external candle ingest (`bin/data_downloader`) → manifold encoder (`scripts/trading/regime_manifold_service.py`) → Valkey (`gate:last:{instrument}` / `gate:index:{instrument}`) → `PortfolioManager` → OANDA connector.
- Gate evidence is promoted weekly through the **signal study** pipeline (`scripts/tools/backfill_gate_history.py` + `scripts/tools/signal_outcome_study.py`). The latest JSON lives at `SIGNAL_EVIDENCE_PATH` and drives the dashboard.
- No rolling evaluator, quant allocator, or research sidecars remain; everything between Valkey and OANDA is Python and intentionally small.

## Module Orientation
| Area | File(s) | Notes |
| --- | --- | --- |
| Service bootstrap | `scripts/trading_service.py` | Creates the OANDA connector, `PortfolioManager`, HTTP API, and evidence cache. |
| Portfolio + risk | `scripts/trading/portfolio_manager.py`, `scripts/trading/risk_planner.py` | Gate loading, session policy, trade stack, and exposure bookkeeping. |
| Signal tooling | `scripts/tools/signal_analytics.py`, `signal_outcome_study.py`, `backfill_gate_history.py` | CLI helpers for inspecting the live stream and producing multi-horizon studies. |
| Native metrics | `src/core/*`, `bin/manifold_generator` | C++ encoder that feeds the manifold service; rarely touched unless metric math changes. |
| Frontend | `apps/frontend` | React/Vite dashboard that surfaces health, gates, and the weekly signal analytics panel. |
| Config/docs | `config/echo_strategy.yaml`, `docs/01_System_Concepts.md`, `docs/02_Operations_Runbook.md`, `docs/03_Signal_Analytics.md` | Keep these synchronized with code changes. |

## Signal-First Workflow
1. **Validate gate freshness** via `scripts/tools/check_gate_freshness.py` (automated or manual).
2. **Backfill + study**: run `backfill_gate_history.py` for the desired window, then `signal_outcome_study.py --export-json docs/evidence/outcome_weekly_costs.json`. This JSON is the authoritative weekly snapshot.
3. **Review / publish**: surface the study in the dashboard (Weekly Signal Analytics panel) and reference it in runbooks. Archive every snapshot.
4. **Only after signal confidence is established**, design targeted backtests (grid runner lives under `scripts/research/` for that phase) and consider profile/risk changes.

Everything we deploy, document, or visualize should support steps 1‑3 before we touch step 4.

## Development Workflow
- `make install` sets up Python deps; `make frontend-install` for the dashboard.
- Local stack: `docker compose -f docker-compose.hotband.yml up valkey backend` (add `regime` when you need fresh gates).
- Tests: `make lint`; targeted `pytest` for modules you touch.
- Deploy: `./deploy.sh` builds/publishes backend + frontend and seeds Valkey defaults.

## Coding & Ops Expectations
- Python: Black/flake8 style, 4 spaces, 120 cols max. Add comments only when logic isn’t obvious.
- C++: match existing `src/core` formatting; rebuild via `make build-core` if touched.
- Secrets stay in `OANDA.env` / `.env.hotband`. Never commit real credentials.
- Kill switch lives in `ops:kill_switch`; confirm its state before allowing live orders.
- Monitoring focus: gate freshness, OANDA connectivity, signal evidence generation, and dashboard health.

Keep this file updated whenever modules move or workflow expectations shift; stale guidance here wastes everyone’s time.
