# System Overview

The lean SEP stack keeps just enough moving parts to turn quantum fingerprints into trades:

1. **Metrics (C++ core).** `src/core` consumes OANDA candles and emits coherence, stability, entropy, hazard, plus the extended structural suite: `coherence_tau_slope`, `domain_wall_slope`, `spectral_lowf_share`, the Reynolds-like stability ratio, correlation half-life, and pinned-alignment diagnostics. Fresh results are mirrored into Valkey under `gate:last:{instrument}` by the metrics pipeline.
2. **Portfolio manager.** `scripts/trading/portfolio_manager.py` pulls each gate snapshot, evaluates session windows, and assigns a target direction/size.
3. **Risk planner.** `scripts/trading/risk_planner.py` guards exposure and translates desired sizing into concrete units.
4. **Execution.** Orders are forwarded through the minimal OANDA connector (`scripts/trading/oanda.py`), with results recorded back into the risk inventory.

## Core Modules

| Area | File | Responsibility |
| --- | --- | --- |
| Portfolio | `scripts/trading/portfolio_manager.py` | Gate polling, session logic, and trade orchestration. |
| Risk & trades | `scripts/trading/risk_planner.py` | Risk limits, sizing, and trade stack maintenance. |
| Broker | `scripts/trading/oanda.py` | Thin REST wrapper for pricing, orders, and positions. |
| API | `scripts/trading/api.py` | JSON health/pricing surface for operators. |
| Service | `scripts/trading_service.py` | Boots the connector, manager, weekly evidence cache, and HTTP server. |
| Bundles | `config/bundle_strategy.yaml`, `scripts/research/bundle_rules.py`, `scripts/tools/bundle_outcome_study.py` | Defines MB003/NB001/CB002 filters, annotates gates with bundle hits, and produces the bundle evidence JSON served to the dashboard. |
| Signal studies | `scripts/tools/backfill_gate_history.py`, `scripts/tools/signal_outcome_study.py` | Rebuild historical gates and correlate them with price moves to produce the weekly signal analytics snapshot that informs every decision. |

## Valkey Keys in Use

- `gate:last:{instrument}` – latest gate snapshot used by the manager.
- `ops:kill_switch` – runtime kill switch toggled by the service.

Everything else from the original telemetry stack has been removed.

## Operational Notes

- Only the backend service, Valkey, the manifold worker, and the optional dashboard need to run. Rolling evaluators and research sidecars are gone.
- Risk sizing assumes a flat `EXPOSURE_SCALE` (default `0.02`) and divides the NAV risk budget equally across admitted instruments. Tune via `PORTFOLIO_NAV_RISK_PCT`, `PM_MAX_PER_POS_PCT`, and `PM_ALLOC_TOP_K`.
- Session windows are defined in `config/echo_strategy.yaml`. Temporary overrides can be pushed via the HTTP API or by editing the strategy profile.
- Strategy guardrails understand the new metrics. Configure `max_reynolds_ratio`, `min_temporal_half_life`, `min_spatial_corr_length`, and `min_pinned_alignment` per instrument to block unstable gates before they reach the risk planner.
- Bundle directives live alongside the strategy profile. `config/bundle_strategy.yaml` drives the `regime_manifold_service` (to tag gates) and the portfolio manager (to size MB003 promotions, NB001 fades, and CB002 quarantine blocks). Refresh `docs/evidence/bundle_outcomes.json` with `make bundle-study` before changing sizing or going live.
- Generate and archive the weekly signal evidence (`docs/evidence/outcome_weekly_costs*.json`) before discussing any backtest or live change; the dashboard consumes this file directly.
- Before enabling live trading, confirm `/health` returns `kill_switch: false`, fetch fresh pricing, and monitor fills via the `/api/order` and `/api/trade/close` endpoints.
