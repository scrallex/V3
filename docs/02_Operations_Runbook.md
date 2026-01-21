# Operations Runbook

The lean SEP stack consists of three pieces: Valkey, the backend (`scripts/trading_service.py`), and an nginx-hosted monitoring dashboard. Follow these steps to keep it healthy.

## Environment & Configuration

| Source | Keys | Notes |
| --- | --- | --- |
| `.env.hotband` | `READ_ONLY`, `VALKEY_URL` | Defaults to read-only trading; Valkey points to the internal container. |
| `OANDA.env` | `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, `OANDA_ENVIRONMENT`, `OANDA_BASE_URL` | Loaded last so credentials override other files. Keep out of version control. |
| Docker compose env | `SERVER_NAME=mxbikes.xyz`, `BACKEND_UPSTREAM=http://backend:8000`, `WS_UPSTREAM=http://backend:8000` | Used by the frontend entrypoint to render nginx/tls config. |
| Runtime env (optional) | `HTTP_HOST`, `HTTP_PORT`, `LOG_LEVEL`, `PORTFOLIO_NAV_RISK_PCT`, etc. | Export before running `deploy.sh` to override defaults. |

Flip to live trading by setting `READ_ONLY=0`, redeploying, then clearing the kill switch via `/api/kill-switch`. Document approvals before doing so.

## Deployment Topology

- **valkey** (`redis:7-alpine`): stores gates, kill switch, risk snapshots. Persistent volume `valkey-data`.
- **backend** (Python 3.12 slim image): exposes REST API on `:8000`, consumes Valkey, talks to OANDA. Mounts `scripts/`, `config/`, `bin/` read-only.
- **frontend** (nginx + Vite bundle): serves the React dashboard on `80/443`, proxies `/api` and `/ws` to backend. Mounts `/etc/letsencrypt` and `/var/www/certbot` to read Let’s Encrypt certs.
- **regime** (Python 3.12 worker): polls `md:candles:{instrument}:M1`, computes market manifolds, publishes the enriched gate payloads (`gate:last:{instrument}`) with adaptive hazard thresholds and Prometheus metrics on `9105`.

All services are orchestrated with `docker-compose.hotband.yml`. `deploy.sh` builds images, restarts containers, and performs a `/health` check before reporting success.

## Data Pipeline Responsibilities

The compose stack does **not** run the candle downloader or manifold generator. Ensure external cron/systemd services execute the candle streamer (`python scripts/tools/stream_candles.py`, wired via `sep-data-downloader.service`) and `bin/manifold_generator`, writing gate payloads into Valkey (`gate:last:{instrument}`). Without these processes the backend will see stale gates and refuse to trade.

## Logging & Monitoring

- Backend logs stream to stdout/stderr; collect them with `docker compose logs backend` or ship to a central sink.
- nginx access/error logs live inside the frontend container. Consider binding a host volume or using a log driver.
- Valkey includes no built-in monitoring. Use Redis CLI (`docker exec sep-valkey redis-cli info`) or integrate with an external monitor.
- Establish alerting (e.g., Datadog/Prometheus) for gate staleness, OANDA failures, expiring certificates, disk usage, and container health.

## TLS & Domain

`apps/frontend/docker-entrypoint.sh` renders the HTTPS server block when `/etc/letsencrypt/live/mxbikes.xyz/fullchain.pem` exists. Certificates are expected to be renewed by a host-level Certbot timer. Verify renewal regularly (`sudo certbot renew --dry-run`).

## Before Enabling Live Trading
1. **Bring up Valkey and the backend**
   ```bash
   docker compose -f docker-compose.hotband.yml up valkey backend
   ```
2. **Confirm broker credentials** – `curl /health` should return `kill_switch:true` on first boot. Then call `/api/status` and `/api/pricing` to verify the service talks to OANDA.
3. **Check gate freshness** – inspect `gate:last:{instrument}` in Valkey. If the key is missing or stale, restart the metrics pipeline before disabling the kill switch.

## During Market Hours
- **Status** – `/health` and `/api/status` report whether the portfolio manager thread is running and which instruments are enabled.
- **Pricing** – `/api/pricing` returns the latest bid/ask/mid map used by the manager.
- **Order flow** – `POST /api/order` and `POST /api/trade/close` are the only mutating endpoints. Both honour `kill_switch` and `READ_ONLY` flags.

## Routine Controls
- **Kill switch** – `POST /api/kill-switch` with body `{ "kill_switch": true }` pauses trading immediately.
- **Manual close** – `POST /api/trade/close` with `{ "instrument": "EUR_USD" }` flattens a pair. Omit `units` to close everything.
- **Manual candle fetch** – `POST /api/candles/fetch` with `{ "instrument": "GBP_USD", "granularity": "M5", "count": 200 }`.
- **Hazard breaches** – the regime worker emits `hazard_fallback_requested` in `gate:last:{instrument}` when λ exceeds the adaptive threshold×1.5. Treat this as a soft kill-switch: ensure the `/api/kill-switch` is engaged until the fallback clears or optical validation succeeds.

## Troubleshooting
| Symptom | Action |
| --- | --- |
| Kill switch refuses to clear | Service booted without OANDA credentials; export `OANDA_API_KEY` and `OANDA_ACCOUNT_ID`, then restart. |
| Pricing payload empty | Valkey ok but OANDA down – check network, credentials, or switch to practice mode with `OANDA_ENVIRONMENT=practice`. |
| Orders ignored | `READ_ONLY=1` or kill switch enabled. Inspect `/api/status` and reset the flags. |
| Exposure drift | Call `/api/trade/close` followed by a restart; the risk planner reloads positions on boot. |

Keep this runbook short. If behaviour changes, update this file or delete obsolete steps.

### Logging & Monitoring

- Backend logging now emits JSON to stdout and rotates files under `/app/logs/backend.log` (default 5×5 MB). The compose stack mounts `./logs/backend` so host copies persist. Adjust behaviour with `LOG_FORMAT`, `LOG_TO_FILE`, `LOG_FILE_PATH`, `LOG_MAX_BYTES`, and `LOG_BACKUP_COUNT`.
- nginx access/error logs live inside the frontend container; consider binding a host volume or shipping via a log driver.
- Valkey has no native monitoring—use `docker exec sep-valkey redis-cli info` or integrate with Redis exporters.
- Establish alerting (Prometheus/Datadog/etc.) for gate staleness, OANDA failures, expiring TLS certs, disk usage, and container health.

### Switching Between Paper & Live Trading

Default state: `READ_ONLY=1` with the kill switch engaged—no orders are sent.

To go live:
1. Set `READ_ONLY=0` in `.env.hotband` (or export before deploy).
2. Redeploy (`./deploy.sh` or `docker compose -f docker-compose.hotband.yml up -d --build backend`).
3. Confirm `/health` shows `kill_switch:true`, `trading_active:false` (read-only still protecting you).
4. Disable the kill switch: `curl -X POST https://mxbikes.xyz/api/kill-switch -H 'Content-Type: application/json' -d '{"kill_switch": false}'`.
5. Verify `/api/status` reports `kill_switch:false`, `trading_active:true`; monitor fills closely.

To revert to paper mode, set `READ_ONLY=1`, redeploy, and optionally re-enable the kill switch via the same endpoint.

### Gate Freshness Checks

Use `scripts/tools/check_gate_freshness.py` to monitor `gate:last:*` keys:

```bash
python scripts/tools/check_gate_freshness.py \
  --redis ${VALKEY_URL:-redis://localhost:6379/0} \
  --threshold 180
```

Exit codes: `0` = healthy, `2` = no gates, `3` = stale data. Schedule it via cron and alert if non-zero.

When the gates are fresh, run the signal analytics helper to understand why admits are (or are not) flowing:

```bash
python scripts/tools/signal_analytics.py \
  --redis ${VALKEY_URL:-redis://localhost:6379/0} \
  --lookback-minutes 240 \
  --top-count 5
```

It summarises admit rates, structural averages, the most common rejection reasons, and highlights the strongest setups per instrument. Use `--json` to feed the payload into dashboards or weekly ops reports.

When inspecting payloads, confirm structural metrics trend contractive:
- `structure_metrics.coherence_tau_slope ≤ configured max` (negative)
- `structure_metrics.domain_wall_slope ≤ configured max` (negative)
- `structure_metrics.spectral_lowf_share ≥ configured min` (rising low-frequency power)
Positive slopes or collapsing low-frequency share indicate entropy-spreading behaviour—keep the kill switch engaged until the upstream manifolds recover.

#### Correlation & Stability Diagnostics

Each `gate:last:{instrument}` payload now carries additional stability metrics. Inspect them alongside the classic slopes before clearing the kill switch:

- `reynolds_ratio < 1` ⇒ contractive dynamics. Values > 1 mean perturbations are spreading faster than domain walls are collapsing; do **not** trade until the ratio drops below 1 again.
- `temporal_half_life` (seconds) ⇒ exponential time constant extracted from lag correlations. Healthy tapes sit in the low single digits; spikes signal sticky/chaotic structure.
- `spatial_corr_length` ⇒ effective correlation radius (in cells). Contractive regimes keep this short; growing length foreshadows broad domain walls.
- `pinned_alignment` ⇒ fraction of pinned bits matching their anchors. Use it to confirm the reference-following experiments lock on quickly (ideally ≥ 0.95 within the first few phases).
- `pinned_lock_in_phase` ⇒ first phase index where all anchors aligned; treat large values as a warning that the manifold is struggling to stabilise.

Guardrail keys are available in `config/echo_strategy.yaml`:

```yaml
guards:
  max_reynolds_ratio: 1.0
  min_temporal_half_life: 1.5
  min_spatial_corr_length: 0.80
  min_pinned_alignment: 0.93
```

The portfolio manager rejects gates that violate these thresholds and annotates the decision reason (`reynolds_above_max`, `temporal_half_life_below_min`, etc.). Adjust values per instrument as more data accumulates.

### Data Pipeline Summary

1. `scripts/tools/stream_candles.py` fetches recent OANDA candles (using the credentials in `.env.hotband`) and stores them under `md:candles:{instrument}:M1` in Valkey.
2. `bin/manifold_generator` (C++ metrics engine) reads those candles, computes quantum metrics, and writes signal blobs plus the latest gate payload to `gate:last:{instrument}`.
3. The backend polls `gate:last:*`, applies the strategy profile and session policy, and requests trades through OANDA when admissible.

Ensure steps 1 and 2 are supervised outside of docker-compose (systemd timers, cron, or a separate orchestrator). Without them the backend will see stale gates and stay idle.

### Systemd Integration Templates

Example service files live in `ops/systemd/`:

- `sep-data-downloader.service` – runs `/usr/bin/env python3 /sep/scripts/tools/stream_candles.py` on boot (pulls instruments from `HOTBAND_PAIRS`).
- `sep-manifold.service` – runs `/sep/bin/manifold_generator --config /sep/config/echo_strategy.yaml` after the downloader.

Install on the host with:

```bash
sudo cp ops/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sep-data-downloader.service sep-manifold.service
```

Review and adjust `User`, paths, and CLI flags before enabling.

### Auxiliary Key Initialization

The backend expects certain Valkey keys (e.g., `ops:kill_switch`) to exist. Use:

```bash
python scripts/tools/check_aux_keys.py --redis ${VALKEY_URL:-redis://localhost:6379/0}
```

If keys are missing, initialize them manually:

```bash
redis-cli -u ${VALKEY_URL} set ops:kill_switch 1
```

### Scheduled Gate Monitoring

Use `ops/cron/gate_freshness_to_slack.sh` to run the freshness check every minute and post to Slack on failure:

```
* * * * * root VALKEY_URL="redis://valkey:6379/0" \
  SLACK_WEBHOOK="https://hooks.slack.com/services/..." \
  /sep/ops/cron/gate_freshness_to_slack.sh
```

The script also logs to syslog (`sep-gate-freshness`) so traditional monitoring can ingest it.

### Weekly Signal Study (replaces automated backtests)

Every Tuesday (or whenever you pick) produce a fresh seven-day signal study and publish it to the dashboard. The flow is:

1. **Rebuild gate history for the window**
   ```bash
   docker compose -f /sep/docker-compose.hotband.yml exec -T backend \
     python /app/scripts/tools/backfill_gate_history.py \
       --start "2025-11-03T00:00:00Z" \
       --end   "2025-11-10T00:00:00Z" \
       --redis redis://valkey:6379/0 \
       --flush \
       --export-json /app/logs/backend/gates_2025-11-03_to_2025-11-10.jsonl
   ```
   Adjust the window to whatever period you are studying (default is the prior seven days if `--start/--end` are omitted).

2. **Correlate those gates with OANDA prices**
   ```bash
   docker compose -f /sep/docker-compose.hotband.yml exec -T backend \
     python /app/scripts/tools/signal_outcome_study.py \
       --redis redis://valkey:6379/0 \
       --start "2025-11-03T00:00:00Z" \
       --end   "2025-11-10T00:00:00Z" \
       --horizons "5,15,30,60,240" \
       --cost-model median_spread_plus_half_slip \
       --export-json /app/docs/evidence/outcome_weekly_costs.json
   ```
   Point `SIGNAL_EVIDENCE_PATH` at that exported JSON (default `docs/evidence/outcome_weekly_costs.json`). The backend caches the latest file and serves it at `/api/evidence/signal-outcomes`, which the dashboard renders as “Weekly Signal Analytics.”

3. **Archive and announce**: copy the freshly generated JSON to a dated filename (`docs/evidence/outcome_weekly_costs_2025-11-03_to_2025-11-10.json`) before overwriting the canonical path. Reference the snapshot in ops notes / Slack so everyone knows which study the dashboard reflects.

4. **Promote bundle evidence** by rebuilding the activation tape, simulator recap, and dashboard JSON in one go:
   ```bash
   make bundle-study
   ```
   This command runs `scripts/research/build_bundle_activation_tape.py`, `scripts/research/bundle_activation_simulator.py`, and `scripts/tools/bundle_outcome_study.py`, emitting artefacts under `output/strand_bundles/` plus `docs/evidence/bundle_outcomes.json`. The backend serves this JSON at `/api/evidence/bundle-outcomes` so the dashboard’s “Bundle Readiness” panel reflects MB003/NB001 expectancy before we consider any live change.

5. **Verify the dashboard** once both evidence files land. The hero metrics should show fresh timestamps, the Weekly Signal Analytics panel should reference the new window, and the bundle panel should confirm MB003/NB001 are positive after costs. If CB002 is blocking a pair, keep the kill switch engaged until the quarantine clears.

No `/api/backtests/*` endpoints or cron hooks remain in the critical path; do not schedule grid backtests until you have a signal study that justifies them.

### Semantic Regime Research

- Run `make discover-semantic-regimes` to sweep the canonical tag set (`highly_stable`, `improving_stability`, `high_rupture_event`, `chaotic_price_action`, `low_hazard_environment`).
- Override the backtest definition with `CONFIG=/path/to/config.json` (defaults to `configs/research/semantic_pilot.json`). Pass additional discovery flags via `DISCOVER_ARGS`.
- Each sweep shells out to `make unified-backtest`, enforcing the semantic filter via `--semantic-regime-filter` and writing per-instrument JSON snapshots under `output/semantic_runs/`.
- Aggregated metrics land at `output/semantic_regime_summary.csv`; sanity-check Sharpe, drawdown, and tag utilisation before updating the live strategy profile.

### Valkey Default Seeding

`deploy.sh` automatically seeds `ops:kill_switch` (set to `1`) and `risk:nav_snapshot` if they are missing by executing `/app/scripts/tools/seed_valkey_defaults.py` inside the backend container. Run manually when needed:

```bash
docker compose -f docker-compose.hotband.yml exec backend \
  python /app/scripts/tools/seed_valkey_defaults.py --force
```

### Valkey Retention Guidelines

- **Candles (`md:candles:{instrument}:M1`)**: one 1-minute bar per instrument (~140 bytes). For 7 FX pairs this is ~1.4 MB/day. Retain 30 days (~45 MB) by scheduling `redis-cli --scan` cleanup or using `VALKEY_CANDLE_RETENTION` (set in `.env.hotband`, default `0` = unlimited). Recommendation: run a weekly prune script to delete entries older than 30 days.
- **Gate snapshots (`gate:last:{instrument}`)**: single JSON per instrument (<1 KB). No significant storage impact; ensure they’re updated every loop.
- **Risk snapshots / telemetry**: currently minimal. If future history keys are added, adopt a 90-day retention unless compliance dictates longer.

Example prune command (keep 30 days of candles):

```bash
redis-cli -u ${VALKEY_URL} --scan --pattern 'md:candles:*' | \
while read key; do
  redis-cli -u ${VALKEY_URL} ZREMRANGEBYSCORE "$key" -inf $(( ( $(date +%s) - 30*86400 ) * 1000 ))
done
```

Document the chosen retention policy and schedule in your ops calendar.

### Weekly Roster Policy

We now key roster decisions off the **weekly signal study**, not grid PnL:

- Require **≥200 samples** and **positive% ≥ 55 %** at the 60 m or 240 m horizon before keeping an instrument in `HOTBAND_PAIRS`.
- Treat AUROC < 0.55 or Holm-adjusted p-value ≥ 0.05 as “unproven.” Tighten the strategy profile (hazard caps, semantic filters) and re-run the study before re-enabling.
- If an instrument posts two consecutive weeks with negative average return (after costs) at every horizon, remove it until the signal study shows improvement.

Document the roster decision each week alongside the archived JSON so future backtests know which periods had what eligibility.

### Research Environment

A lightweight toolkit lives under `scripts/research/` and `research/`:

1. Create a venv and install deps:
   ```bash
   cd /sep/research
   make env
   ```
2. Export gates to CSV/JSONL:
   ```bash
   make export-gates OUTPUT=../output/gates.csv REDIS=redis://valkey:6379/0
   ```
3. Run baseline admit analysis:
   ```bash
   make simulate-day INPUT=../output/gates.csv NAV=10000
   ```

4. Download historical candles:
   ```bash
   ./../scripts/research/fetch_history.py EUR_USD 2024-01-01T00:00:00Z 2024-03-01T00:00:00Z --granularity M5 --output ../data/EUR_USD_M5.csv
   ```

Extend `simulate_day.py` to compute PnL once sizing rules are finalised.

### Deferred Backtest Grid (use only after signals are proven)

The historical grid runner still lives under `research/`, but it is no longer part of the weekly cadence. Only touch it when the signal study highlights a concrete hypothesis. The flow when (and if) you need it:

1. Identify the hypothesis from the latest `outcome_weekly_costs*.json` (e.g., “EUR/USD 240 m lift justifies widening hazard caps”).
2. Manually run the grid in the research env:
   ```bash
   cd /sep/research
   make backtest-grid \
     REDIS="${VALKEY_URL:-redis://localhost:6379/0}" \
     OUTPUT="../output/backtests/<label>.json" \
     GRID="../config/backtest_grid.json" \
     START="2025-11-03T00:00:00Z" \
     END="2025-11-10T00:00:00Z" \
     INSTRUMENTS="EUR_USD"
   ```
3. Review the artefact locally; do **not** pipe it back into the dashboard or `/api/backtests/*`. Document outcomes in `docs/Task.md` and tie any future profile/risk changes to both the signal study and the one-off backtest.

This keeps the production surface focused on signals while still giving research a path to deeper analysis when warranted.
