# Stage 1 Status — 2025-11-17

## Environment & Connectivity
- `make install` re-ran inside `.venv` (Python 3.12) to confirm requirements; added `pyarrow`, `statsmodels`, `lightgbm`, `xgboost`, and `scikit-learn` for corpus + baseline tasks.
- Droplet services: `sep-valkey`, `sep-backend`, `sep-regime`, and `sep-frontend` are running (`docker ps` @ 01:57 UTC).
- Gate freshness: `.venv/bin/python scripts/tools/check_gate_freshness.py --redis redis://localhost:6379/0` reports **7 stale instruments (~74 h)**—Valkey has payloads but the live manifold/connector needs a restart before future admits (`logs/signal/gate_health_2025-11-17.json`).

## Data Exports
- Weekly ROC history already covered 13 Nov 2020 → 14 Nov 2025; the Nov 7→14 gap was replayed directly from OANDA (`gates_with_roc_2025-11-07_to_2025-11-14.jsonl`, `roc_summary_2025-11-07_to_2025-11-14.json`).
- Canonical corpus build (`output/ml_corpus/gates_2020_2025.parquet`): 2,298,105 gates combining weekly rolling (868,431), span rolling (714,837), and span isolation (714,837) rows with full structural metrics, semantic tags, hazard deciles, and multihorizon ROC labels (`output/ml_corpus/corpus_summary.json`).
- Lead/lag dataset regenerated via `PYTHONPATH=/sep .venv/bin/python scripts/research/lead_lag_regression.py --roc-dir docs/evidence/roc_history --dataset-csv output/ml_corpus/lead_lag_features.csv --window-weeks 26`.

## Baseline Analytics
- Tabular baselines (weekly subset only) using LightGBM and XGBoost on hazard + structural features; time-based split at 80th percentile of `ts`.
  - **LightGBM:** ROC AUC 0.505, AP 0.505, precision 0.503 / recall 0.516 at 0.5 threshold, ROC delta (top decile) +0.0021.
  - **XGBoost:** ROC AUC 0.507, AP 0.507, precision 0.505 / recall 0.516, ROC delta +0.0028.
- Predictions for the hold-out slice logged in `output/ml_corpus/baseline_predictions.csv`; metrics live in `output/ml_corpus/baseline_report.json`.

## Packaging & Transfer Prep
- Full evidence bundle tarred per plan: `output/ml_corpus_2020_2025.tgz` (includes `docs/evidence/roc_history/**`, corpus parquet, lead/lag CSV, baseline report/predictions). SHA256: `5444b70e1e33fcd1e38365f16b97291a89deafd63e2e59de3690ce9eaa1e6a43` (logged in `output/ml_corpus_2020_2025.tgz.sha256` and `handoff_base/02_signals_and_research/transfer_log.md`).
- SCP to the workstation is pending (blocked on ops approval + Valkey refresh).

## Blockers / Follow-ups
1. **Valkey freshness:** restart manifold/regime services (or resume live streamer) so gate timestamps drop below the 5 h SLA before the next admit cycle.
2. **Lead/lag warnings:** Statsmodels raised overflow/perfect-separation warnings on the 26-week fit; acceptable but note in downstream labs.
3. **Stage 2 handoff:** once SCP cleared, verify checksum locally and begin feature-builder scaffolding (`data/ml_corpus`). 
4. **Automation:** capture these commands into a reusable script (`scripts/tools/export_ml_corpus.py`) for future weekly refresh (tracked for next iteration).
