# ML Gate-Stream Modeling Plan

## Overview
Objective: quantify how modern ML models (GBMs, LSTMs, Transformers) respond to the SEP gate stream and promote the winners into the lean trading stack. We split the workflow:

1. **Stage 1 – Droplet (“lab” stack with 2020‑2025 history + live Valkey/OANDA wiring).** Codex on the droplet handles reproducible data exports and baseline analytics directly where the raw evidence already lives.
2. **Stage 2 – Local workstation (this repo).** After copying the curated datasets here we run heavier training (GPU, Hugging Face, etc.), fit sequence models, and pipe scores back to the simulator/portfolio manager.

The rest of this doc is the cookbook for both stages plus the cross-stage handoff.

---

## Stage 1 — Droplet Data Prep & Validation
_Owner_: Codex instance on droplet  
_Source repos_: `/sep/trader` (plus legacy evidence dirs already on disk)  
_Key commands_: `make install`, scripts under `scripts/tools` and `scripts/research`

### 1. Environment & Sanity Checks
- `make install` (or ensure the existing venv is current).  
- Confirm OANDA + Valkey credentials loaded via `OANDA.env`, `docker compose -f docker-compose.hotband.yml up valkey backend regime`.  
- `python scripts/tools/check_gate_freshness.py --redis ${VALKEY_URL}` – record status.

### 2. Regenerate / Consolidate Gate Evidence (2020‑2025)
1. **Weekly replays**  
   ```bash
   python scripts/tools/backfill_gate_history.py \
     --start 2020-11-13T00:00:00Z --end 2025-11-11T00:00:00Z \
     --profile config/echo_strategy.yaml \
     --export-json docs/evidence/roc_history/gates_{start}_{end}.jsonl \
     --export-roc-summary docs/evidence/roc_history/roc_summary_{start}_{end}.json
   ```  
   - Use automation in `scripts/research/signal_regime_workflow.py` / `run_span_backfills.py` to iterate weeks.  
   - Capture both rolling + `--isolation` variants (per docs/08).
2. **Canonical parquet/duckdb build**  
   - Merge all `gates_with_roc*.jsonl` + `roc_summary*.json` into `output/ml_corpus/gates_2020_2025.parquet`.  
   - Columns: instrument, ts, admit, reasons, structure.*, canonical.*, hazard_decile, repetitions, semantic_tags, roc_forward_pct (5/15/30/60/240), session info.  
   - Include `run_mode` (rolling vs isolation) + `source`.
3. **Compression of decoded bitplanes (optional)**  
   - Use `research/regime_manifold/codec.decode_window_bits` to store sequences per gate if needed.

### 3. Baseline Analytics
- Re-run `python scripts/research/lead_lag_regression.py --roc-dir docs/evidence/roc_history --dataset-csv output/ml_corpus/lead_lag_features.csv --window-weeks 26`.  
- Export gradient-boosted baselines (LightGBM/XGBoost) on tabular features, log metrics (AUC, precision/recall, ROC delta) into `output/ml_corpus/baseline_report.json`.  
- Update evidence summaries (`docs/04`, `docs/05`, `docs/06`) with any new span stats if values shift.

### 4. Packaging for Transfer
- Tar the prepared assets:
  ```
  tar -czf output/ml_corpus_2020_2025.tgz \
    docs/evidence/roc_history \
    output/ml_corpus/*.parquet \
    output/ml_corpus/*.json \
    output/ml_corpus/*.csv
  ```
- Generate SHA256 manifest.  
- SCP bundle to local workstation (`scp user@droplet:/sep/output/ml_corpus_2020_2025.tgz ./data/ml_corpus_2020_2025.tgz`).  
- Record manifest + export log in `handoff_base/02_signals_and_research/transfer_log.md`.

### 5. Status Report
- Drop `Stage1_status_<date>.md` summarizing: gate replay coverage %, dataset sizes, baseline metrics, any blockers (e.g., missing ROC spans). Include commands used + outputs.

---

## Stage 2 — Local Modeling & Integration
_Owner_: Codex here (GPU workstation)  
_Source repos_: `/sep/trader`, `/sep/hf_repo`, STM assets, huggingface cache

### 1. Setup & Data Ingest
- Verify CUDA/PyTorch stack (`make install` + optional `pip install lightgbm xgboost pytorch-lightning transformers datasets`).  
- Untar corpus into `data/ml_corpus/`. Validate checksums.  
- Load parquet/duckdb into a local `mlflow` or plain notebooks; document schema in `docs/telemetry_schema.md`.

### 2. Feature Pipelines
- Implement a `scripts/ml/feature_builder.py` that:
  - Reads parquet, normalises metrics, encodes categorical fields (instrument, run_mode, hazard_decile).  
  - Generates sequence tensors (length 32/64) with sliding windows per instrument.  
  - Saves train/val/test splits (by week) into `output/ml_training/{split}.pt` or parquet.
- Reuse canonical features from Stage 1; include optional decoded bitplanes.

### 3. Modeling Tracks
1. **Tabular Baseline Refresh**  
   - Train LightGBM/XGBoost on per-gate features; log metrics vs Stage 1 baseline.  
   - Save models (`output/ml_models/baseline_lgbm.txt`) + shapley plots.
2. **Sequence Models**  
   - LSTM/GRU: PyTorch module predicting forward ROC sign/magnitude; log to `output/ml_models/lstm_*`.  
   - Transformer/TFT: use PyTorch Lightning; track validation AUC, calibration.  
   - Hyperparam sweeps with `wandb` or local JSON logs; archive configs.
3. **Evaluation Harness**  
   - Extend `scripts/research/simulator/backtest_simulator.py` (if missing locally) to accept `--ml-scores path` and replay sample weeks (26‑week set).  
   - Compare NAV, Sharpe, drawdowns vs heuristics; store results in `output/ml_backtests/`.

### 4. Integration Plan
- Add `ml_score` ingestion into `scripts/trading/portfolio_manager.py` (feature flag).  
- Define guardrail: e.g., require `ml_score >= threshold` + hazard guard before sizing.  
- Draft ops doc update for `docs/02_Operations_Runbook.md` describing monitoring & rollback.

### 5. Deliverables
- `docs/ML_Modeling_Report_<date>.md` summarizing dataset stats, model configs, metrics, backtest impacts.  
- Versioned artifacts (`output/ml_models/`, `output/ml_training/`).  
- Git diffs for feature builder, simulator hooks, docs.

---

## Cross-Stage Checklist
| Item | Droplet | Local |
| --- | --- | --- |
| Gate replays + ROC summaries | ✅ (Stage 1) | ↘ consume |
| Parquet corpus + manifests | ✅ produce | ✅ verify, ingest |
| Baseline analytics | ✅ produce | ✅ extend (seq models) |
| Model training | – | ✅ LGBM/LSTM/Transformer |
| Simulator with ML scores | prepare config refs in docs (`handoff_base/01_system_stack/Task.md`) | implement + test |
| Deployment hooks | update docs/ops notes, tag git commit | integrate + plan release |

When both stages are complete, run an end-to-end rehearsal: droplet exports latest week → SCP to local → retrain/infer → push `ml_score` thresholds back via profile updates → optional redeploy of backend with new scoring.

---

## Immediate Next Actions
1. Approve Codex-on-droplet to start Stage 1 checklist (env sanity, weekly replays, corpus build).  
2. Here: prep GPU env + storage, create `data/ml_corpus` target, and outline feature builder scaffolding.  
3. After first SCP, validate dataset integrity and begin baseline model notebooks before jumping into LSTMs/Transformers.
