## SEP Signal Regime Restructure Workflow

This playbook captures how to execute the “signal-first” restructuring that links the legacy 26‑week validation cut to the five‑year longitudinal study. The intent is to (1) keep every artefact reproducible directly from `/sep/docs/evidence/roc_history`, (2) prove that isolation runs converge to the rolling baseline, and (3) stage the results for the whitepaper + dashboard refresh.

### Phase Overview

| Phase | Goal | Outputs |
| --- | --- | --- |
| 1. Preparation | Load every `roc_summary_*.json` and materialize a canonical five-year frame. | `docs/evidence/roc_history/gameplan/processed_5y_roc.pkl` |
| 2. Span detection | Slice the five-year window into intentional multi-month spans keyed off the mean-revert 60 m z-score meta regime. The first ≥26 w span becomes the validation anchor. | `docs/evidence/roc_history/gameplan/span_catalog.json` |
| 3. Subset backfills | Re-run `backfill_gate_history.py` per span (rolling + isolation) and export gates + ROC summaries. | `docs/evidence/roc_history/gameplan/subsets/*.json[l]`, `.../isolation/*.json[l]` |
| 4. Isolation studies | Replay `signal_outcome_study.py` + `simulate_day.py` on the exported gates to quantify admit quality and trade stack behavior without rolling history. | `.../subsets/outcomes_*.json`, `.../isolation/outcomes_*.json`, sim logs |
| 5. Rolling vs isolation comparison | Diff admit stats + ROC lifts between the two modes to quantify stacking influence. | `docs/evidence/roc_history/gameplan/comparisons/stacking_*.json` |
| 6. Documentation | Roll the findings into the signal regime whitepaper, dashboard copy, and this folder’s `final_gameplan_results.md`. | docs + dashboard PRs |

### Updated Tooling

- `scripts/tools/backfill_gate_history.py` now accepts `--isolation`, which swaps Valkey writes for an in-memory stub and resets hazard/signature buffers before each instrument. Every JSONL row now carries `run_mode`.
- `scripts/tools/signal_outcome_study.py` now supports offline analysis:
  - `--input /path/to/gates.jsonl` (or `--gates-file`) reads the exported gates directly.
  - `--price-mode embedded` (auto when `--input` is supplied, or via `--isolation`) consumes the stored `roc_forward_pct` values instead of hitting OANDA.
  - The CLI still falls back to live Valkey + broker candles when no input file is supplied.

### Running the Workflow

1. **Generate the orchestration assets**

   ```bash
   python scripts/research/signal_regime_workflow.py \
     --roc-dir docs/evidence/roc_history \
     --output-root docs/evidence/roc_history/gameplan \
     --meta-regime mean_revert --target-horizon 60 \
     --min-span-days 90 --validation-days 182
   ```

   This command produces:

   - `processed_5y_roc.pkl` – pandas DataFrame of every regime/horizon sample.
   - `span_catalog.json` – ordered spans with classification, avg bp, anchor flag.
   - `job_specs.json` – ready-to-run command lists per span (rolling + isolation).

2. **Execute per-span jobs**

   Each span in `job_specs.json` contains three commands per mode:

   - Backfill gates (rolling or `--isolation`) → JSONL + ROC summary.
   - Offline `signal_outcome_study.py --input <gate_file> --price-mode embedded --horizons 5,15,30,60,240`.
   - `simulate_day.py <gate_file> --profile config/echo_strategy.yaml --nav 10000`.

   Run the rolling set first, then the isolation set, so the comparison step has both artefacts.

3. **Compare rolling vs isolation**

   After both runs finish for a span, diff the metrics and drop the result in `docs/evidence/roc_history/gameplan/comparisons/stacking_<span>.json`. The orchestration file already reserves the output paths; a simple notebook or Pandas script can load the two `outcomes_*.json` files, compute absolute deltas (hazard AUROC, positive_pct, admit counts), and serialize them for the dashboard / whitepaper.

4. **Update evidence + docs**

   - Summarize every span inside `docs/evidence/roc_history/gameplan/final_gameplan_results.md`.
   - Fold the span narrative (26 w validation → 5 y longitudinal + isolation proof) into `docs/whitepapers/sep_signal_regime_whitepaper.tex` and the dashboard’s Weekly Signal Analytics panel copy once the comparisons land.

### Notes for Whitepaper + Dashboard

- Lead with the validation anchor span (`2021-04-09 → 2021-12-10`) so it is obvious how the 26w cut validates the 5y study.
- Reference the exact files committed under `docs/evidence/roc_history/gameplan` whenever citing outcomes or robustness claims.
- Document the new gauge robustness variants (hazard thresholds, window adjustments) as you populate the robustness folder; the CLI plumbing already exists, so future PRs just drop the CSVs + interpretation.

### Week 46 Evidence Integration (Nov 2025 Cut)

- **Source bundle.** `docs/09_Signal_Evidence_Update_2025W46.md` captures the refreshed longitudinal + weekly stats after replaying `scripts/tools/backfill_gate_history.py --start 2025-11-07T00:00:00Z --end 2025-11-14T00:00:00Z --profile config/echo_strategy.yaml --export-json docs/evidence/roc_history/gates_with_roc_2025-11-07_to_2025-11-14.jsonl --export-roc-summary docs/evidence/roc_history/roc_summary_2025-11-07_to_2025-11-14.json`. Treat this as the canonical reference for MR +39 bp / neutral −23 bp / chaotic +16 bp heading into the latest Sydney open.
- **Gate freshness guardrail.** Before any weekly automation, run `.venv/bin/python scripts/tools/check_gate_freshness.py --redis redis://localhost:6379/0` and log the result under `logs/signal/gate_health_<date>.json`. If payloads are missing or >5 h old (see `logs/signal/gate_health_2025-11-16.json` for “missing” and `.../gate_health_2025-11-17.json` for “stale”), block admits and either restart the live manifold stack or replay the missing week from OANDA.
- **Dashboard feed.** Point the Weekly Signal Analytics panel at `output/latest_weekly_roc_window.json` (regenerated each Sunday after the backfill) plus `docs/evidence/longitudinal_2020_2025/*` so ops/investors see the same MR/neutral/chaotic drift cited in docs/09. This file now only contains full seven-day windows; partial cuts are forbidden.
- **Allocator rules.** Carry over the Week 46 guardrails: (1) require the rolling 14-day MR 60 m average ≥ 0 bp before scaling size, (2) keep neutral as an exit regime until monthly 60 m ≥ +20 bp (Nov is −30 bp), and (3) only trade chaotic regimes with simultaneous positive 60 m/360 m drift and enriched-strand confirmation. Instrument-level alerts should reference `output/latest_weekly_roc_window.json` and `docs/evidence/roc_horizon_rollup.json`.
- **Runbook automation.** The Sunday workflow is now: gate freshness check → `backfill_gate_history.py` replay if needed → `roc_longitudinal_summary.py`, `roc_horizon_rollup.py`, `signal_regime_workflow.py`, `analyze_regime_comparison.py` → publish docs/09 update + dashboard asset refresh. Capture any deviations or manual interventions in the `docs/09_*.md` series so the whitepaper + ops copy stay synchronized.
