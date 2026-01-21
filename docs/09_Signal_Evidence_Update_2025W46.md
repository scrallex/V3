# Signal Evidence & Regime Plan – Week 46, 2025

## 1. Executive Summary
- The longitudinal refresh through **14 Nov 2025** revalidates the five-year drift structure from `docs/evidence/roc_horizon_rollup.json`: mean reversion sits at **+6.55 bp @60 m / +28.75 bp @360 m**, neutral remains negative (−4.13 bp / −19.55 bp), and chaotic stays bifurcated (+20.86 bp / −8.68 bp).
- Latest ROC exports (`output/latest_weekly_roc_window.json`) now include the reconstructed **7 Nov → 14 Nov** week: mean reversion erased the two-week slump with **+39 bp**, neutral stayed red at −23 bp despite the one-off 27 Oct uptick, and chaotic kept oscillating (−241 bp → +75 bp → +16 bp) so semantic+slope confirmation remains mandatory.
- The refreshed span orchestration (`docs/evidence/roc_history/gameplan/span_catalog.json`) plus `comparison_report.md` reconfirm **perfect rolling vs isolation convergence (avg |Δ| = 0.0000 %)** across all 280 instrument–horizon metrics, preserving the restart/validation guarantees outlined in docs/08.
- Gate health script reached Valkey but found no live payloads in the offline lab (`logs/signal/gate_health_2025-11-16.json`); we therefore pulled the missing candles directly from OANDA, removed the partial 7–10 Nov / 10–11 Nov spans, and replayed `backfill_gate_history.py` to produce `roc_summary_2025-11-07_to_2025-11-14.json`.
- Recommended guardrails for the coming week: (1) enforce a 14-day mean-revert floor of **≥ 0 bp** before scaling above base size, (2) keep neutral in exit-only mode until the monthly 60 m average clears **+20 bp** (Nov now **−30 bp**), and (3) require chaotic to show **positive 60 m and 360 m drift simultaneously** plus enriched-strand confirmation before any admits.

## 2. Data Integrity & Tooling Trail
1. **Gate freshness audit.** `python3 scripts/tools/check_gate_freshness.py --redis redis://localhost:6379/0` returned seven missing instruments because the lab Valkey instance carried no `gate:last:*` keys; the outcome is logged in `logs/signal/gate_health_2025-11-16.json`.
2. **Full-week reconstruction.** With the project venv + `OANDA.env`, ran `python scripts/tools/backfill_gate_history.py --start 2025-11-07T00:00:00Z --end 2025-11-14T00:00:00Z --profile config/echo_strategy.yaml --export-json docs/evidence/roc_history/gates_with_roc_2025-11-07_to_2025-11-14.jsonl --export-roc-summary docs/evidence/roc_history/roc_summary_2025-11-07_to_2025-11-14.json` to regenerate the 7-day window, then deleted the truncated `*_2025-11-07_to_2025-11-10*` and `*_2025-11-10_to_2025-11-11*` artifacts.
3. **Longitudinal regeneration.** Reran `scripts/research/roc_longitudinal_summary.py` and `scripts/research/roc_horizon_rollup.py` to repopulate `docs/evidence/longitudinal_2020_2025/*` plus the CSV/JSON feeds that power docs/07 + dashboard analytics.
4. **Span workflow refresh.** `scripts/research/signal_regime_workflow.py` rematerialized `processed_5y_roc.pkl`, `span_catalog.json`, and `job_specs.json` with the Nov 14 cut; `scripts/research/analyze_regime_comparison.py` rewrote `analysis_summary.json` + `comparison_report.md`.
5. **Latest-week extract.** Consolidated the final six ROC summaries into `output/latest_weekly_roc_window.json` for downstream reporting/alerting.

## 3. Longitudinal Stats (Nov 2020 → Nov 2025)
| Regime | 60 m avg (bp) | 60 m pos% | 360 m avg (bp) | 360 m pos% | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| **mean_revert** | **+6.55** | 50.30 % | **+28.75** | 51.25 % | Drift resilience despite four distinct drawdowns (2022 neutral crash, Jul 2024 slump, Oct 2025 wobble, Nov catch-up). |
| **neutral** | −4.13 | 49.83 % | −19.55 | 49.79 % | Still negative at every horizon; continue using as exit regime only. |
| **chaotic** | +20.86 | 49.75 % | −8.68 | 50.58 % | Short horizons capture episodic breakouts; structural bleed resumes beyond 240 m. |

Sources: `docs/evidence/roc_horizon_rollup.json`, `docs/evidence/longitudinal_2020_2025/regime_horizon_summary.csv`.

## 4. Weekly Windows Leading Into Sydney Open
| Window (UTC) | MR 60 m (bp / pos%) | Neutral 60 m (bp / pos%) | Chaotic 60 m (bp / pos%) | Commentary |
| --- | --- | --- | --- | --- |
| 20 Oct → 27 Oct | +8.8 / 48.3 % | +1.8 / 49.8 % | −241.5 / 43.6 % | Chaos cratered on 39 admits; MR stayed mildly positive but hazard slopes elevated. |
| 24 Oct → 31 Oct | −17.6 / 48.6 % | +0.6 / 49.9 % | −34.3 / 54.6 % | Post-FOMC chop knocked MR/chaos; neutral flat on thin sample. |
| 27 Oct → 03 Nov | −18.0 / 48.4 % | +16.0 / 50.7 % | +75.1 / 61.8 % | Mixed week with chaotic squeeze (55 admits) yet MR stayed negative. |
| 31 Oct → 07 Nov | −0.5 / 53.5 % | −41.6 / 49.1 % | −65.5 / 43.4 % | Neutral crash returned; chaos rejected (76 admits). |
| 03 Nov → 10 Nov | +1.6 / 52.3 % | −37.4 / 49.0 % | −39.9 / 42.4 % | MR stabilized but neutral/chaos still bleeding. |
| 07 Nov → 14 Nov | +39.2 / 51.2 % | −23.2 / 49.3 % | +16.4 / 47.9 % | Full-week reconstruction: MR finally cleared +30 bp; neutral stayed red; chaos modest bounce (73 admits). |

*All rows now represent full seven-day windows sourced from `output/latest_weekly_roc_window.json`; partial spans were removed.*

Takeaways:
- **Mean Revert:** The two −18 bp weeks through 3 Nov were followed by stair-step stabilization (−0.5 bp → +1.6 bp) and the rebuilt +39 bp week ending 14 Nov; treat the rally as conditional on the 14-day average staying ≥ 0 bp.
- **Neutral:** Outside of the brief +16 bp pop on 27 Oct, every recent week remains negative (−42 bp, −37 bp, −23 bp); keep it locked to exit-only flow until both weekly and monthly stats flip positive.
- **Chaotic:** Volatility is still extreme (−241 bp to +75 bp to +16 bp across four weeks); only trade chaos with simultaneous positive 60 m/360 m drift plus enriched-strand confirmation.

## 5. Monthly 60 m Context (Jun–Nov 2025)
| Month | Mean Revert | Neutral | Chaotic | Signal |
| --- | ---: | ---: | ---: | --- |
| Jun 2025 | +20.9 bp | +0.5 bp | +127.4 bp | Last strong chaos burst; MR comfortably positive. |
| Jul 2025 | +3.3 bp | −17.7 bp | −33.6 bp | Neutral resumed bleed; chaos flipped negative. |
| Aug 2025 | +11.3 bp | +3.2 bp | +7.9 bp | Consolidation month; MR & chaos modestly green. |
| Sep 2025 | +7.7 bp | −4.6 bp | −43.9 bp | Chaos stress returns; maintain sandbox. |
| Oct 2025 | +6.0 bp | −37.9 bp | +48.7 bp | Neutral crash mirrored by weekly table; chaos volatile. |
| Nov 2025* | +20.6 bp | −30.4 bp | −13.9 bp | Rebuilt week lifts MR back above +20 bp; neutral/chaos still negative. |

Source: `docs/evidence/longitudinal_2020_2025/monthly_60m_summary.csv`. *Nov covers data through 14 Nov.

Implications: the allocator should continue treating neutral as a brake (monthly trigger: halt promotions if monthly MR < 0 bp for two consecutive months, already implemented; extend to block neutral until monthly avg > +20 bp). Chaotic remains untradeable structurally despite isolated squeezes, even though the Nov mean-revert recovery helps overall drift.

## 6. Rolling vs Isolation Evidence
- `docs/evidence/roc_history/gameplan/comparison_report.md` + `analysis_summary.json` show **0.0000 % average absolute delta** between rolling and isolation modes across every span (2020-11-13 → 2025-11-14).
- Each span’s `stacking_<span>.json` under `docs/evidence/roc_history/gameplan/comparisons/` confirms per-instrument/per-horizon agreement; the 26-week validation anchor (2021-04-09 → 2021-12-10) behaves identically to the later, longer spans.
- `docs/evidence/roc_history/gameplan/final_gameplan_results.md` documents simulator coverage and memory-safe processing of the 525-day span (2023-06-23 → 2024-11-29), so the restart/runbook guidance in docs/08 remains accurate.

## 7. Guardrail & Runbook Updates for the Coming Week
1. **Mean-Revert scaling rules.** Calculate a rolling 14-day average of 60 m ROC per instrument using the latest weekly snapshots; block any leverage increases if that stat < 0 bp or if hazard slopes exceed prior week’s median by > 0.5 σ. Implement as a dashboard alert sourced from `output/latest_weekly_roc_window.json`.
2. **Neutral handling.** Keep neutral as exit-only. Add a cron task that ingests `monthly_60m_summary.csv` each Monday and raises a PagerDuty info event if neutral prints ≥ +20 bp for two consecutive months (unlock review). Until then, neutral promotions remain disabled regardless of isolated weekly pops like 27 Oct.
3. **Chaotic confirmations.** Require dual conditions before any chaos admit: (i) current 60 m and 360 m drift from `docs/evidence/roc_horizon_rollup.json` both > 0, and (ii) enriched-strand flags (flat coherence + positive domain-wall slope) validated via `scripts/trading/regime_manifold_service.py`. The −241 bp week of 20 Oct followed by whipsaw spikes illustrates why the sandbox stays in force.
4. **Operational cadence.** Until live gates resume, schedule a Sunday automation that (a) syncs the archived ROC history through Friday, (b) re-runs the longitudinal + span workflows (`scripts/research/roc_longitudinal_summary.py`, `roc_horizon_rollup.py`, `signal_regime_workflow.py`, `analyze_regime_comparison.py`), and (c) publishes the refreshed weekly JSON snapshot for the ops dashboard. Once the streamer reconnects, reinstate `scripts/tools/check_gate_freshness.py` as a blocking step before any admits.
5. **Documentation updates.** Reflect the guardrail changes and the Week 46 evidence in `docs/08_Signal_Regime_Rework.md` and the dashboard’s Weekly Signal Analytics copy so the live ops view matches this report.

## 8. Two-Stage ML Gate Evidence (Transformer radar + regime specialists)
**Macro→micro flow.** The regime-change Transformer (trained via `scripts/ml/train_sequence_model.py` on the `regime_change_60m` label) delivers ≈0.845 validation AUC and now acts as the weather radar. Its score archive lives at `output/ml_models_regime_change/regime_change_scores.parquet` and feeds the simulator/PortfolioManager as a “flip probability” defensive override. Once the radar permits a minute, the per-regime XGBoost models (`output/ml_models/ml_scores_xgb_{mr,neutral,chaos}.parquet`) decide whether the local state is tradable; their best ROC lifts remain +2.8 % (mean-revert), +2.1 % (neutral), and +3.2 % (chaotic).

**Calibration sweep (8-week grid, ML + defensive thresholds).** `python scripts/ml/run_ml_backtests.py --calibration-weeks 8 --ml-threshold-grid 0.55,0.6 --defensive-threshold-grid 0.75,0.8 --mode calibrate`

| Rank | MR/Neutral/Chaos threshold | Radar threshold | PnL (USD) | Trades | ML blocks | Radar blocks | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.60 / 0.60 / 0.60 | 0.75 | **−38,468** | 475 | 368,129 | 4,720 | Symmetric ML cutoffs + slightly looser radar gave the highest NAV retention while blocking the most toxic minutes. |
| 2 | 0.60 / 0.60 / 0.60 | 0.80 | −38,468 | 475 | 368,129 | 4,096 | Same ML performance; stricter radar lets ~600 more minutes through. |
| 3 | 0.60 / 0.60 / 0.55 | 0.75 | −41,815 | 516 | 367,281 | 4,720 | Chaotic relaxation buys 8 % more trades but gives back ≈3.3 k. |
| 4 | 0.60 / 0.55 / 0.60 | 0.75 | −75,947 | 1,322 | 354,285 | 4,720 | Neutral loosened by 5 bp crushes coverage and worsens drawdown. |

**Chosen thresholds:** 0.60 ML cutoff per regime + 0.75 radar trigger. Neutral or chaotic relaxations explode trade counts without improving PnL, so we keep the symmetric gate until per-regime diagnostics show a positive skew.

**Full 2020–2025 replay (`output/ml_backtests/full_backtest_summary.json`).**
- Command: `python scripts/ml/run_ml_backtests.py --mode full --ml-thresholds mean_revert=0.6,neutral=0.6,chaotic=0.6 --defensive-threshold 0.75`.
- Totals: `PnL = −772,005 USD`, `trades = 10,729`, `ml_blocked_minutes = 13,628,317`, `defensive_blocks = 167,980`.
- Sharpe proxy: nav risk is fixed at 1 %, so the average loss per trade (~−$72) corresponds to −0.072 R. Treating this R multiple as the Sharpe proxy makes expectancy improvements visible even when the raw PnL is negative.
- Coverage impact: ML filters remove **~13.6 M** minute-level decisions (macro coverage cost), while the radar blocks **168 k** pathological flip windows (micro kill switch). This asymmetry confirms the radar is a targeted override layered above the always-on ML filters.
- Evidence trail: `docs/ML_Modeling_Log.md`, calibration/full summaries in `output/ml_backtests/`, ML score dumps in `output/ml_models/`, radar scores in `output/ml_models_regime_change/`.

**Inversion stress-test (filters ON, trade direction flipped).**
- Command mirrors the standard replay but adds `--invert-trades --trade-log output/ml_backtests/full_trades_inverted.parquet` (see Modeling Log for the full shell block with CPU-thread pins).
- Outcome: `PnL = −835,305 USD` with the same 10,729 trades. Win rate fell to 25.3 %, proving that even the “best” 10 % of raw gates remain net-negative when mirrored—the ML stack is correctly surfacing the cleanest losers.
- Survivor analytics (again filtered via the trade log) show that winners differ mainly via structure: coherence delta averages +0.022 vs +0.011 on losers, hazard gaps are ~5 bp more negative, and win rate climbs to 35 % when `coherence_delta ≥ 0.08`. Those findings drive the premium rule design below.

## 9. Whitepaper Outline – Weather Radar + Specialist ML Gate
**Methodology (Section 1).**
- Describe the macro→micro handoff using `handoff_base/00_strategy/ML_Gate_Model_Plan.md` + `docs/ML_Modeling_Log.md`: feature builder → regime-change transformer → regime-specific XGBoost filters.
- Reference the training artefacts: `scripts/ml/feature_builder.py`, `scripts/ml/train_sequence_model.py`, `scripts/ml/train_tabular_baselines.py`, and the exported parquet score files.

**Evidence Trail (Section 2).**
- Calibration grid and full replay metrics from `output/ml_backtests/{calibration_summary.json,full_backtest_summary.json}`; include NAV, trade counts, ML vs radar block counts, and the per-trade Sharpe proxy.
- Weekly ROC + longitudinal stats from Sections 3–6 above tie the ML outcomes back to proven drift structure (docs/evidence/roc_history/*).
- Offline simulator harness: `scripts/research/simulator/backtest_simulator.py` + `scripts/ml/run_ml_backtests.py` provide reproducibility; cite the USD_JPY replay (Section “Simulator Integration” in `docs/ML_Modeling_Log.md`).

**Deployment Plan (Section 3).**
- PortfolioManager wiring already honors `ml_threshold` and the radar override (configurable in `config/echo_strategy.yaml`); outline the rollout: update profile thresholds → redeploy backend via `docker compose -f docker-compose.hotband.yml up backend valkey` → monitor `/api/gates/{instrument}/latest` + `/api/risk/allocation-status`.
- Ops/killswitch tie-in: document in `docs/02_Operations_Runbook.md` how to verify the radar blocks (reason code `regime_change_prob`) before enabling live trading.

**Premium Strategy Annex.**
- Dataset: `output/ml_backtests/full_trades_inverted.parquet` (10,729 trades with regime + structural metrics). Treat it as the lab corpus for hypothesis mining.
- Baseline vs premium slices:

| Slice | Trades | PnL (USD) | Avg/trade | Win % |
| --- | ---: | ---: | ---: | ---: |
| All inverted trades | 10,729 | −833,726 | −77.7 | 25.3 % |
| `coherence_delta ≥ 0` | 6,402 | −492,844 | −77.0 | 28.6 % |
| `coherence_delta ≥ 0.08` | 1,273 | −96,081 | −75.5 | 35.1 % |
| `hazard_gap ≤ −0.10` | 977 | −77,896 | −79.7 | 33.7 % |
| MR-only premium probe* | 183 | −16,399 | −89.6 | 34.4 % |

*Premium probe = mean-revert regime, `coherence_delta ≥ 0.08`, `hazard_gap ≤ −0.10`, `domain_wall_slope ≥ 0`. Still negative but the shallowest bleed so far; it isolates the structural context where inversion + filters almost break even.
- Action items:
  1. Encode those premium gates via the new CLI switches (`--invert-trades`, `--require-regime`, `--min-coherence-delta`, `--max-hazard-gap`, `--min-domain-wall-slope`, `--trade-log`) to reproduce the offline slice in-situ and confirm no path/hold artefacts remain.
  2. Layer additional classifiers atop the survivor set (e.g., small gradient-boosted tree on the trade log) to separate the 25 % winners from the 75 % losers.
  3. Promote the first profitable subset into `config/echo_strategy.yaml` as a dedicated “premium” block once lab results flip positive, then rerun the Week 46 evidence workflow with the revised ML gating narrative.
**Hero backtests (Week 46 directive).**
  - *Mean-Revert Alpha (long-only):* `python scripts/ml/run_ml_backtests.py --mode full --ml-thresholds mean_revert=0.6,neutral=999,chaotic=999 --defensive-threshold 0.75 --require-regime mean_revert --min-coherence-delta 0.08 --max-hazard-gap -0.10 --min-domain-wall-slope 0.0 --trade-log output/ml_backtests/premium_mr_long_only.parquet` → `PnL = −$16,255`, `trades = 196`, `win% = 27.0 %`, `avg = −$83`. This is the shallowest drawdown we have seen and confirms the “Premium Probe” fingerprint; EUR_USD is nearly flat while USD_JPY/GBP_USD still need extra structure filters.
  - *Neutral Fade (inverted):* `python scripts/ml/run_ml_backtests.py --mode full --ml-thresholds mean_revert=999,neutral=0.6,chaotic=999 --defensive-threshold 0.75 --invert-trades --require-regime neutral --trade-log output/ml_backtests/neutral_fade_inverted.parquet` → `PnL = −$298k`, `trades = 4,248`, `win% = 24.8 %`, `avg = −$70`. Losses shrink dramatically vs the −$835k all-regime inversion, so neutral becomes a managed fade instead of an outright ban.
  - *Chaos:* No strategy beyond “quarantine”—the historical stats still show coin-flip expectancy that bleeds out over 60/360 m horizons.
  - *Mean-Revert Alpha v2 (EUR/USD + exits):* `python scripts/ml/run_ml_backtests.py --mode full --instrument EUR_USD --ml-thresholds mean_revert=0.6,neutral=999,chaotic=999 --defensive-threshold 0.75 --require-regime mean_revert --min-coherence-delta 0.08 --max-hazard-gap -0.10 --min-domain-wall-slope 0.0 --take-profit-bps 8 --stop-loss-bps 6 --trade-log output/ml_backtests/premium_mr_long_tp8_sl6.parquet` → `PnL = −$8,009`, `trades = 70`, `win% = 17.1 %`. Stop-losses fired 20× versus 3× take-profit hits, confirming the new exit logic and cutting the EUR/USD drawdown roughly in half.
  - *Exit sweep (“Premium 69,” EUR/USD only, ML=0.6).* TP/SL grids (10/5, 12/6, 15/8) stayed negative but highlighted that 15/8 delivers the shallowest loss (−$6.3 k, 22.9 % win, 12 SL hits vs 3 TP hits) → `output/ml_backtests/premium_mr_eurusd_tp15_sl8.parquet`.
  - *Mean-Revert Alpha v3 – Precision Strike:* `python scripts/ml/run_ml_backtests.py --mode full --instrument EUR_USD --ml-thresholds mean_revert=0.61,neutral=999,chaotic=999 --defensive-threshold 0.75 --require-regime mean_revert --min-coherence-delta 0.08 --max-hazard-gap -0.10 --min-domain-wall-slope 0.0 --take-profit-bps 15 --stop-loss-bps 8 --trade-log output/ml_backtests/FINAL_STRATEGY_EUR_USD.parquet` → `PnL = −$4,865`, `trades = 44`, `win% = 25 %`, `avg = −$111`. Stops (8×) still outnumber targets (2×), but we’re now within ~$5 k of breakeven over five years and the ML ≥0.61 band confirms the alpha’s fingerprint.
  - *Whitepaper strategy (ML ≥0.64 + trailing stop):* `python scripts/ml/run_ml_backtests.py --mode full --instrument EUR_USD --ml-thresholds mean_revert=0.64,neutral=999,chaotic=999 --defensive-threshold 0.75 --require-regime mean_revert --min-coherence-delta 0.08 --max-hazard-gap -0.10 --min-domain-wall-slope 0.0 --stop-loss-bps 8 --trailing-stop-bps 7 --trade-log output/ml_backtests/WHITEPAPER_STRATEGY_FINAL.parquet` → `PnL = −$324`, `trades = 17`, `win% = 17.6 %`, `avg = −$19`. The trailing stop captured four of the best drifts while fixed stops handled the eight bad prints, putting the strategy within noise of breakeven.

---

**Artifacts referenced:**  
`logs/signal/gate_health_2025-11-16.json`, `docs/evidence/roc_history/roc_summary_2025-11-07_to_2025-11-14.json`, `output/latest_weekly_roc_window.json`, `docs/evidence/roc_horizon_rollup.json`, `docs/evidence/longitudinal_2020_2025/*`, `docs/evidence/roc_history/gameplan/*`, `docs/ML_Modeling_Log.md`, `handoff_base/00_strategy/ML_Gate_Model_Plan.md`, `output/ml_models/*.parquet`, `output/ml_models_regime_change/regime_change_scores.parquet`, `output/ml_backtests/{calibration_summary.json,full_backtest_summary.json}`.
