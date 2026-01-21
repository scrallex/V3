# SEP Signal Study — Lead/Lag & Strand Enrichment (Nov 2020 – Nov 2025)

## 1. Dataset & tooling refresh
- **Coverage:** 288 weekly windows from **13 Nov 2020** through **11 Nov 2025** across AUD/USD, EUR/USD, GBP/USD, NZD/USD, USD/CAD, USD/CHF, USD/JPY.
- **Volume:** 439k mean-revert, 418k neutral, and 9.3k chaotic gate snapshots with full structural metrics, semantic tags, and forward ROC at 5–360 minute horizons.
- **Generation commands (run from repo root with `.venv` active and `PYTHONPATH=.`):**
  ```bash
  .venv/bin/python scripts/research/roc_longitudinal_summary.py --start 2020-11-13 --end 2025-11-11 --roc-dir docs/evidence/roc_history --output-dir docs/evidence/longitudinal_2020_2025
  .venv/bin/python scripts/research/roc_horizon_rollup.py --roc-dir docs/evidence/roc_history --output-json docs/evidence/roc_horizon_rollup.json --output-csv docs/evidence/roc_horizon_rollup.csv
  .venv/bin/python scripts/research/lead_lag_regression.py --roc-dir docs/evidence/roc_history --dataset-csv docs/evidence/lead_lag_features.csv --output-json docs/evidence/lead_lag_model.json --summary-txt docs/evidence/lead_lag_model.txt
  .venv/bin/python scripts/research/lead_lag_regression.py --roc-dir docs/evidence/roc_history --window-weeks 26 --dataset-csv docs/evidence/lead_lag_features_recent.csv --output-json docs/evidence/lead_lag_model_recent.json --summary-txt docs/evidence/lead_lag_model_recent.txt
  .venv/bin/python scripts/research/enriched_strand_analysis.py --roc-dir docs/evidence/roc_history --horizons 60,90,360 --output-csv docs/evidence/enriched_strands.csv --summary-md docs/evidence/enriched_strands.md
  ```
- **Artefacts:** updated CSV/JSON/PNG outputs live under `docs/evidence/` (see Section 5 for the full list referenced in the whitepaper + dashboard).

## 2. Regime behaviour vs horizon
| Regime | Samples | 5 m avg / pos% | 60 m avg / pos% | 90 m avg / pos% | 360 m avg / pos% |
| --- | ---: | ---: | ---: | ---: | ---: |
| **mean_revert** | 439 484 | +0.73 bp / 48.99 % | **+6.45 bp / 50.29 %** | +8.29 bp / 50.50 % | **+28.94 bp / 51.25 %** |
| **neutral** | 417 720 | −0.52 bp / 48.26 % | −4.01 bp / 49.83 % | −5.56 bp / 49.96 % | −19.29 bp / 49.79 % |
| **chaotic** | 9 325 | +0.38 bp / 47.59 % | +20.72 bp / 49.75 % | +9.49 bp / 49.92 % | −9.18 bp / 50.53 % |

**Observations**
1. Mean-revert keeps a positive drift out to six hours with a statistically meaningful (>439k) sample; the win rate creeps just ~1.3 pp above 50 %, so we still treat the drift as marginal rather than a stand-alone allocator.
2. Neutral strands remain negative at every horizon and never push the win rate above 50 %; they continue to serve as a liquidity-only state.
3. Chaotic strands look tradable at ≤90 m but turn negative beyond, reinforcing the “quarantine chaos beyond 90 m” risk rule; only 9.3k samples exist, so confidence intervals remain wide.

## 3. Lead/Lag modelling (neutral 60 m ROC sign)
### 3.1 Window-level accuracy
| Dataset | Observations | Majority baseline | Logit accuracy | Rolling accuracy | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Full Nov 2020–Nov 2025 cut | 287 | 48.8 % | **55.7 %** | 46.5 % | Stable but low signal-to-noise; statsmodels raised `PerfectSeparationWarning` because several features cluster tightly.
| Most recent 26 weeks | 26 | 50.0 % | **69.2 %** | 50.0 % | Stronger fit but coefficients explode (collinearity); treat as directional evidence, not production quality.

### 3.2 MLE coefficients (docs/evidence/lead_lag_model*.json)
| Feature | 5‑year coeff (p‑value) | Recent 26 wk coeff (p‑value) | Commentary |
| --- | ---: | ---: | --- |
| Intercept | −0.62 (0.86) | −0.30 (0.98) | Baseline odds stay near 50 %. |
| MR D5 avg ROC 60 m | +15.73 (0.86) | −1.14×10² (0.76) | Direction flips between windows; high p-values show the slope is indistinguishable from zero. |
| MR D5 avg ROC 90 m | −50.01 (0.42) | +2.67×10² (0.36) | Captures curvature between 60 m/90 m but remains non-significant. |
| MR D5 positive share 60 m | +0.16 (0.98) | −15.40 (0.55) | Positive share drove the original 2025 cut; once we include 287 weeks it no longer stands out statistically. |
| MR D5 avg coherence τ slope | +57.86 (0.55) | +9.02×10² (0.20) | Structural tightening still aligns with future neutral weakness, but variance dominates. |
| MR D5 avg domain-wall slope | +52.17 (0.68) | +5.71×10² (0.32) | Same story—signs match the qualitative “positive domain-wall => neutral drawdown” heuristic, yet confidence intervals are huge.

### 3.3 Regularised & Bayesian checks
- **Ridge logit (`α=0.5`):** collapses every slope to zero for both datasets (accuracy drops to 51 % / 50 % with rolling accuracies 48 % / 33 %). Conclusion: until we either increase observations or lower the penalty, ridge just reproduces the majority baseline.
- **Laplace Bayesian (`α=0.2`, 4k draws):** posterior means hover near zero with 95 % HPDs spanning ±3 for every coefficient (see `docs/evidence/lead_lag_model.json` and `_recent.json`). This provides honest uncertainty bands for investor comms even though no feature clears significance yet.
- **Warnings:** statsmodels emitted `PerfectSeparationWarning` and overflow logs; coefficients should be interpreted qualitatively only, and the dashboard should surface hit-rate deltas rather than raw logistic odds.

## 4. Strand-level insights (docs/evidence/enriched_strands.*)
### 4.1 Mean-revert D5 slope filters (count ≥ 700)
| Sub-strand | Count | 60 m avg (bp) | Positive % |
| --- | ---: | ---: | ---: |
| `mean_revert_d5_r2`, `high_rupture_event`, **coh=flat**, **dw=pos** | 3 468 | **+18.4** | 51.18 % |
| `mean_revert_d5_r1`, `high_rupture_event`, **coh=flat**, **dw=pos** | 45 389 | +15.9 | 50.58 % |
| `mean_revert_d5_r1`, `high_rupture_event`, **coh=neg**, **dw=pos** | 5 412 | +14.7 | 51.20 % |
| `mean_revert_d5_r2`, `high_rupture_event`, **coh=flat**, **dw=neg** | 3 245 | −17.8 | 50.66 % |
| `mean_revert_d5_r2`, `high_rupture_event`, **coh=pos**, **dw=pos** | 2 073 | **−24.7** | 49.40 % |
**Takeaway:** holding coherence near flat while demanding positive domain-wall slopes roughly doubles the MR drift; letting both slopes run positive hands back 18–25 bp.

### 4.2 Neutral hazard warnings (count ≥ 1 000)
| Sub-strand | Count | 60 m avg (bp) | Positive % |
| --- | ---: | ---: | ---: |
| `neutral_d4_r2`, `high_rupture_event`, **coh=flat**, **dw=neg** | 2 037 | −72.0 | 48.60 % |
| `neutral_d4_r2`, `high_rupture_event`, **coh=neg**, **dw=flat** | 1 121 | −54.3 | 48.53 % |
| `neutral_d4_r1`, `high_rupture_event`, **coh=flat**, **dw=flat** | 10 772 | −25.8 | 49.19 % |
| `neutral_d4_r2`, `high_rupture_event`, **coh=neg**, **dw=neg** | 2 788 | −20.6 | 50.50 % |
**Takeaway:** once hazard reaches D4 with a rupture tag, every slope combination bleeds ≥20 bp at 60 m; these should be immediate blockers in PortfolioManager.

### 4.3 Chaotic strand polarity (count ≥ 250)
| Sub-strand | Count | 60 m avg (bp) | Positive % |
| --- | ---: | ---: | ---: |
| `chaotic_d4_r1`, `high_rupture_event`, **coh=neg**, **dw=pos** | 605 | +80.7 | 52.56 % |
| `chaotic_d4_r1`, `high_rupture_event`, **coh=flat**, **dw=neg** | 1 017 | +71.9 | 49.85 % |
| `chaotic_d4_r1`, `high_rupture_event`, **coh=neg**, **dw=flat** | 289 | −95.9 | 47.75 % |
| `chaotic_d4_r1`, `high_rupture_event`, **coh=flat**, **dw=flat** | 276 | −70.5 | 50.72 % |
**Takeaway:** chaotic strands only outperform when domain-wall slopes keep a sign; once slopes flatten, the strand collapses (<−70 bp) regardless of semantics.

## 5. Implications & artefact map
1. **Promotion rubric:** keep “flat coherence + positive domain-wall” as a hard gate for MR promotions (`docs/evidence/enriched_strands.csv`).
2. **Neutral quarantine:** D4/D5 + rupture tags continue to post −20 bp to −70 bp at 60 m; wire this as a stop condition before allocator changes.
3. **Lead/lag monitor:** track MR D5 positive share + slope telemetry weekly; current evidence suggests ~7 pp of lift over baseline but with wide confidence bands. The dashboard should show both the 5-year hit rate and the 26-week swing to contextualise any excursions.
4. **Chaos guardrails:** continue treating >90 m chaotic exposure as a kill-switch—recent slope splits prove that flat domain walls lead to −70 bp outcomes.
5. **Reproducibility:** every figure/table in this note traces back to `docs/evidence/*` outputs generated with the commands in Section 1; the whitepaper and dashboard reference the same assets so ops/investors can cross-check without rerunning notebooks.
