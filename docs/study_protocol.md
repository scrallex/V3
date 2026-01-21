# Study Protocol (ABA Design)

This protocol validates gains from the telemetry optimizer across mining rigs. It mirrors
the existing SEP ABA methodology and captures the metrics required for a publishable case
study.

## Design Overview

- **Baseline (A1)**: 24–48 h recording with collectors running but no advisory or control.
- **Treatment (B)**: 72 h with advisory enabled (auto-tune off initially). Record every
  recommendation and operator action.
- **Washout (A2)**: 24 h back to passive collection to ensure no lingering effects.
- **Optional**: Repeat with guarded auto-tune once advisory lifts efficiency reliably.

Stagger rigs when possible to avoid correlated pool variance.

## Primary Metrics

1. **Efficiency gain**  
   \[
   \Delta \mathrm{GH/J} = \frac{\mathrm{GH/J}_{B} - \mathrm{GH/J}_{A}}{\mathrm{GH/J}_{A}}
   \]
2. **Reject delta** (percentage point change in reject rate).
3. **Healthy band uptime** – fraction of time `sep_score ≥ 70` and `reject_rate ≤ 1 %`.
4. **Rupture MTBF** – mean time between macro ruptures.

Secondary metrics include thermal headroom distribution, latency budget trends, and
operator load (number of advisory actions executed).

## Data Collection

- NDJSON logs (`logs/telemetry/*.ndjson`) provide raw snapshots and scores per rig.
- Advisory log (`logs/telemetry/*_advice.ndjson`) captures recommendations.
- Export daily Parquet summaries via `scripts/telemetry/export_day.py` (future helper) for
  notebook analysis.
- Record environmental changes (ambient temp, PSU swaps) in `reports/aba_annotations.md`.

## Analysis Workflow

1. Use a notebook (`results/report.ipynb`) to load baseline/treatment windows.
2. Compute paired metrics per rig; apply Wilcoxon signed-rank tests when normality is not
   guaranteed.
3. Generate plots:
   - GH/J vs time (before/after) with action markers.
   - SEP score and rupture flags.
   - Reject rate histograms.
4. Summarise each rig in a one-page Result Card (median, IQR, confidence intervals).

## Acceptance Criteria

- Efficiency gains statistically significant (p < 0.05) and ≥ 2 %.
- Reject rate does not worsen (≤ 0.5 pp increase).
- No increase in macro ruptures; ideally MTBF improves.
- Operator acceptance: advisory volume manageable (< 12 actions per day per rig).

Once criteria are met, proceed to guarded auto-tune trials, following the safety rails in
`docs/control_policy.md`.
