# Signal Analytics Playbook

Signals are the only input the lean trading stack trusts. This playbook explains what a signal contains, how to measure signal health, and how to interpret the analytics when deciding whether the portfolio manager should trade.

## 1. Signal Anatomy

Each gate payload stored in Valkey (`gate:index:{instrument}`, `gate:last:{instrument}`) contains:

- **Top-level verdict** – `admit` flag, `lambda`/`hazard` (rupture proxy), `repetitions`.
- **Structural metrics** – `coherence`, `stability`, `entropy`, slope metrics, Reynolds ratio, temporal half-life, spatial correlation length, pinned alignment.
- **Regime context** – canonical regime label + confidence, semantic tags, session window outcome.
- **Rejection reasons** – array of guardrails hit (`hazard_exceeds_adaptive_threshold`, `session_closed`, `regime_filtered`, etc.).

The portfolio manager mirrors these checks before sizing trades, so the signal stream is the earliest warning system when the execution loop should stand down.

## 2. Analytics Workflow

1. **Confirm gate freshness**
   ```bash
   python scripts/tools/check_gate_freshness.py --redis ${VALKEY_URL} --threshold 180
   ```
   Only move forward if the exit code is `0`.

2. **Run the signal analytics helper**
   ```bash
   python scripts/tools/signal_analytics.py \
     --redis ${VALKEY_URL} \
     --lookback-minutes 240 \
     --top-count 5 \
     --json > output/signal_analytics/latest.json
   ```
   Drop `--json` to get a human-readable summary. Use `--instruments EUR_USD,USD_JPY` to focus on a subset, or `--max-signals` to cap the sample size per pair.

3. **Archive / alert**
   - Store the JSON artefact in `output/signal_analytics/`.
   - Feed it to the Ops dashboard or Slack for daily reporting.
4. **Backfill longer windows when needed**
   ```bash
   python scripts/tools/backfill_gate_history.py \
     --start 2025-11-03T00:00:00Z \
     --end   2025-11-10T00:00:00Z \
     --redis ${VALKEY_URL} \
     --flush \
     --export-json docs/evidence/gates_2025-11-03_to_2025-11-10.jsonl
   ```
   This script fetches historical OANDA candles for every configured instrument, rebuilds the manifold windows, writes each gate payload into Valkey (`gate:index:{instrument}` + `gate:last:{instrument}`), and emits a JSONL evidence file under `docs/evidence/`. Run it any time you need more than the live 3‑hour ring buffer.

## 3. Experimental Evidence (10 Nov 2025 @ 19:51 UTC)

Command:

```bash
docker compose -f docker-compose.hotband.yml exec -T backend \
  python /app/scripts/tools/signal_analytics.py \
    --lookback-minutes 240 \
    --top-count 5
```

Findings:

| Instrument | Signals | Admit % | Avg Hazard | Top Rejection Drivers |
|------------|---------|---------|------------|-----------------------|
| AUD_USD    | 250     | 91.2%   | 0.488      | hazard_exceeds adaptive threshold (17), hazard_fallback_requested (14) |
| EUR_USD    | 250     | 90.4%   | 0.503      | hazard_exceeds adaptive threshold (18), hazard_fallback_requested (14) |
| GBP_USD    | 250     | 88.0%   | 0.502      | hazard_exceeds adaptive threshold (30) |
| NZD_USD    | 250     | 91.2%   | 0.517      | hazard_exceeds adaptive threshold (22), regime_filtered (19) |
| USD_CAD    | 250     | 92.4%   | 0.516      | hazard_exceeds adaptive threshold (19) |
| USD_CHF    | 250     | 92.0%   | 0.478      | hazard_exceeds adaptive threshold (15), hazard_fallback_requested (14) |
| USD_JPY    | 250     | 91.6%   | 0.504      | hazard_exceeds adaptive threshold (15), hazard_fallback_requested (14) |

Interpretation:

- **Throughput:** 1,750 signals in four hours, 91 % admitted. The gate is live and feeding the portfolio manager – lack of fills would signal downstream issues (kill switch, read-only mode), not missing signals.
- **Structural quality:** Average coherence hovered near zero while hazard sat at ~0.5. Signals are barely acceptable; keep risk caps tight until coherence improves.
- **Hazard guardrails:** GBP/USD and NZD/USD triggered `hazard_exceeds_adaptive_threshold` the most. Revisit the hazard percentile in `regime_manifold_service.py` or investigate manifold stability for those pairs.
- **Session policy:** After this run we removed the duplicate session check from the manifold service; session windows are now enforced only inside the portfolio manager so the signal payload stays pure.
- **Regime filters:** NZD/USD’s `regime_filtered` hits imply its semantic/regime filter disagrees with the live tape. Either relax the filter in `config/echo_strategy.yaml` or retrain the regime classifier.
- **Best setups:** USD_CAD and GBP_USD produced the highest highlight scores (~0.58) with hazard ≈0.51, coherence ≈0.01. Treat them as priority candidates once hazard drifts lower.

Store this table (and future runs) alongside market context so strategy reviews have concrete evidence.

## 4. Using Analytics to Drive Decisions

- **Enable trading only when** admit rates stay high *and* structural averages (coherence, stability) trend upward while hazard trends down.
- **Investigate hazard spikes** by checking `hazard_fallback_requested` counts; if frequent, lower leverage or disable the instrument until manifolds stabilise.
- **Adjust sessions/filters** when `session_closed` or `regime_filtered` dominate rejection reasons but the underlying tape looks compelling.
- **Automate monitoring** by cronning:
  ```
  */15 * * * * VALKEY_URL="redis://valkey:6379/0" \
    python /sep/scripts/tools/signal_analytics.py --lookback-minutes 180 --json \
    > /sep/output/signal_analytics/latest.json
  ```

Armed with these artefacts, every trading enablement decision can cite real metrics instead of intuition.

## 5. Bundle evidence

MB003 (mean-revert promote), NB001 (neutral fade), and CB002 (chaotic quarantine) only ship when their post-cost expectancy stays positive. Refresh their artefacts once you finish the weekly signal study:

```bash
make bundle-study
```

This target:

1. Rebuilds the bundle activation tape (`output/strand_bundles/bundle_activation_tape.jsonl`) from the latest gate exports.
2. Simulates trades per bundle (`output/strand_bundles/bundle_trades.csv`) so we can audit win-rate dispersion by instrument.
3. Emits `docs/evidence/bundle_outcomes.json`, which the backend serves via `/api/evidence/bundle-outcomes` and the dashboard renders as the Bundle Readiness panel.

Keep the kill switch engaged until MB003 and NB001 show positive bp after costs and CB002 is idle. Treat any `bundle_block:*` entry in `/api/metrics/gates` as a soft stop even if other signals look attractive.

## 6. Forward return study (30 min horizon)

Use `scripts/tools/signal_outcome_study.py` to correlate gate metadata with actual price moves:

```bash
docker compose -f docker-compose.hotband.yml exec -T backend \
  python /app/scripts/tools/signal_outcome_study.py \
    --redis redis://valkey:6379/0 \
    --lookback-minutes 180 \
    --horizons "30" \
    --cost-model none
```

**Window analysed:** 10 Nov 2025 16:10–19:50 UTC (latest ~700 signals per pair, forward horizon 30 minutes).

| Instrument | Samples | Avg Return (30 m) | Median | % Positive | Notes |
|------------|---------|-------------------|--------|------------|-------|
| AUD_USD    | 702     | +0.017 %          | +0.016 % | 87.6 % | Low hazard (<0.2) dominates and aligns with mild updrift. |
| EUR_USD    | 703     | +0.012 %          | +0.013 % | 93.7 % | Even hazard-fallback rejects moved higher → guardrails may be too conservative. |
| GBP_USD    | 703     | +0.011 %          | +0.014 % | 87.5 % | Regime filters clipped profitable signals—review semantic filters. |
| NZD_USD    | 698     | +0.004 %          | 0.000 % | 48.1 % | Essentially noise; hazard buckets flat. Pause until manifold improves. |
| USD_CAD    | 702     | −0.008 %          | −0.009 % | 25.2 % | Signals skew negative—either fade or disable this pair. |
| USD_CHF    | 702     | −0.004 %          | −0.007 % | 31.3 % | Down drift despite low hazard → structure logic likely wrong. |
| USD_JPY    | 702     | +0.013 %          | +0.008 % | 87.5 % | Consistently positive; top candidate once risk caps loosen. |

Takeaways:

- Hazard buckets show usable separation: `<0.2` hazard carries the positive expectancy, validating the manifold’s hazard scoring.
- Guardrails (`hazard_exceeds_adaptive_threshold`, `regime_filtered`) are clipping opportunities on EUR/USD and GBP/USD—recalibrate per instrument.
- USD/CAD and USD/CHF signals underperform; keep them sidelined (or trade the opposite) until outcome studies flip positive.

Re-run this study whenever the manifold or risk policy changes so every trading decision references hard evidence, not vibes.

## 7. Weekly multi-horizon evidence (3–10 Nov 2025)

Workflow executed (UTC):

```bash
# 1) Rebuild one week of gates (seven FX pairs).
docker compose -f docker-compose.hotband.yml exec -T backend \
  python /app/scripts/tools/backfill_gate_history.py \
    --start "2025-11-03T00:00:00Z" \
    --end   "2025-11-10T00:00:00Z" \
    --redis redis://valkey:6379/0 \
    --flush \
    --export-json /app/logs/backend/gates_2025-11-03_to_2025-11-10.jsonl

# 2) Analyse the backfilled gates across five forward horizons.
docker compose -f docker-compose.hotband.yml exec -T backend \
  python /app/scripts/tools/signal_outcome_study.py \
    --redis redis://valkey:6379/0 \
    --start "2025-11-03T00:00:00Z" \
    --end   "2025-11-10T00:00:00Z" \
    --horizons "5,15,30,60,240" \
    --export-json /app/logs/backend/outcome_weekly_2025-11-03_to_2025-11-10.json
```

Artifacts: copied to `docs/evidence/gates_2025-11-03_to_2025-11-10.jsonl`, the raw outcome summary `docs/evidence/outcome_weekly_2025-11-03_to_2025-11-10.json`, and the cost-adjusted version `docs/evidence/outcome_weekly_costs.json` (uses median-spread + 0.5×slippage + 0.1 pip commission per trade). Each JSON now includes AUROC, hazard-decile lift tables, t-stats with Holm–Bonferroni adjustments, and per-sample cost accounting, so dashboards can pull the exact metrics without recomputing them.

To run guardrail A/B comparisons, regenerate two evidence files (baseline + variant hazard percentile) and diff them with:

```bash
python scripts/tools/compare_guardrail_studies.py \
  docs/evidence/outcome_weekly_costs.json \
  docs/evidence/outcome_weekly_costs_relaxed.json
```

Headline findings (returns net of mid-price, before explicit cost modelling):

- **EUR_USD** – Expectation rises with horizon: +0.35 bp (60 m) and +1.11 bp (240 m) with ≥57 % positive moves. Guardrails are clipping profitable tapes; hazard percentile can likely be raised for this pair.
- **GBP_USD** – Flat at intraday horizons but +0.60 bp at 240 m with 54 % positive outcomes. Long-hold admits are worth pursuing; short horizons remain noise.
- **AUD_USD** – Essentially flat to negative beyond 30 m. Until cost-adjusted expectancy turns positive, keep AUD in “monitor only.”
- **NZD_USD** – Broadly negative across all horizons (−5.4 bp at 240 m, only 40 % positive). Disable this instrument until the manifold is re-tuned.
- **USD_CAD / USD_CHF** – Slight positive drift at 60 m/240 m but sub-1 bp. Treat as neutral; they require real cost modelling before being allowed back on.
- **USD_JPY** – Mixed: +0.10 bp at 5 m but flat at 30 m and mildly negative at 60 m. Needs further guardrail/AUROC analysis before promotion.

Use the JSON output to compute AUROC, decile/quantile lifts, and cost-adjusted simulations as you tighten policy. Most importantly, every future dashboard update should link to these weekly evidence drops instead of celebrating 3‑hour blips.

## 7. Regime-aligned price-change mapping

The next phase is to anchor every gate to the **instantaneous rate of change** (ROC) of price so we can ask “which regimes preceded this move?” instead of blindly counting admits.

### 7.1 Instantaneous ROC extractor
1. Stream M1 candles from OANDA (same payloads already stored under `md:candles:*`).
2. For each candle, compute `roc = (close - prev_close) / prev_close`, storing both absolute and pip-normalised deltas.
3. Maintain a rolling z-score of ROC so we can classify “spikes” versus noise; persist alongside the candle so downstream tooling can read it without recomputing.

### 7.2 Signal → price alignment
1. For every gate event (`gate:index:{instrument}`), pull:
   - Regime label + confidence
   - Semantic tags / signal signature
   - Hazard, lambda, and structural quartiles
2. Join the gate timestamp to the nearest candle and capture:
   - `roc_prev`: instantaneous ROC at the gate timestamp (what the market just did)
   - `roc_next_{h}`: ROC over each forward horizon already in the outcome study (5/15/30/60/240 m)
3. Store these joins in a fact table (Valkey hash or flat Parquet) so we can query “when regime=trend_bear & hazard<0.45, what did `roc_next_60` look like?”

### 7.3 Regime clustering strategy
- **Primary axes**: regime label, hazard decile, signature repetition count.
- **Secondary context**: semantic tags, structural slopes (coherence_tau, domain_wall), session state (open vs exit window).
- **Outcome bins**: classify each forward ROC as positive, neutral, or negative with configurable pip thresholds; compute conditional probabilities and AUROC per cluster.

This gives us a loose but testable mapping from the *set of signals* to the price structures they precede, which is exactly the “complex trendline déjà vu” we are chasing. Once a cluster exhibits a stable lift, we can graduate it to a backtest candidate.

### 7.4 Current weekly observations (11 Nov 2025 snapshot)
Using `docs/evidence/outcome_weekly_costs.json` (dated 2025‑11‑11 00:50 UTC):

- **No instrument cleared the cost-adjusted bar** – every horizon still prints negative average returns after applying median spread + half-slip + 0.1 pip commission. Positive outcome ratios climb with horizon (e.g., EUR/USD hits 49.5 % at 240 m), but not enough to cover costs.
- **NZD/USD remains the worst offender** – −9.3 bp at 240 m with only 28 % positive outcomes. These ROC bins will serve as the “anti-pattern” cluster when training the mapping logic.
- **USD pairs show the cleanest structure** – USD/JPY and USD/CAD trend toward neutrality at long horizons (46 % positive) with AUROC ≈ 0.50, suggesting hazard/semantic filters, not the manifold itself, are constraining expectancy.
- **AUD/USD and NZD/USD share the same failure mode** – hazard deciles never produce lift; use them as control groups when validating the ROC-regime mapping (if the method flags these as “no edge,” it’s working).

Action items coming out of this study:
1. Backfill the ROC-enhanced fact table described above for the same 3–10 Nov window so we can query it alongside the existing outcome JSON.
2. Prototype a “regime hit-map” visual in the dashboard that, for a selected horizon, colors each regime label by forward ROC sign.
3. Define promotion criteria: e.g., “regime X + hazard decile ≤3 + repetition ≥4” must deliver ≥0.8 bp cost-adjusted lift at 60 m before we allow it into a grid backtest.

Once these pieces are in place, we can finally talk about testing and live configuration tweaks with evidence instead of anecdotes.

### 7.5 Promotion criteria for backtests

Before queuing any new backtest or profile tweak, insist on the following:

1. **Data sufficiency** – at least 100 samples for the cluster (regime label + hazard decile + repetition bucket) in the ROC fact table, spanning ≥5 trading days.
2. **Directional lift** – cost-adjusted `roc_forward_pct[60]` ≥ +0.80 bp and positive percentage ≥ 0.6. For 240 m holds, require ≥ +1.5 bp.
3. **Structural agreement** – hazard decile ≤ 3 (best 30 %), `repetitions ≥ 4`, and no outstanding guardrail reasons in the gate payloads.
4. **Stability check** – the same cluster must meet the thresholds in two consecutive weekly studies (archive links in `docs/evidence/`).

Only when all four are satisfied do we green-light a focused grid test (single instrument, narrow parameter slice). This keeps the research backlog aligned with statistically defensible signals.
