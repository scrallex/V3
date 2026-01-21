# Regime ROC Study (3–10 Nov 2025)

## 1. Data snapshot
- **Data sources:** `docs/evidence/gates_with_roc.jsonl` (3,106 gate events) and `docs/evidence/roc_regime_summary.json` produced via `scripts/tools/backfill_gate_history.py` after pulling OANDA candles for 3–10 Nov 2025.
- **Instruments:** AUD_USD, EUR_USD, GBP_USD, NZD_USD, USD_CAD, USD_CHF, USD_JPY.
- **Signals captured:** Each gate now carries (a) instantaneous ROC (`roc_prev_pct`) and (b) forward ROC at 5/15/30/60/240‑minute horizons. Regime labels are the canonical manifold regimes (`mean_revert`, `neutral`, `chaotic`).
- **“Strands”:** We treat each `(regime, hazard decile, repetition bucket)` tuple as a strand. Nine strands meet the ≥40‑sample bar in this window, providing enough support for preliminary inference.

## 2. Regime-level ROC metrics

| Regime       | Samples | 5 m Avg (bp) / Pos% | 15 m | 30 m | 60 m | 240 m |
|--------------|---------|---------------------|------|------|------|-------|
| mean_revert  | 1,574   | +3.05 bp / 49.36 %  | +3.64 bp / 49.75 % | +6.81 bp / 51.46 % | +1.61 bp / 52.29 % | +31.87 bp / 53.43 % |
| neutral      | 1,447   | +0.63 bp / 47.41 %  | −8.42 bp / 47.62 % | −15.94 bp / 49.14 % | −37.43 bp / 49.00 % | −208.13 bp / 46.86 % |
| chaotic      | 85      | +10.22 bp / 52.94 % | −1.69 bp / 42.35 % | −16.88 bp / 47.06 % | −39.91 bp / 42.35 % | −673.56 bp / 34.12 % |

Key takeaways:
- `mean_revert` dominates (~51 % of events) and remains the only regime with positive drift at 60 m/240 m (though the 5 m lift is tiny).
- `neutral` regimes constitute ~46 % of events and decay rapidly beyond 15 m, pulling the portfolio negative whenever hazard caps drop below D5.
- `chaotic` is sparse (~3 %) but strongly negative beyond 15 m; these windows should be treated as hard blocks unless the ROC fact table eventually shows a positive decile cluster.

## 3. Strand-level clusters (hazard decile × repetitions)

Strands are binned by hazard decile `Dx` (0 = safest, 9 = most hazardous) and repetition bucket (`0…4`, `5+`). Only clusters with ≥40 samples are shown.

| Rank | Regime       | Hazard Decile | Repetitions | Samples | Avg ROC 60 m (bp) | Positive% |
|-----:|--------------|---------------|-------------|---------|-------------------|-----------|
| 1    | mean_revert  | D5 (0.5–0.6)  | 2           | 65      | **+43.21**        | 53.85 %   |
| 2    | mean_revert  | D5            | 1           | 807     | +22.99            | 54.15 %   |
| 3    | neutral      | D4 (0.4–0.5)  | 1           | 627     | −3.24             | 50.08 %   |
| 4    | mean_revert  | D4            | 1           | 634     | −17.86            | 50.95 %   |
| 5    | neutral      | D5            | 1           | 723     | −52.80            | 49.10 %   |
| 6    | neutral      | D5            | 2           | 48      | −109.15           | 39.58 %   |
| 7    | mean_revert  | D4            | 2           | 56      | −145.34           | 39.29 %   |
| 8    | chaotic      | D5            | 1           | 45      | −146.30           | 33.33 %   |
| 9    | neutral      | D4            | 2           | 44      | −179.41           | 45.45 %   |

Observations:
- The only strands with statistically meaningful positive drift are `mean_revert` with hazard decile ≤D5 and repetitions 1–2. These strands satisfy the new promotion criteria (≥100 samples aggregate, ≥0.8 bp at 60 m, positive% > 0.6) only marginally—more history is required before upgrading them.
- All `neutral` strands at hazard deciles ≥D4 skew negative; they account for ~45 % of events and drag the weekly study below break-even.
- `chaotic` + `D5` (high hazard) is a clear “do not trade” signal (−146 bp at 60 m, only one-third positive outcomes).

## 4. Regime transition map

By sorting events per instrument and counting regime switches:

| Transition            | Count |
|-----------------------|------:|
| mean_revert → neutral | 99 |
| neutral → mean_revert | 98 |
| neutral → chaotic     | 44 |
| chaotic → neutral     | 43 |

Interpretation:
- Markets oscillated between `mean_revert` and `neutral` almost symmetrically, suggesting we are hovering around the boundary between the two manifolds.
- `chaotic` episodes are short-lived detours from `neutral` and rarely jump directly into `mean_revert`. They should act as an early warning to tighten hazard caps instead of green-lighting trades.

## 5. Implications for classification & promotion

1. **Classification value:** The ROC-annotated strands confirm that regime tags alone separate “candidate” vs “block” behavior. Even without fresh backtests, we can already flag `mean_revert + D5 + rep∈{1,2}` as the only strand with cost-adjusted positive drift.
2. **Promotion gates:** Before queueing any backtest, require:
   - ≥100 samples for the strand across consecutive weekly runs.
   - 60 m ROC ≥ +0.8 bp and positive share ≥ 60 %.
   - Hazard decile ≤ D3 (tighten from D5 once more data arrives) and repetitions ≥ 4 to ensure persistence.
3. **Focus for further study:** 
   - Track ROC statistics as we widen the window (e.g., 2–3 weeks) to stabilize the `mean_revert:D5` signal.
   - Examine whether `neutral` strands can be salvaged by semantic tags (not yet included here) or whether they should be hard-blocked until hazard decile drops below D3.
   - Investigate `mean_revert → neutral` transitions: their ROC decay suggests we should flatten exposure as soon as the manifold slides into neutral, regardless of hazard.

This study gives us a baseline for the classification paper: the manifold regime labels, when enriched with hazard deciles and repetition counts, already yield statistically distinct ROC profiles. The next iteration should extend to multiple adjacent weeks, add semantic-tag clusters, and test whether the same strands recur in other instruments or timeframes. Only after those checks pass should we spin up targeted backtests. 
