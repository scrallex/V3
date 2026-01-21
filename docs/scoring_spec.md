# Scoring Specification

This document defines how telemetry snapshots are transformed into structural metrics and
high-level KPIs. The pipeline mirrors the Observe → Score → Act loop:

1. Encode a sliding window of snapshots into a binary bitstream.
2. Analyse the bitstream with the native QFH kernel (`sep_quantum`).
3. Combine structural metrics with explicit KPIs to produce the `sep_score` and rupture flag.

## 1. Bit Encoding

The encoder (`scripts/telemetry/bit_encoder.py`) compares each sample against the previous
one and emits a single composite bit per timestep:

- Bit = 1 when
  - Efficiency (`hashrate_1m / power_w`) is rising or flat,
  - Thermal headroom ≥ 3 °C,
  - Stratum latency is not worsening,
  - Power draw is not spiking (> 5 % increase).
- Bit is forced to 0 if reject rate rises by > 0.1 pp or throttle flags are asserted.

The encoder also exposes recent efficiency/hash-rate series so the scorer can compute
variance and provide stability baselines.

## 2. QFH Metrics

Windows (default 32 samples) are analysed via `sep_text_manifold.native.analyze_window`,
returning:

- `coherence` – structured consistency.
- `rupture_ratio` – instability share of the window.
- `entropy` – randomness estimate.
- `lambda_hazard` – hazard rate combining entropy/coherence.

Live stability is `1 - rupture_ratio`.

## 3. KPIs

For the latest snapshot the scorer computes:

- `efficiency` = `hashrate_1m / power_w`.
- `reject_rate` = `rejected / (accepted + rejected + stale)`.
- `thermal_headroom` = `temp_throttle_c - temp_core_c`.
- `stability_index` = `1 / (1 + var(hashrate_5s_window))`.
- `latency_budget` = `max(0, 200 - stratum_latency_ms)`.

## 4. SEP Score

`sep_score` is a weighted combination (bounded to [50, 100]):

```
sep_score = 100 * clamp(0,
    0.35*coherence +
    0.25*stability +
    0.15*normalized_efficiency +
    0.10*stability_index +
    0.08*(thermal_headroom / 20) +
    0.07*(latency_budget / 200) -
    0.20*entropy -
    0.20*rupture_ratio
)
```

Efficiency is normalised against the 32-sample median and MAD. Thermal and latency terms
are capped at 1.0.

## 5. Rupture Classes

| Class | Conditions |
| --- | --- |
| `macro` | `lambda_hazard ≥ 0.75` or `rupture_ratio ≥ 0.6` |
| `meso` | `lambda_hazard ≥ 0.5` or `rupture_ratio ≥ 0.4` |
| `micro` | `lambda_hazard ≥ 0.3` or `rupture_ratio ≥ 0.2` |
| `none` | Below all thresholds |

Rupture classes trigger burst sampling and actuation cooldowns.

## 6. Output Payload

Scores are published as JSON:

```json
{
  "rig_id": "RIG01",
  "timestamp_ns": 1700000000000000000,
  "metrics": {
    "coherence": 0.82,
    "stability": 0.76,
    "entropy": 0.18,
    "rupture": 0.24,
    "lambda_hazard": 0.38,
    "sep_score": 74.2
  },
  "kpis": {
    "efficiency": 0.46,
    "reject_rate": 0.008,
    "thermal_headroom": 6.5,
    "stability_index": 0.91,
    "latency_budget_ms": 142.0
  },
  "rupture_flag": "micro"
}
```

This payload feeds both the advisory layer and external monitoring pipelines.
