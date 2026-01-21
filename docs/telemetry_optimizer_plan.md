# Telemetry Optimizer Integration Plan

This plan lays out how the miner telemetry optimizer plugs into the existing SEP stack
while reusing the native quantum metrics and streaming scaffolding already in the repo.

## High-Level Flow

```
Rig Telemetry → Snapshot Buffer → Bit Encoder → QFH Metrics → SEP Score + KPIs
                           ↓                              ↓
                        Parquet                      Valkey / Prometheus
                           ↓                              ↓
                    Study Protocol               Advisory & Auto-Tune Control
```

1. **Collect** metrics from GPU/NVML or ASIC APIs at 1 Hz with bursts to 5–10 Hz when a rupture is suspected.
2. **Encode** the short history into structural bit flags (up/accel/range/zpos) plus raw values.
3. **Score** the encoded windows with the existing `sep_quantum` bindings to obtain coherence/stability/entropy/rupture and lambda hazard.
4. **Combine** structural metrics with explicit KPIs (efficiency, reject rate, thermal headroom, latency budget) to produce a `sep_score` and rupture class.
5. **Stream** results into Valkey (`telemetry:last:{rig}`) for fast consumers and persist full histories into Parquet for ABA analysis.
6. **Advise/Act** based on guardrails mirroring `scripts/trading/portfolio_manager.py`: advisory suggestions first, then guarded auto-tune after validation.

## Modules to Reuse

- **Native QFH**: The pybind11 module in `score/setup.py` (`sep_quantum`) already exports `analyze_window`.
- **Bit Encoding**: `score/scripts/encode_struct_bits.py` demonstrates structural deltas for telemetry features.
- **Streaming Runtime**: `score/src/stm_stream/runtime.py` handles tailing windows into a FastAPI server.
- **Guard Heuristics**: `scripts/trading/guards.py` provides a throttle-like curve we can adapt for actuator nudging.

## New Components (under `scripts/telemetry/`)

| Module | Responsibility |
| --- | --- |
| `config.py` | Load rig metadata, threshold YAML, and validation helpers. |
| `snapshot.py` | Define the canonical 1 Hz telemetry record structure and derived KPIs. |
| `ingest.py` | Collect metrics from NVML/ASIC/pool APIs and push snapshots into an in-memory buffer. |
| `bit_encoder.py` | Convert recent snapshots into encoded bit arrays suitable for `sep_quantum`. |
| `scoring.py` | Call native QFH, compute rupture buckets, SEP score, and prepare Valkey payloads. |
| `storage.py` | Persist rolling snapshots to Parquet and expose a Prometheus gauge surface. |
| `runtime.py` | Orchestrate ingest → score → stream (advisory hooks, action queue placeholders). |
| `advisory.py` | Generate human-readable recommendations per rig. |
| `autotune.py` | Guarded actuator executor (power limit, clocks, fan curve) with rollback logic. |

## Data Contracts

- **Snapshot**: includes raw telemetry fields (`hashrate_5s`, `power_w`, `temp_core_c`, etc.) plus EMA deltas.
- **Scoring Output**: JSON with structural metrics, KPIs, rupture class, and `sep_score`. Stored in Valkey at `telemetry:last:{rig}` plus appended to an NDJSON log for audit.
- **Control Intent**: Advisory records (`kind`, `target`, `delta`, `reason`, `ttl`) published to `telemetry:advice:{rig}`; executed actions mirrored to `telemetry:actions:{rig}`.

## GPU Acceleration

- Heavy scoring relies on the native `sep_quantum` C++ implementation, which is already vectorised and benefits from the 3080 Ti.
- Future work: benchmark batching multiple windows per CUDA stream once the current bindings expose GPU kernels. For now, the plan queues window analysis on the 3080 Ti via the compiled extension.

## Deliverables

1. `scripts/telemetry/` package with ingest, scoring, and control scaffolding.
2. Prometheus exporter and Valkey integration for live dashboards (`telemetry_exporter.py`).
3. Docs additions: this plan, plus control policy and scoring specs under `docs/`.
4. Study harness reusing `score/src/stm_backtest/backtester.py` patterns for ABA analysis on rig telemetry.
5. Configurable YAML thresholds enabling fast per-rig tuning without code edits.

With this layout we can build the Observe → Score → Act loop quickly while leaning on the QFH kernel and streaming utilities already maintained in the SEP and SCORE repositories.
