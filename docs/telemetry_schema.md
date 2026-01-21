# Telemetry Schema

The telemetry optimizer samples each rig at 1 Hz (with burst sampling during rupture events)
and records the following fields. All numeric values are stored as IEEE‑754 doubles unless
otherwise specified.

| Field | Type | Description |
| --- | --- | --- |
| `timestamp` | ISO‑8601 string | UTC timestamp for the snapshot. |
| `rig_id` | string | Uppercase rig identifier (`RIG01`, `RIG02`, …). |
| `hashrate_5s` | float | Rolling 5‑second hash rate (H/s). |
| `hashrate_1m` | float | Rolling 1‑minute hash rate (H/s). |
| `accepted_shares` | int | Shares accepted since miner start. |
| `rejected_shares` | int | Shares rejected since miner start. |
| `stale_shares` | int | Shares marked stale since miner start. |
| `power_w` | float | Instantaneous power draw (W). |
| `voltage_v` | float | Reported PSU voltage (V). |
| `current_a` | float | Calculated amperage (A). |
| `temp_core_c` | float | GPU/ASIC core temperature (°C). |
| `temp_mem_c` | float | Memory temperature (°C). |
| `fan_rpm` | float | Fan speed (RPM equivalent; converted from percentage when necessary). |
| `clock_core_mhz` | float | Core clock (MHz). |
| `clock_mem_mhz` | float | Memory clock (MHz). |
| `plimit_w` | float | Active power limit (W). |
| `asic_chip_errs` | int | Chip error counters (ASIC) or ECC errors (GPU). |
| `throttle_flags` | int | Vendor-specific throttle flags bitmask. |
| `net_rtt_ms` | float | Measured pool round-trip time (ms). |
| `stratum_latency_ms` | float | Average stratum submission latency (ms). |
| `submit_latency_ms` | float | Latency between share ready and submit (ms). |
| `pool_diff` | float | Pool difficulty for current job. |
| `job_rate_hz` | float | New job cadence (Hz). |
| `temp_target_c` | float (optional) | Target operating temperature (°C). |
| `reject_rate_ema` | float (optional) | EMA of reject rate (ratio). |
| `efficiency_ema` | float (optional) | EMA of efficiency (GH/J). |

Snapshots are persisted as NDJSON lines (`logs/telemetry/<rig>.ndjson`) and mirrored into
Valkey under `telemetry:last:{rig}` for low-latency consumers. The schema is intentionally
flat so Prometheus exporters can map fields directly to gauges.

## Burst Sampling

When a window’s rupture score crosses the `micro` threshold the collector increases sample
frequency to 5–10 Hz for up to 10 s. Burst samples reuse the same schema and simply carry
more granular timestamps. Downstream scoring modules automatically align these to the
rolling window history.
