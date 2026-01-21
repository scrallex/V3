#!/usr/bin/env python3
"""Generate synthetic telemetry replay data for local testing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class Segment:
    duration: int
    hashrate_gh: float
    power_w: float
    temp_core_c: float
    latency_ms: float
    fan_pct: float


def _build_segments() -> Iterable[Segment]:
    return [
        Segment(duration=40, hashrate_gh=72.0, power_w=320.0, temp_core_c=66.0, latency_ms=120.0, fan_pct=55.0),
        Segment(duration=30, hashrate_gh=75.0, power_w=315.0, temp_core_c=64.5, latency_ms=112.0, fan_pct=58.0),
        Segment(duration=30, hashrate_gh=78.0, power_w=312.0, temp_core_c=63.5, latency_ms=108.0, fan_pct=59.0),
    ]


def main() -> None:
    root = Path("data")
    root.mkdir(exist_ok=True)
    path = root / "rig01_telemetry.ndjson"

    start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    timestamp = start

    accepted = 20_000
    rejected = 120
    stale = 60

    records = []
    for segment in _build_segments():
        for i in range(segment.duration):
            jitter = 1.0 + 0.003 * (-1) ** i
            hashrate_gh = segment.hashrate_gh * jitter
            power_w = segment.power_w * (1.0 + 0.002 * (-1) ** (i + 1))
            temp = segment.temp_core_c + 0.3 * (-1) ** i
            latency = segment.latency_ms + 1.5 * (-1) ** (i // 4)
            fan_pct = segment.fan_pct + 0.4 * (-1) ** (i // 5)

            accepted += 9
            rejected += 0 if i % 15 else 1
            stale += 0 if i % 25 else 1

            record = {
                "timestamp": timestamp.isoformat(),
                "rig_id": "RIG01",
                "hashrate_5s": hashrate_gh * 0.98 * 1e9,
                "hashrate_1m": hashrate_gh * 1e9,
                "accepted_shares": accepted,
                "rejected_shares": rejected,
                "stale_shares": stale,
                "power_w": power_w,
                "voltage_v": 230.0,
                "current_a": power_w / 230.0,
                "temp_core_c": temp,
                "temp_mem_c": temp + 4.5,
                "fan_rpm": fan_pct * 100.0,
                "clock_core_mhz": 1815.0,
                "clock_mem_mhz": 9500.0,
                "plimit_w": 330.0,
                "asic_chip_errs": 0,
                "driver_throttle_flags": 0,
                "net_rtt_ms": 118.0,
                "stratum_latency_ms": latency,
                "submit_latency_ms": latency + 4.0,
                "pool_diff": 7.4,
                "job_rate_hz": 0.92,
                "temp_target_c": 74.0,
            }
            records.append(record)
            timestamp += timedelta(seconds=1)

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")
    print(f"Wrote {len(records)} snapshots to {path}")


if __name__ == "__main__":
    main()
