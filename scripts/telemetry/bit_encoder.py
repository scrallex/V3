"""Convert telemetry snapshots into structural bitstreams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .snapshot import TelemetrySnapshot


@dataclass(slots=True)
class TelemetryBitEncoder:
    """Encode sliding windows into composite bits for QFH analysis."""

    lookback: int = 32

    def encode(self, window: Sequence[TelemetrySnapshot]) -> List[int]:
        if len(window) < 2:
            return []

        bits: List[int] = []
        for idx in range(1, len(window)):
            prev = window[idx - 1]
            curr = window[idx]

            eff_now = curr.efficiency()
            eff_prev = prev.efficiency()
            eff_trending = eff_now >= eff_prev * 0.995

            reject_now = curr.reject_rate()
            reject_prev = prev.reject_rate()
            reject_worsening = reject_now > reject_prev + 0.001

            thermal_headroom = curr.thermal_headroom(curr.temp_target_c or (curr.temp_core_c + 15.0))
            cooling_margin = thermal_headroom >= 2.0

            latency_ok = curr.stratum_latency_ms <= max(120.0, prev.stratum_latency_ms * 1.05)
            power_ok = curr.power_w <= max(prev.power_w + 15.0, prev.power_w * 1.07)

            bit = 1 if (eff_trending and cooling_margin and latency_ok and power_ok) else 0
            if reject_worsening or curr.throttle_flags:
                bit = 0
            bits.append(bit)
        return bits

    def recent_efficiency(self, window: Sequence[TelemetrySnapshot]) -> Iterable[float]:
        for snap in window[-self.lookback :]:
            yield snap.efficiency()

    def recent_hashrate(self, window: Sequence[TelemetrySnapshot]) -> Iterable[float]:
        for snap in window[-self.lookback :]:
            yield snap.hashrate_5s
