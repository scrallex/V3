"""Snapshot structures and KPI helpers for telemetry records."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


@dataclass(slots=True)
class TelemetrySnapshot:
    """Canonical miner telemetry row (1 Hz baseline)."""

    timestamp: datetime
    rig_id: str

    hashrate_5s: float
    hashrate_1m: float
    accepted_shares: int
    rejected_shares: int
    stale_shares: int
    power_w: float
    voltage_v: float
    current_a: float
    temp_core_c: float
    temp_mem_c: float
    fan_rpm: float
    clock_core_mhz: float
    clock_mem_mhz: float
    plimit_w: float
    asic_chip_errs: int
    throttle_flags: int
    net_rtt_ms: float
    stratum_latency_ms: float
    submit_latency_ms: float
    pool_diff: float
    job_rate_hz: float

    # Optional extended metrics
    temp_target_c: Optional[float] = None
    reject_rate_ema: Optional[float] = None
    efficiency_ema: Optional[float] = None

    def efficiency(self) -> float:
        power = max(self.power_w, 1e-6)
        hashrate_gh = self.hashrate_1m / 1e9  # convert H/s to GH/s
        return hashrate_gh / power

    def reject_rate(self) -> float:
        accepted = max(self.accepted_shares, 0)
        rejected = max(self.rejected_shares, 0)
        stale = max(self.stale_shares, 0)
        total = accepted + rejected + stale
        if total <= 0:
            return 0.0
        return rejected / total

    def thermal_headroom(self, throttle_c: float) -> float:
        return max(0.0, throttle_c - self.temp_core_c)

    def stability_index(self, window_variance: float) -> float:
        return 1.0 / (1.0 + max(window_variance, 0.0))

    def latency_budget(self, threshold_ms: float = 200.0) -> float:
        return max(0.0, threshold_ms - self.stratum_latency_ms)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TelemetrySnapshot":
        ts_raw = payload.get("timestamp")
        if isinstance(ts_raw, datetime):
            timestamp = ts_raw
        elif isinstance(ts_raw, (int, float)):
            timestamp = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
        elif isinstance(ts_raw, str):
            timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        else:
            timestamp = datetime.now(timezone.utc)

        return cls(
            timestamp=timestamp,
            rig_id=str(payload.get("rig_id", "UNKNOWN")).upper(),
            hashrate_5s=_safe_float(payload.get("hashrate_5s")),
            hashrate_1m=_safe_float(payload.get("hashrate_1m")),
            accepted_shares=_safe_int(payload.get("accepted_shares")),
            rejected_shares=_safe_int(payload.get("rejected_shares")),
            stale_shares=_safe_int(payload.get("stale_shares")),
            power_w=_safe_float(payload.get("power_w")),
            voltage_v=_safe_float(payload.get("voltage_v")),
            current_a=_safe_float(payload.get("current_a")),
            temp_core_c=_safe_float(payload.get("temp_core_c")),
            temp_mem_c=_safe_float(payload.get("temp_mem_c")),
            fan_rpm=_safe_float(payload.get("fan_rpm")),
            clock_core_mhz=_safe_float(payload.get("clock_core_mhz")),
            clock_mem_mhz=_safe_float(payload.get("clock_mem_mhz")),
            plimit_w=_safe_float(payload.get("plimit_w")),
            asic_chip_errs=_safe_int(payload.get("asic_chip_errs")),
            throttle_flags=_safe_int(payload.get("driver_throttle_flags", payload.get("throttle_flags"))),
            net_rtt_ms=_safe_float(payload.get("net_rtt_ms")),
            stratum_latency_ms=_safe_float(payload.get("stratum_latency_ms")),
            submit_latency_ms=_safe_float(payload.get("submit_latency_ms")),
            pool_diff=_safe_float(payload.get("pool_diff")),
            job_rate_hz=_safe_float(payload.get("job_rate_hz")),
            temp_target_c=payload.get("temp_target_c"),
            reject_rate_ema=payload.get("reject_rate_ema"),
            efficiency_ema=payload.get("efficiency_ema"),
        )


def rolling_variance(series: Iterable[float]) -> float:
    """Compute sample variance for a short series."""

    values = list(series)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    accum = sum((v - mean) ** 2 for v in values)
    return accum / (len(values) - 1)
