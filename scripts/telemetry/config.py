"""Configuration helpers for the telemetry optimizer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


@dataclass(frozen=True)
class MinerAPIConfig:
    """Connection details for a miner telemetry endpoint."""

    kind: str
    url: str
    timeout: float = 1.0


@dataclass(frozen=True)
class RigThresholds:
    """Per-rig guardrail settings."""

    sep_score_min: float = 70.0
    reject_rate_max: float = 0.02
    power_step_w: float = 5.0
    clock_step_mhz: float = 15.0
    fan_step_pct: float = 10.0
    rupture_cooldown_s: int = 600
    power_limit_min_w: float = 250.0
    power_limit_max_w: float = 360.0
    clock_core_offset_min_mhz: float = -150.0
    clock_core_offset_max_mhz: float = 150.0
    fan_percent_min: float = 25.0
    fan_percent_max: float = 95.0


@dataclass(frozen=True)
class RigMetadata:
    """Static metadata describing the hardware footprint."""

    rig_id: str
    source: str
    miner_api: Optional[MinerAPIConfig]
    miner_type: str
    pool: str
    location: str
    psu_model: str
    ambient_c: float
    temp_throttle_c: float
    device_index: int = 0
    thresholds: RigThresholds = field(default_factory=RigThresholds)


@dataclass(frozen=True)
class TelemetryConfig:
    """Top-level configuration shared across the telemetry pipeline."""

    rigs: Dict[str, RigMetadata]
    sample_interval_s: float = 1.0
    burst_interval_s: float = 0.2
    burst_duration_s: float = 10.0
    window_size: int = 32
    history_size: int = 512
    redis_url: Optional[str] = None
    ndjson_path: Path = Path("logs/telemetry")
    driver: str = "logging"
    autotune_enabled: bool = False

    def rig(self, rig_id: str) -> RigMetadata:
        key = rig_id.upper()
        if key not in self.rigs:
            raise KeyError(f"Rig '{rig_id}' not present in telemetry config")
        return self.rigs[key]


def _load_thresholds(payload: Mapping[str, Any] | None) -> RigThresholds:
    if not payload:
        return RigThresholds()
    return RigThresholds(
        sep_score_min=float(payload.get("sep_score_min", 70.0)),
        reject_rate_max=float(payload.get("reject_rate_max", 0.02)),
        power_step_w=float(payload.get("power_step_w", 5.0)),
        clock_step_mhz=float(payload.get("clock_step_mhz", 15.0)),
        fan_step_pct=float(payload.get("fan_step_pct", 10.0)),
        rupture_cooldown_s=int(payload.get("rupture_cooldown_s", 600)),
        power_limit_min_w=float(payload.get("power_limit_min_w", 250.0)),
        power_limit_max_w=float(payload.get("power_limit_max_w", 360.0)),
        clock_core_offset_min_mhz=float(payload.get("clock_core_offset_min_mhz", -150.0)),
        clock_core_offset_max_mhz=float(payload.get("clock_core_offset_max_mhz", 150.0)),
        fan_percent_min=float(payload.get("fan_percent_min", 25.0)),
        fan_percent_max=float(payload.get("fan_percent_max", 95.0)),
    )


def _load_miner_api(payload: Mapping[str, Any] | None) -> Optional[MinerAPIConfig]:
    if not payload:
        return None
    kind = payload.get("kind")
    url = payload.get("url")
    if not kind or not url:
        return None
    return MinerAPIConfig(
        kind=str(kind).lower(),
        url=str(url),
        timeout=float(payload.get("timeout", 1.0)),
    )


def _load_rig(name: str, payload: Mapping[str, Any]) -> RigMetadata:
    thresholds = _load_thresholds(payload.get("thresholds"))
    return RigMetadata(
        rig_id=name.upper(),
        source=str(payload.get("source", "nvml")),
        miner_api=_load_miner_api(payload.get("miner_api")),
        miner_type=str(payload.get("miner_type", "unknown")),
        pool=str(payload.get("pool", "unknown")),
        location=str(payload.get("location", "unknown")),
        psu_model=str(payload.get("psu_model", "unknown")),
        ambient_c=float(payload.get("ambient_c", 25.0)),
        temp_throttle_c=float(payload.get("temp_throttle_c", 90.0)),
        device_index=int(payload.get("device_index", 0)),
        thresholds=thresholds,
    )


def load_config(path: str | os.PathLike[str]) -> TelemetryConfig:
    """Load telemetry optimizer configuration from YAML."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    rigs_raw: Mapping[str, Any] = data.get("rigs") or {}
    rigs: Dict[str, RigMetadata] = {}
    for name, spec in rigs_raw.items():
        if not isinstance(spec, Mapping):
            continue
        rig = _load_rig(name, spec)
        rigs[rig.rig_id] = rig

    if not rigs:
        raise ValueError(f"No rigs defined in {config_path}")

    redis_url = data.get("redis_url") or os.getenv("VALKEY_URL") or os.getenv("REDIS_URL")
    ndjson_path = Path(data.get("ndjson_path", "logs/telemetry")).expanduser()

    return TelemetryConfig(
        rigs=rigs,
        sample_interval_s=float(data.get("sample_interval_s", 1.0)),
        burst_interval_s=float(data.get("burst_interval_s", 0.2)),
        burst_duration_s=float(data.get("burst_duration_s", 10.0)),
        window_size=int(data.get("window_size", 32)),
        history_size=int(data.get("history_size", 512)),
        redis_url=redis_url,
        ndjson_path=ndjson_path,
        driver=str(data.get("driver", "logging")).lower(),
        autotune_enabled=bool(data.get("autotune_enabled", False)),
    )
