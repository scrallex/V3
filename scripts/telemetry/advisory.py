"""Advisory recommendations derived from telemetry scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import RigMetadata
from .scoring import TelemetryScore


@dataclass(frozen=True)
class AdvisoryAction:
    """Suggested adjustment for operators or the guarded auto-tuner."""

    rig_id: str
    actuator: str
    delta: float | None
    reason: str
    ttl_seconds: int

    def to_dict(self) -> dict[str, object]:
        return {
            "rig_id": self.rig_id,
            "actuator": self.actuator,
            "delta": self.delta,
            "reason": self.reason,
            "ttl_seconds": self.ttl_seconds,
        }


def generate_advice(score: TelemetryScore, rig: RigMetadata, previous: Optional[TelemetryScore]) -> List[AdvisoryAction]:
    thresholds = rig.thresholds
    actions: List[AdvisoryAction] = []

    if score.sep_score < thresholds.sep_score_min:
        actions.append(
            AdvisoryAction(
                rig_id=score.rig_id,
                actuator="plimit_w",
                delta=-thresholds.power_step_w,
                reason=f"sep_score {score.sep_score:.1f} below {thresholds.sep_score_min}",
                ttl_seconds=120,
            )
        )

    if score.reject_rate > thresholds.reject_rate_max:
        delta = None
        reason = f"reject_rate {score.reject_rate:.3f} exceeds {thresholds.reject_rate_max}"
        actions.append(
            AdvisoryAction(rig_id=score.rig_id, actuator="pool_switch", delta=delta, reason=reason, ttl_seconds=60)
        )

    if score.thermal_headroom < 3.0:
        actions.append(
            AdvisoryAction(
                rig_id=score.rig_id,
                actuator="fan_pct",
                delta=thresholds.fan_step_pct,
                reason=f"thermal headroom {score.thermal_headroom:.1f}°C",
                ttl_seconds=180,
            )
        )

    if score.rupture_flag == "macro":
        actions.append(
            AdvisoryAction(
                rig_id=score.rig_id,
                actuator="clock_core_mhz",
                delta=-thresholds.clock_step_mhz,
                reason="macro rupture detected",
                ttl_seconds=thresholds.rupture_cooldown_s,
            )
        )
    elif score.rupture_flag == "micro" and previous and previous.rupture_flag == "none":
        actions.append(
            AdvisoryAction(
                rig_id=score.rig_id,
                actuator="fan_pct",
                delta=thresholds.fan_step_pct,
                reason="micro rupture emerging",
                ttl_seconds=90,
            )
        )

    return actions[:2]
