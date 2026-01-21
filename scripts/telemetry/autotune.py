"""Guarded auto-tune controller for miner actuators."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple

from .advisory import AdvisoryAction
from .config import RigMetadata
from .scoring import TelemetryScore

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import pynvml
except Exception:  # pragma: no cover - optional dependency
    pynvml = None


class ActuatorDriver(Protocol):
    """Abstract actuator client driven by the auto-tune controller."""

    def apply(self, rig_id: str, actuator: str, delta: float, ttl_seconds: int) -> bool:
        ...

    def rollback(self, rig_id: str, actuator: str) -> bool:
        ...


@dataclass
class PendingAction:
    action: AdvisoryAction
    applied_at: float
    baseline_efficiency: float
    baseline_reject: float
    ttl_seconds: int
    rollback_deadline: float = field(init=False)

    def __post_init__(self) -> None:
        self.rollback_deadline = self.applied_at + 180.0  # default rollback window (3 minutes)


class AutoTuneController:
    """Stateful controller that applies advisory deltas with guard rails."""

    def __init__(self, rig: RigMetadata, driver: ActuatorDriver) -> None:
        self.rig = rig
        self.driver = driver
        self._pending: Optional[PendingAction] = None
        self._cooldown_until: float = 0.0
        self._last_action_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API

    def evaluate(self, score: TelemetryScore, advice: AdvisoryAction) -> bool:
        """Attempt to apply the advisory action, returning True when executed."""

        if advice.delta is None:
            return False  # advisory-only actions are ignored by auto-tune

        now = time.time()
        if now < self._cooldown_until:
            logger.debug("auto-tune cooldown active for %s (%.0fs remaining)", self.rig.rig_id, self._cooldown_until - now)
            return False

        if self._pending and now - self._pending.applied_at < 120.0:
            logger.debug("skipping action for %s: minimum spacing not met", self.rig.rig_id)
            return False

        if now - self._last_action_ts < 120.0:
            logger.debug("last action for %s was %.1fs ago – waiting", self.rig.rig_id, now - self._last_action_ts)
            return False

        if score.rupture_flag in {"meso", "macro"}:
            logger.debug("structure unstable (%s rupture) – deferring auto-tune for %s", score.rupture_flag, self.rig.rig_id)
            return False

        if not self.driver.apply(self.rig.rig_id, advice.actuator, float(advice.delta), advice.ttl_seconds):
            logger.warning("failed to apply %s delta=%s on %s", advice.actuator, advice.delta, self.rig.rig_id)
            return False

        self._pending = PendingAction(
            action=advice,
            applied_at=now,
            baseline_efficiency=score.efficiency,
            baseline_reject=score.reject_rate,
            ttl_seconds=advice.ttl_seconds,
        )
        self._last_action_ts = now
        logger.info("applied %s delta=%s on %s", advice.actuator, advice.delta, self.rig.rig_id)
        return True

    def observe(self, score: TelemetryScore) -> None:
        """Evaluate rollback and cooldown logic given the latest score."""

        now = time.time()
        if self._pending:
            degrade_eff = score.efficiency < self._pending.baseline_efficiency * 0.98
            degrade_reject = score.reject_rate > self._pending.baseline_reject + 0.005
            expired = now >= self._pending.rollback_deadline

            if degrade_eff or degrade_reject:
                logger.warning(
                    "auto-tune regression detected on %s (%s eff Δ=%.3f reject Δ=%.4f) – rolling back",
                    self.rig.rig_id,
                    self._pending.action.actuator,
                    score.efficiency / max(self._pending.baseline_efficiency, 1e-6),
                    score.reject_rate - self._pending.baseline_reject,
                )
                self._rollback()
                self._cooldown_until = max(self._cooldown_until, now + 600.0)
                return

            if expired:
                logger.debug("rollback window elapsed for %s on %s", self.rig.rig_id, self._pending.action.actuator)
                self._pending = None

        if score.rupture_flag == "macro":
            logger.warning("macro rupture detected on %s – entering cooldown", self.rig.rig_id)
            self._cooldown_until = max(self._cooldown_until, now + self.rig.thresholds.rupture_cooldown_s)
            self._rollback()

    # ------------------------------------------------------------------
    # Internals

    def _rollback(self) -> None:
        if not self._pending:
            return
        action = self._pending.action
        if self.driver.rollback(self.rig.rig_id, action.actuator):
            logger.info("rolled back %s on %s", action.actuator, self.rig.rig_id)
        else:
            logger.warning("failed to rollback %s on %s", action.actuator, self.rig.rig_id)
        self._pending = None


class LoggingActuatorDriver:
    """Placeholder driver that only logs applied deltas (no hardware calls)."""

    def __init__(self) -> None:
        self._state: Dict[str, Dict[str, float]] = {}

    def apply(self, rig_id: str, actuator: str, delta: float, ttl_seconds: int) -> bool:
        state = self._state.setdefault(rig_id, {})
        state[actuator] = state.get(actuator, 0.0) + delta
        logger.debug("[mock] apply %s Δ%.2f (ttl=%ss)", actuator, delta, ttl_seconds)
        return True

    def rollback(self, rig_id: str, actuator: str) -> bool:
        state = self._state.setdefault(rig_id, {})
        state.pop(actuator, None)
        logger.debug("[mock] rollback %s", actuator)
        return True


class NVMLActuatorDriver:
    """NVML-backed actuator driver controlling power, clocks, and fans."""

    _nvml_initialised = False

    def __init__(self, rigs: Dict[str, RigMetadata]) -> None:
        if pynvml is None:
            raise RuntimeError("pynvml is required for NVMLActuatorDriver but is not installed")
        if not NVMLActuatorDriver._nvml_initialised:
            pynvml.nvmlInit()
            NVMLActuatorDriver._nvml_initialised = True
        self._rigs = rigs
        self._previous: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    # Public API

    def apply(self, rig_id: str, actuator: str, delta: float, ttl_seconds: int) -> bool:
        rig = self._rig(rig_id)
        actuator = actuator.lower()
        if actuator == "plimit_w":
            return self._apply_power_limit(rig, delta)
        if actuator == "clock_core_mhz":
            return self._apply_core_clock(rig, delta)
        if actuator == "fan_pct":
            return self._apply_fan_speed(rig, delta)
        logger.warning("unsupported actuator '%s' for NVML driver on %s", actuator, rig_id)
        return False

    def rollback(self, rig_id: str, actuator: str) -> bool:
        rig = self._rig(rig_id)
        previous = self._previous.get(rig.rig_id, {})
        entry = previous.pop(actuator, None)
        if entry is None:
            return False
        try:
            handle = self._handle(rig)
            if actuator == "plimit_w":
                target_w = float(entry)
                pynvml.nvmlDeviceSetPowerManagementLimit(handle, int(target_w * 1000))
            elif actuator == "clock_core_mhz":
                graphics, memory = entry  # type: ignore[assignment]
                pynvml.nvmlDeviceSetApplicationsClocks(handle, int(memory), int(graphics))
            elif actuator == "fan_pct" and hasattr(pynvml, "nvmlDeviceSetFanSpeed_v2"):
                target = int(entry)
                try:
                    pynvml.nvmlDeviceSetFanSpeed_v2(handle, 0, target)
                except Exception:
                    return False
            else:
                return False
            logger.debug("restored %s to %s on %s", actuator, entry, rig.rig_id)
            return True
        except pynvml.NVMLError as exc:  # pragma: no cover - hardware path
            logger.error("NVML rollback failed for %s/%s: %s", rig.rig_id, actuator, exc)
            return False

    def shutdown(self) -> None:
        if NVMLActuatorDriver._nvml_initialised:
            try:  # pragma: no cover - best-effort cleanup
                pynvml.nvmlShutdown()
            except Exception:
                pass
            NVMLActuatorDriver._nvml_initialised = False

    # ------------------------------------------------------------------
    # Helpers

    def _rig(self, rig_id: str) -> RigMetadata:
        key = rig_id.upper()
        if key not in self._rigs:
            raise KeyError(f"Unknown rig '{rig_id}' in NVML driver")
        return self._rigs[key]

    def _handle(self, rig: RigMetadata):
        return pynvml.nvmlDeviceGetHandleByIndex(rig.device_index)

    def _apply_power_limit(self, rig: RigMetadata, delta_w: float) -> bool:
        thresholds = rig.thresholds
        if abs(delta_w) > thresholds.power_step_w + 1e-6:
            logger.debug("delta %.2fW exceeds configured step %.2fW for %s", delta_w, thresholds.power_step_w, rig.rig_id)
            return False
        try:
            handle = self._handle(rig)
            current_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            current_w = current_mw / 1000.0
            min_mw, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            min_w = max(min_mw / 1000.0, thresholds.power_limit_min_w)
            max_w = min(max_mw / 1000.0, thresholds.power_limit_max_w)
            target_w = self._clamp(current_w + delta_w, min_w, max_w)
            if math.isclose(target_w, current_w, abs_tol=0.25):
                logger.debug("power limit unchanged (%.2fW) on %s", current_w, rig.rig_id)
                return False
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, int(target_w * 1000))
            self._previous.setdefault(rig.rig_id, {})["plimit_w"] = current_w
            logger.info("power limit adjusted %.1fW -> %.1fW on %s", current_w, target_w, rig.rig_id)
            return True
        except pynvml.NVMLError as exc:  # pragma: no cover - hardware path
            logger.error("NVML power adjustment failed for %s: %s", rig.rig_id, exc)
            return False

    def _apply_core_clock(self, rig: RigMetadata, delta_mhz: float) -> bool:
        thresholds = rig.thresholds
        if abs(delta_mhz) > thresholds.clock_step_mhz + 1e-6:
            logger.debug("clock delta %.1fMHz exceeds step %.1fMHz for %s", delta_mhz, thresholds.clock_step_mhz, rig.rig_id)
            return False
        try:
            handle = self._handle(rig)
            current_graphics = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_GRAPHICS)
            current_memory = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM)
            try:
                default_graphics = pynvml.nvmlDeviceGetDefaultApplicationsClock(handle, pynvml.NVML_CLOCK_GRAPHICS)
            except (AttributeError, pynvml.NVMLError):
                default_graphics = current_graphics
            min_clock = default_graphics + thresholds.clock_core_offset_min_mhz
            max_clock = default_graphics + thresholds.clock_core_offset_max_mhz
            target_graphics = self._clamp(current_graphics + delta_mhz, min_clock, max_clock)
            if math.isclose(target_graphics, current_graphics, abs_tol=1.0):
                logger.debug("graphics clock unchanged (%.0fMHz) on %s", current_graphics, rig.rig_id)
                return False
            pynvml.nvmlDeviceSetApplicationsClocks(handle, int(current_memory), int(target_graphics))
            self._previous.setdefault(rig.rig_id, {})["clock_core_mhz"] = (current_graphics, current_memory)
            logger.info("core clock adjusted %.0fMHz -> %.0fMHz on %s", current_graphics, target_graphics, rig.rig_id)
            return True
        except pynvml.NVMLError as exc:  # pragma: no cover
            logger.error("NVML clock adjustment failed for %s: %s", rig.rig_id, exc)
            return False

    def _apply_fan_speed(self, rig: RigMetadata, delta_pct: float) -> bool:
        thresholds = rig.thresholds
        if abs(delta_pct) > thresholds.fan_step_pct + 1e-6:
            logger.debug("fan delta %.1f%% exceeds step %.1f%% for %s", delta_pct, thresholds.fan_step_pct, rig.rig_id)
            return False
        if not hasattr(pynvml, "nvmlDeviceSetFanSpeed_v2"):
            logger.warning("fan control not supported by installed NVML on %s", rig.rig_id)
            return False
        try:
            handle = self._handle(rig)
            try:
                current_pct = pynvml.nvmlDeviceGetFanSpeed(handle, 0)
            except TypeError:
                current_pct = pynvml.nvmlDeviceGetFanSpeed(handle)
            target_pct = int(self._clamp(current_pct + delta_pct, thresholds.fan_percent_min, thresholds.fan_percent_max))
            if target_pct == current_pct:
                logger.debug("fan speed unchanged (%s%%) on %s", current_pct, rig.rig_id)
                return False
            pynvml.nvmlDeviceSetFanSpeed_v2(handle, 0, target_pct)
            self._previous.setdefault(rig.rig_id, {})["fan_pct"] = current_pct
            logger.info("fan speed adjusted %d%% -> %d%% on %s", current_pct, target_pct, rig.rig_id)
            return True
        except pynvml.NVMLError as exc:  # pragma: no cover
            logger.error("NVML fan adjustment failed for %s: %s", rig.rig_id, exc)
            return False

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))
