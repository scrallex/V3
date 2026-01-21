"""Telemetry collectors for GPUs and ASIC miners."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, List, Optional

from .snapshot import TelemetrySnapshot
from .miner_clients import MinerClient, MinerStats

try:  # pragma: no cover - optional dependency
    import pynvml
except Exception:  # pragma: no cover - optional dependency
    pynvml = None


logger = logging.getLogger(__name__)


class TelemetryCollector:
    """Collect telemetry snapshots at a fixed cadence."""

    def __init__(
        self,
        rig_id: str,
        *,
        sample_interval: float = 1.0,
        history_size: int = 512,
        source: str = "nvml",
        device_index: int = 0,
        miner_client: Optional[MinerClient] = None,
    ) -> None:
        self.rig_id = rig_id.upper()
        self.sample_interval = max(0.1, sample_interval)
        self.history_size = max(16, history_size)
        self.source = source
        self.device_index = device_index
        self._miner_client = miner_client
        self._history: Deque[TelemetrySnapshot] = deque(maxlen=self.history_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._json_index = 0
        self._nvml_handle = None
        self._accepted_shares = 0
        self._rejected_shares = 0
        self._stale_shares = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self.source == "nvml":
            self._init_nvml()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name=f"{self.rig_id}-collector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self.source == "nvml":
            self._release_nvml()

    def latest(self) -> Optional[TelemetrySnapshot]:
        with self._lock:
            return self._history[-1] if self._history else None

    def window(self, count: int) -> List[TelemetrySnapshot]:
        with self._lock:
            if not self._history:
                return []
            return list(self._history)[-count:]

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                snapshot = self._sample_once()
            except Exception:
                snapshot = None
            if snapshot:
                with self._lock:
                    self._history.append(snapshot)
            time.sleep(self.sample_interval)

    def _sample_once(self) -> Optional[TelemetrySnapshot]:
        if self.source == "nvml":
            return self._sample_nvml()
        if self.source == "json":
            return self._sample_json_file()
        raise ValueError(f"Unsupported telemetry source '{self.source}'")

    # ------------------------------------------------------------------
    # NVML helpers

    def _init_nvml(self) -> None:
        if pynvml is None:
            raise RuntimeError("pynvml is required for NVML telemetry collection")
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as exc:  # pragma: no cover - hardware path
            raise RuntimeError(f"Failed to initialise NVML: {exc}") from exc
        try:
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except pynvml.NVMLError as exc:  # pragma: no cover
            raise RuntimeError(f"Unable to access device index {self.device_index}: {exc}") from exc

    def _release_nvml(self) -> None:
        # NVML has global process state; avoid repeated shutdown/start churn.
        # Shutdown only when no other collectors are active.
        try:
            pynvml.nvmlShutdown()
        except Exception:  # pragma: no cover - best effort
            pass
        self._nvml_handle = None

    def _sample_nvml(self) -> Optional[TelemetrySnapshot]:
        if self._nvml_handle is None:
            return None
        device = self._nvml_handle

        stats: Optional[MinerStats] = None
        if self._miner_client:
            try:
                stats = self._miner_client.fetch()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("miner client fetch failed: %s", exc)

        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(device) / 1000.0
        except pynvml.NVMLError:
            power_w = 0.0

        try:
            temp_core_c = pynvml.nvmlDeviceGetTemperature(device, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            temp_core_c = 0.0

        try:
            fan_pct = pynvml.nvmlDeviceGetFanSpeed(device)
        except (pynvml.NVMLError, TypeError):
            fan_pct = 0.0

        try:
            clocks = pynvml.nvmlDeviceGetClockInfo(device, pynvml.NVML_CLOCK_GRAPHICS)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(device, pynvml.NVML_CLOCK_MEM)
        except pynvml.NVMLError:
            clocks = 0.0
            mem_clock = 0.0

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(device)
            gpu_util = getattr(util, "gpu", 0.0)
        except pynvml.NVMLError:
            gpu_util = 0.0

        try:
            min_limit, _ = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(device)
            voltage = min_limit / 1000.0
        except pynvml.NVMLError:
            voltage = 0.0

        if stats and stats.accepted_shares is not None:
            self._accepted_shares = stats.accepted_shares
        if stats and stats.rejected_shares is not None:
            self._rejected_shares = stats.rejected_shares
        if stats and stats.stale_shares is not None:
            self._stale_shares = stats.stale_shares

        if stats and stats.hashrate_5s is not None:
            hashrate_5s = stats.hashrate_5s
        else:
            hashrate_5s = max(0.0, gpu_util) * 1_000_000.0

        if stats and stats.hashrate_1m is not None:
            hashrate_1m = stats.hashrate_1m
        else:
            hashrate_1m = hashrate_5s

        net_rtt_ms = stats.net_rtt_ms if stats else None
        stratum_latency_ms = stats.stratum_latency_ms if stats else None
        submit_latency_ms = stats.submit_latency_ms if stats else None
        pool_diff = stats.pool_diff if stats else None
        job_rate_hz = stats.job_rate_hz if stats else None

        timestamp = datetime.now(timezone.utc)
        return TelemetrySnapshot(
            timestamp=timestamp,
            rig_id=self.rig_id,
            hashrate_5s=hashrate_5s,
            hashrate_1m=hashrate_1m,
            accepted_shares=self._accepted_shares,
            rejected_shares=self._rejected_shares,
            stale_shares=self._stale_shares,
            power_w=power_w,
            voltage_v=voltage,
            current_a=power_w / max(voltage, 1.0) if voltage > 0 else 0.0,
            temp_core_c=float(temp_core_c),
            temp_mem_c=float(temp_core_c),
            fan_rpm=float(fan_pct) * 100.0,
            clock_core_mhz=float(clocks),
            clock_mem_mhz=float(mem_clock),
            plimit_w=power_w,
            asic_chip_errs=0,
            throttle_flags=0,
            net_rtt_ms=net_rtt_ms or 0.0,
            stratum_latency_ms=stratum_latency_ms or 0.0,
            submit_latency_ms=submit_latency_ms or 0.0,
            pool_diff=pool_diff or 0.0,
            job_rate_hz=job_rate_hz or 0.0,
        )

    def _sample_json_file(self) -> Optional[TelemetrySnapshot]:
        """Support offline prototyping by tailing a JSON lines file."""

        json_path = Path(f"data/{self.rig_id.lower()}_telemetry.ndjson")
        if not json_path.exists():
            return None
        try:
            lines = json_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return None
        if not lines:
            return None
        if self._json_index >= len(lines):
            # Hold the final snapshot once playback completes.
            self._json_index = len(lines) - 1
        line = lines[self._json_index]
        if self._json_index < len(lines) - 1:
            self._json_index += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None
        return TelemetrySnapshot.from_dict(payload)
