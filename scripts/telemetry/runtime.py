"""Runtime orchestration for the telemetry optimizer."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from .advisory import AdvisoryAction, generate_advice
from .autotune import AutoTuneController, LoggingActuatorDriver, NVMLActuatorDriver
from .bit_encoder import TelemetryBitEncoder
from .config import TelemetryConfig, load_config
from .ingest import TelemetryCollector
from .scoring import QFHScorer, TelemetryScore
from .storage import TelemetryStorage
from .miner_clients import create_miner_client, MinerClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionEvent:
    rig_id: str
    actuator: str
    delta: float
    timestamp: float


class TelemetryPipeline:
    """End-to-end pipeline: collect → score → publish."""

    def __init__(self, config: TelemetryConfig) -> None:
        self.config = config
        self.encoder = TelemetryBitEncoder(lookback=config.window_size)
        self.scorer = QFHScorer(self.encoder)
        self.storage = TelemetryStorage(redis_url=config.redis_url, ndjson_root=config.ndjson_path)
        self.collectors: Dict[str, TelemetryCollector] = {}
        self._latest_scores: Dict[str, TelemetryScore] = {}
        self._miner_clients: Dict[str, MinerClient] = {}
        self._controllers: Dict[str, AutoTuneController] = {}
        self._driver = self._build_driver()
        self._threads: Deque[threading.Thread] = deque()
        self._stop = threading.Event()
        self._action_lock = threading.Lock()
        self._action_events: Deque[ActionEvent] = deque(maxlen=512)
        self.autotune_enabled = config.autotune_enabled

    def _build_driver(self):
        driver_name = (self.config.driver or "logging").lower()
        if driver_name == "nvml":
            try:
                return NVMLActuatorDriver(self.config.rigs)
            except Exception as exc:  # pragma: no cover - hardware path
                logger.error("Falling back to logging driver: %s", exc)
        return LoggingActuatorDriver()

    def start(self) -> None:
        self._stop.clear()
        for rig_id, rig in self.config.rigs.items():
            miner_client = create_miner_client(rig.miner_api)
            self._miner_clients[rig_id] = miner_client
            collector = TelemetryCollector(
                rig_id=rig_id,
                sample_interval=self.config.sample_interval_s,
                history_size=self.config.history_size,
                source=rig.source,
                device_index=rig.device_index,
                miner_client=miner_client,
            )
            self.collectors[rig_id] = collector
            collector.start()

            controller = AutoTuneController(rig, self._driver)
            self._controllers[rig_id] = controller

            thread = threading.Thread(
                target=self._run_scoring_loop,
                args=(rig_id,),
                name=f"{rig_id}-scoring",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for collector in self.collectors.values():
            collector.stop()
        while self._threads:
            thread = self._threads.popleft()
            thread.join(timeout=2.0)
        shutdown = getattr(self._driver, "shutdown", None)
        if callable(shutdown):  # pragma: no cover - optional cleanup
            shutdown()

    def set_autotune_enabled(self, enabled: bool) -> None:
        self.autotune_enabled = bool(enabled)

    def pop_action_events(self) -> List[ActionEvent]:
        with self._action_lock:
            events = list(self._action_events)
            self._action_events.clear()
        return events

    def _record_action_event(self, rig_id: str, actuator: str, delta: float) -> None:
        with self._action_lock:
            self._action_events.append(ActionEvent(rig_id=rig_id, actuator=actuator, delta=delta, timestamp=time.time()))

    def _run_scoring_loop(self, rig_id: str) -> None:
        rig_cfg = self.config.rig(rig_id)
        collector = self.collectors[rig_id]
        while not self._stop.is_set():
            window = collector.window(self.config.window_size)
            if not window:
                time.sleep(self.config.sample_interval_s)
                continue
            try:
                score = self.scorer.score_window(window, temp_throttle_c=rig_cfg.temp_throttle_c)
            except Exception as exc:
                logger.exception("Scoring failure for %s: %s", rig_id, exc)
                score = None
            if score:
                self.storage.publish(score)
                controller = self._controllers[rig_id]
                controller.observe(score)

                previous = self._latest_scores.get(rig_id)
                advice = generate_advice(score, rig_cfg, previous)
                if advice:
                    payload = [item.to_dict() for item in advice]
                    self.storage.publish_advice(rig_id, json.dumps(payload, ensure_ascii=False))
                    if self.autotune_enabled:
                        for item in advice:
                            if controller.evaluate(score, item):
                                self.storage.record_action(score.rig_id, item)
                                self._record_action_event(score.rig_id, item.actuator, float(item.delta or 0.0))

                self._latest_scores[rig_id] = score
            time.sleep(self.config.sample_interval_s)


def bootstrap_from_yaml(path: str) -> TelemetryPipeline:
    config = load_config(path)
    return TelemetryPipeline(config)
