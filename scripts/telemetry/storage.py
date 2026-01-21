"""Persistence and publishing helpers for telemetry scores."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .advisory import AdvisoryAction
from .scoring import TelemetryScore

try:  # pragma: no cover - optional dependency
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None


class TelemetryStorage:
    """Publish telemetry scores to Valkey and append them to NDJSON logs."""

    def __init__(self, *, redis_url: Optional[str], ndjson_root: Path) -> None:
        self._client = None
        if redis and redis_url:
            try:
                self._client = redis.from_url(redis_url)
            except Exception:
                self._client = None
        self.ndjson_root = ndjson_root
        self.ndjson_root.mkdir(parents=True, exist_ok=True)

    def publish(self, score: TelemetryScore) -> None:
        key = f"telemetry:last:{score.rig_id}"
        payload = score.to_json()
        if self._client:
            try:
                self._client.set(key, payload)
            except Exception:
                pass
        advice_key = f"telemetry:score:{score.rig_id}"
        if self._client:
            try:
                self._client.publish(advice_key, payload)
            except Exception:
                pass
        self._append_ndjson(score.rig_id, payload)

    def _append_ndjson(self, rig_id: str, payload: str) -> None:
        path = self.ndjson_root / f"{rig_id.lower()}.ndjson"
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
        except Exception:
            pass

    def publish_advice(self, rig_id: str, advice_payload: str) -> None:
        key = f"telemetry:advice:{rig_id}"
        if self._client:
            try:
                self._client.set(key, advice_payload)
            except Exception:
                pass
        path = self.ndjson_root / f"{rig_id.lower()}_advice.ndjson"
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(advice_payload)
                handle.write("\n")
        except Exception:
            pass

    def record_action(self, rig_id: str, action: AdvisoryAction) -> None:
        payload = {
            "rig_id": rig_id,
            "timestamp": time.time(),
            "actuator": action.actuator,
            "delta": action.delta,
            "reason": action.reason,
            "ttl_seconds": action.ttl_seconds,
        }
        encoded = json.dumps(payload, ensure_ascii=False)
        key = f"telemetry:actions:{rig_id}"
        if self._client:
            try:
                self._client.rpush(key, encoded)
            except Exception:
                pass
        path = self.ndjson_root / f"{rig_id.lower()}_actions.ndjson"
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.write("\n")
        except Exception:
            pass
