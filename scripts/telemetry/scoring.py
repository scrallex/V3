"""Scoring helpers leveraging the native QFH bindings."""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .bit_encoder import TelemetryBitEncoder
from .snapshot import TelemetrySnapshot, rolling_variance

# Ensure the SCORE package is importable when running from repo root
_ROOT = Path(__file__).resolve().parents[2]
_SCORE_SRC = _ROOT / "score" / "src"
if _SCORE_SRC.is_dir() and str(_SCORE_SRC) not in sys.path:
    sys.path.insert(0, str(_SCORE_SRC))

try:
    from sep_text_manifold import native as stm_native  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    stm_native = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(slots=True)
class TelemetryScore:
    rig_id: str
    timestamp_ns: int
    coherence: float
    stability: float
    entropy: float
    rupture: float
    lambda_hazard: float
    sep_score: float
    rupture_flag: str
    efficiency: float
    reject_rate: float
    thermal_headroom: float
    stability_index: float
    latency_budget_ms: float

    def to_json(self) -> str:
        payload = {
            "rig_id": self.rig_id,
            "timestamp_ns": self.timestamp_ns,
            "metrics": {
                "coherence": self.coherence,
                "stability": self.stability,
                "entropy": self.entropy,
                "rupture": self.rupture,
                "lambda_hazard": self.lambda_hazard,
                "sep_score": self.sep_score,
            },
            "kpis": {
                "efficiency": self.efficiency,
                "reject_rate": self.reject_rate,
                "thermal_headroom": self.thermal_headroom,
                "stability_index": self.stability_index,
                "latency_budget_ms": self.latency_budget_ms,
            },
            "rupture_flag": self.rupture_flag,
        }
        return json.dumps(payload, ensure_ascii=False)


class QFHScorer:
    """Wrap the native QFH analyzer with telemetry-friendly helpers."""

    def __init__(self, encoder: TelemetryBitEncoder, *, sep_score_floor: float = 50.0) -> None:
        if stm_native is None:
            raise RuntimeError(
                "sep_text_manifold.native is unavailable. Build the sep_quantum extension to enable scoring."
            ) from _IMPORT_ERROR
        self.encoder = encoder
        self.sep_score_floor = sep_score_floor

    def score_window(self, window: Sequence[TelemetrySnapshot], *, temp_throttle_c: float) -> Optional[TelemetryScore]:
        if len(window) < max(8, self.encoder.lookback // 2):
            return None

        bits = self.encoder.encode(window)
        if not bits:
            return None

        result = stm_native.analyze_window(bits)
        coherence = float(getattr(result, "coherence", 0.0))
        rupture_ratio = float(getattr(result, "rupture_ratio", getattr(result, "rupture", 0.0)))
        entropy = float(getattr(result, "entropy", 0.0))
        stability = 1.0 - rupture_ratio
        lambda_hazard = float(getattr(result, "lambda_hazard", rupture_ratio))

        current = window[-1]
        eff = current.efficiency()
        recent_eff = list(self.encoder.recent_efficiency(window))
        eff_norm = self._normalise(eff, recent_eff)

        hashrate_var = rolling_variance(self.encoder.recent_hashrate(window))
        stability_index = current.stability_index(hashrate_var)

        thermal_headroom = current.thermal_headroom(temp_throttle_c)
        latency_budget = current.latency_budget()

        sep_score = self._combine_sep_score(
            coherence,
            stability,
            entropy,
            rupture_ratio,
            eff_norm,
            stability_index,
            thermal_headroom,
            latency_budget,
        )

        rupture_flag = self._classify_rupture(lambda_hazard, rupture_ratio)

        return TelemetryScore(
            rig_id=current.rig_id,
            timestamp_ns=int(current.timestamp.timestamp() * 1_000_000_000),
            coherence=coherence,
            stability=stability,
            entropy=entropy,
            rupture=rupture_ratio,
            lambda_hazard=lambda_hazard,
            sep_score=sep_score,
            rupture_flag=rupture_flag,
            efficiency=eff,
            reject_rate=current.reject_rate(),
            thermal_headroom=thermal_headroom,
            stability_index=stability_index,
            latency_budget_ms=latency_budget,
        )

    def _normalise(self, value: float, history: Iterable[float]) -> float:
        samples = [v for v in history if math.isfinite(v)]
        if not samples:
            return 0.5
        median = statistics.median(samples)
        mad = statistics.median([abs(v - median) for v in samples]) or 1e-6
        z = max(-4.0, min(4.0, (value - median) / (3 * mad)))
        return 0.5 + (z / 8.0)

    def _combine_sep_score(
        self,
        coherence: float,
        stability: float,
        entropy: float,
        rupture: float,
        eff_norm: float,
        stability_index: float,
        thermal_headroom: float,
        latency_budget: float,
    ) -> float:
        thermal_term = min(1.0, thermal_headroom / 20.0)
        latency_term = min(1.0, latency_budget / 200.0)

        base = (
            0.35 * coherence
            + 0.25 * stability
            + 0.15 * eff_norm
            + 0.10 * stability_index
            + 0.08 * thermal_term
            + 0.07 * latency_term
            - 0.20 * entropy
            - 0.20 * rupture
        )
        score = 100.0 * max(0.0, base)
        return max(self.sep_score_floor, min(100.0, score))

    def _classify_rupture(self, lambda_hazard: float, rupture: float) -> str:
        if lambda_hazard >= 0.75 or rupture >= 0.6:
            return "macro"
        if lambda_hazard >= 0.5 or rupture >= 0.4:
            return "meso"
        if lambda_hazard >= 0.3 or rupture >= 0.2:
            return "micro"
        return "none"
