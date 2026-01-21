"""Lightweight guardrail helpers for research backtests.

The SCORE backtest suite expects a ``scripts.trading.guards`` module providing a
``PathMetrics`` container and ``throttle_factor`` sizing heuristic. The live SEP
stack keeps richer logic elsewhere; for the embedded research environment we
expose a simplified version that captures the intent (scale positions based on
structure while penalising hazard spikes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PathMetrics:
    """Encapsulate structural metrics for a signal path."""

    entropy: float
    coherence: float
    stability: float
    rupture: float
    hazard: float

    def clamp(self) -> "PathMetrics":
        """Return a copy with all metrics clipped to [0, 1] for safety."""

        def _clip(value: float) -> float:
            if value < 0.0:
                return 0.0
            if value > 1.0:
                return 1.0
            return value

        return PathMetrics(
            entropy=_clip(self.entropy),
            coherence=_clip(self.coherence),
            stability=_clip(self.stability),
            rupture=_clip(self.rupture),
            hazard=_clip(self.hazard),
        )


def throttle_factor(current: PathMetrics, previous: Optional[PathMetrics] = None) -> float:
    """Return a fractional position size between 0 and 1.

    The heuristic favours stable/coherent structure while reducing exposure as
    hazard rises. A slight bonus is applied when structure improves versus the
    previous observation so the backtester still reacts to strengthening tapes.
    """

    metrics = current.clamp()
    base_strength = (metrics.coherence + metrics.stability) / 2.0
    hazard_discount = 1.0 - metrics.hazard
    rupture_penalty = 1.0 - min(1.0, metrics.rupture * 1.2)

    bonus = 0.0
    if previous is not None:
        prev = previous.clamp()
        delta = (metrics.coherence - prev.coherence) + (metrics.stability - prev.stability)
        if delta > 0:
            bonus = min(0.15, delta * 0.5)

    raw = base_strength * 0.6 + hazard_discount * 0.3 + rupture_penalty * 0.1 + bonus
    return max(0.0, min(1.0, raw))


__all__ = ["PathMetrics", "throttle_factor"]
