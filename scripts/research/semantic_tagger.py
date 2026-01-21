"""Semantic tagging utilities for QFH manifold signals."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, List, Mapping, MutableMapping, Optional

Number = float | int


@dataclass(frozen=True)
class SemanticThresholds:
    """Threshold defaults for deriving semantic regime tags."""

    high_coherence: float = 0.7
    high_stability: float = 0.7
    high_rupture: float = 0.4
    high_entropy: float = 0.9
    low_hazard: float = 0.1
    lambda_slope_improving: float = 0.0
    coherence_delta_positive: float = 0.0


def _get_numeric(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    components = payload.get("components")
    if isinstance(components, Mapping):
        value = components.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _apply_overrides(
    thresholds: SemanticThresholds, overrides: Mapping[str, float] | None
) -> SemanticThresholds:
    if not overrides:
        return thresholds
    valid = {field.name for field in dataclasses.fields(SemanticThresholds)}
    filtered: MutableMapping[str, float] = {}
    for key, value in overrides.items():
        if key in valid:
            filtered[key] = float(value)
    if not filtered:
        return thresholds
    return dataclasses.replace(thresholds, **filtered)


def generate_semantic_tags(
    qfh_signal: Mapping[str, Any],
    *,
    thresholds: SemanticThresholds | None = None,
    overrides: Mapping[str, float] | None = None,
) -> List[str]:
    """Return semantic regime tags derived from a QFH gate payload."""

    params = _apply_overrides(thresholds or SemanticThresholds(), overrides)

    coherence = _get_numeric(qfh_signal, "coherence")
    stability = _get_numeric(qfh_signal, "stability")
    rupture = _get_numeric(qfh_signal, "rupture")
    entropy = _get_numeric(qfh_signal, "entropy")
    hazard = _get_numeric(qfh_signal, "hazard")
    lambda_val = _get_numeric(qfh_signal, "lambda")
    lambda_slope = _get_numeric(qfh_signal, "lambda_slope")
    coherence_delta = _get_numeric(qfh_signal, "coherence_delta")

    tags: List[str] = []

    if coherence is not None and stability is not None:
        if coherence >= params.high_coherence and stability >= params.high_stability:
            tags.append("highly_stable")
        if coherence_delta is not None and coherence_delta > params.coherence_delta_positive:
            tags.append("strengthening_structure")

    if rupture is not None and rupture >= params.high_rupture:
        tags.append("high_rupture_event")

    target_entropy = entropy if entropy is not None else None
    if target_entropy is None and coherence is not None:
        # entropy not present in some payloads; approximate via coherence distance.
        target_entropy = max(0.0, 1.0 - coherence)
    if target_entropy is not None and target_entropy >= params.high_entropy:
        tags.append("chaotic_price_action")

    hazard_candidate: Optional[float] = None
    if hazard is not None:
        hazard_candidate = hazard
    elif lambda_val is not None:
        hazard_candidate = lambda_val
    if hazard_candidate is not None and hazard_candidate <= params.low_hazard:
        tags.append("low_hazard_environment")

    if lambda_slope is not None and lambda_slope < params.lambda_slope_improving:
        tags.append("improving_stability")

    return tags


__all__ = ["SemanticThresholds", "generate_semantic_tags"]
