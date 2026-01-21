"""Causal feature heuristics used by SCORE compatibility tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _resource_load(state: Mapping[str, object]) -> float:
    resources = state.get("resources") if isinstance(state, Mapping) else {}
    total = 0.0
    if isinstance(resources, Mapping):
        for payload in resources.values():
            if isinstance(payload, Mapping):
                locked = payload.get("locked")
                if isinstance(locked, (int, float)):
                    total += float(locked)
    return total


@dataclass
class CausalFeatureExtractor:
    """Derive coarse causal features from STM windows.

    The implementation is intentionally lightweight – it mirrors the behaviour
    needed by the SCORE regression tests without pulling in the full production
    causal stack.
    """

    def extract(self, window: Mapping[str, object], *, history: Iterable[Mapping[str, object]] = ()) -> Dict[str, float]:
        metrics = window.get("metrics") if isinstance(window, Mapping) else {}
        dilution = window.get("dilution") if isinstance(window, Mapping) else {}
        state = window.get("state") if isinstance(window, Mapping) else {}

        coherence = _clamp(_read_metric(metrics, "coherence"))
        entropy = _clamp(_read_metric(metrics, "entropy"))
        stability = _clamp(_read_metric(metrics, "stability"))

        path_dilution = _clamp(_read_metric(dilution, "path"))
        signal_dilution = _clamp(_read_metric(dilution, "signal"))

        history = list(history)
        if history:
            hist_metrics = [entry.get("metrics", {}) for entry in history if isinstance(entry, Mapping)]
            avg_coherence = sum(_read_metric(m, "coherence") for m in hist_metrics) / max(1, len(hist_metrics))
            avg_entropy = sum(_read_metric(m, "entropy") for m in hist_metrics) / max(1, len(hist_metrics))
        else:
            avg_coherence = coherence
            avg_entropy = entropy

        current_load = _resource_load(state if isinstance(state, Mapping) else {})
        historical_load = sum(_resource_load(entry.get("state", {})) for entry in history if isinstance(entry, Mapping))
        resource_commitment_ratio = _clamp(current_load / max(1.0, current_load + historical_load))

        features = {
            "irreversible_actions": _clamp(path_dilution * 0.6 + entropy * 0.4),
            "resource_commitment_ratio": resource_commitment_ratio,
            "decision_reversibility": _clamp(1.0 - (resource_commitment_ratio * 0.5 + path_dilution * 0.5)),
            "unsatisfied_preconditions": _clamp(entropy * (1.0 - stability)),
            "effect_cascade_depth": _clamp(signal_dilution * 0.5 + path_dilution * 0.5),
            "constraint_violation_distance": _clamp(abs(coherence - stability)),
            "action_velocity": _clamp((coherence + stability) / 2.0),
            "state_divergence_rate": _clamp(abs(coherence - avg_coherence)),
            "pattern_break_score": _clamp(abs(entropy - avg_entropy)),
        }
        return features


def _read_metric(payload: Mapping[str, object], key: str) -> float:
    if not isinstance(payload, Mapping):
        return 0.0
    value = payload.get(key, 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return 0.0
