"""Logistics feature derivations used in SCORE regression tests."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence


MetricProvider = Callable[[Mapping[str, object]], Mapping[str, float]]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _default_metrics_provider(window: Mapping[str, object]) -> Mapping[str, float]:
    metrics = window.get("metrics") if isinstance(window, Mapping) else {}
    if isinstance(metrics, Mapping) and metrics:
        return {
            "coherence": float(metrics.get("coherence", 0.0) or 0.0),
            "stability": float(metrics.get("stability", 0.0) or 0.0),
            "entropy": float(metrics.get("entropy", 0.0) or 0.0),
            "rupture": float(metrics.get("rupture", metrics.get("lambda_hazard", 0.0) or 0.0)),
            "lambda_hazard": float(metrics.get("lambda_hazard", metrics.get("rupture", 0.0) or 0.0)),
            "coherence_tau_slope": float(metrics.get("coherence_tau_slope", 0.0) or 0.0),
            "domain_wall_slope": float(metrics.get("domain_wall_slope", 0.0) or 0.0),
            "spectral_lowf_share": float(metrics.get("spectral_lowf_share", 0.0) or 0.0),
        }
    components = window.get("components") if isinstance(window, Mapping) else {}
    if isinstance(components, Mapping):
        coherence = float(components.get("coherence", 0.0) or 0.0)
        stability = float(components.get("stability", 0.0) or 0.0)
        entropy = float(components.get("entropy", 0.0) or 0.0)
    else:
        coherence = stability = 0.0
        entropy = 1.0
    hazard = float(window.get("lambda_hazard", window.get("hazard", 0.0)) or 0.0)
    rupture = float(window.get("rupture", hazard) or hazard)
    return {
        "coherence": coherence,
        "stability": stability,
        "entropy": entropy,
        "rupture": rupture,
        "lambda_hazard": hazard,
        "coherence_tau_slope": float(window.get("coherence_tau_slope", 0.0) or 0.0),
        "domain_wall_slope": float(window.get("domain_wall_slope", 0.0) or 0.0),
        "spectral_lowf_share": float(window.get("spectral_lowf_share", 0.0) or 0.0),
    }


def _tokens_by_window(string_scores: Mapping[str, Mapping[str, Iterable[int]]]) -> MutableMapping[int, List[str]]:
    buckets: MutableMapping[int, List[str]] = {}
    for token, payload in string_scores.items():
        window_ids = payload.get("window_ids") if isinstance(payload, Mapping) else []
        if not isinstance(window_ids, Iterable):
            continue
        for window_id in window_ids:
            try:
                idx = int(window_id)
            except (TypeError, ValueError):
                continue
            buckets.setdefault(idx, []).append(token)
    return buckets


def _feature_payload(tokens: Sequence[str]) -> Dict[str, float]:
    if not tokens:
        return {
            "logistics_irreversibility": 0.0,
            "logistics_momentum": 0.0,
            "logistics_cluster_entropy": 0.0,
            "logistics_predicate_balance": 0.5,
            "logistics_predicate_delta": 0.0,
        }

    tokens = [token.lower() for token in tokens]
    total = len(tokens)
    negative = sum(1 for token in tokens if token.startswith("not_"))
    progress = sum(1 for token in tokens if token.split("_", 1)[0] in {"deliver", "load", "unload"})
    affirm = sum(1 for token in tokens if token.startswith("at"))

    irreversibility = _clamp((negative + progress * 0.25 + (1.0 if "deliver" in tokens else 0.0)) / (total + 1.0))
    momentum = _clamp(progress / (total or 1))
    unique = len(set(tokens))
    cluster_entropy = _clamp(unique / (total or 1))
    predicate_balance = _clamp((affirm + 1.0) / (affirm + negative + 2.0))
    predicate_delta = _clamp(abs(affirm - negative) / (total + 1.0))

    return {
        "logistics_irreversibility": irreversibility,
        "logistics_momentum": momentum,
        "logistics_cluster_entropy": cluster_entropy,
        "logistics_predicate_balance": predicate_balance,
        "logistics_predicate_delta": predicate_delta,
    }


def build_logistics_features(
    state: Mapping[str, object],
    *,
    metrics_provider: MetricProvider | None = None,
) -> List[Dict[str, object]]:
    signals = state.get("signals") if isinstance(state, Mapping) else []
    string_scores = state.get("string_scores") if isinstance(state, Mapping) else {}
    if not isinstance(signals, Sequence):
        return []
    if not isinstance(string_scores, Mapping):
        string_scores = {}
    provider = metrics_provider or _default_metrics_provider

    buckets = _tokens_by_window(string_scores)
    results: List[Dict[str, object]] = []
    for index, window in enumerate(signals):
        if not isinstance(window, Mapping):
            continue
        tokens = buckets.get(index, [])
        feat = _feature_payload(tokens)
        metrics = {key: _clamp(value) for key, value in provider(window).items()}
        payload: Dict[str, object] = {
            **feat,
            **metrics,
            "signature": str(window.get("signature", f"window_{index}")),
        }
        results.append(payload)
    return results


def native_metrics_provider(window: Mapping[str, object]) -> Mapping[str, float]:
    """Compat shim used by SCORE CLI tests.

    The real project wires native metrics here; for the research harness we fall
    back to the default provider.
    """

    return _default_metrics_provider(window)


__all__ = ["build_logistics_features", "native_metrics_provider"]
