"""Utilities for augmenting manifold windows with derived features."""

from __future__ import annotations

from typing import Mapping, MutableMapping

from scripts.features import CausalFeatureExtractor


def _ensure_feature_map(window: MutableMapping[str, object]) -> MutableMapping[str, object]:
    features = window.get("features")
    if not isinstance(features, MutableMapping):
        features = {}
        window["features"] = features
    return features


def enrich_with_causal_features(state: Mapping[str, object], *, extractor: CausalFeatureExtractor | None = None) -> int:
    """Populate each signal entry with causal features.

    Returns the number of signals processed. The function is deliberately
    tolerant of loosely shaped state dictionaries – it only touches windows that
    behave like mappings.
    """

    signals = state.get("signals") if isinstance(state, Mapping) else None
    if not isinstance(signals, list):
        return 0
    extractor = extractor or CausalFeatureExtractor()
    history: list[Mapping[str, object]] = []
    processed = 0
    for window in signals:
        if not isinstance(window, MutableMapping):
            continue
        causal = extractor.extract(window, history=history)
        feature_map = _ensure_feature_map(window)
        feature_map["causal"] = causal
        history.append({"metrics": window.get("metrics", {}), "state": window.get("state", {})})
        processed += 1
    return processed


__all__ = ["enrich_with_causal_features"]
