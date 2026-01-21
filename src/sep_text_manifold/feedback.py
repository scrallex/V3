"""Feedback helpers for leveraging STM structural twins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from .filters import flatten_metrics, metric_vector


@dataclass(frozen=True)
class TwinSuggestion:
    window_id: int
    signature: Optional[str]
    distance: float
    metrics: Dict[str, float]
    tokens: Sequence[str]


def _ensure_metrics(payload: Mapping[str, Any]) -> Dict[str, float]:
    metrics = flatten_metrics(payload)
    minimal = {key: float(metrics.get(key, 0.0)) for key in ("coherence", "stability", "entropy", "rupture", "lambda_hazard")}
    return minimal


def _derive_target_payload(
    invalid_action: Mapping[str, Any],
    manifold_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    if "metrics" in invalid_action or any(key in invalid_action for key in ("coherence", "stability", "entropy", "rupture")):
        return invalid_action
    window_id = invalid_action.get("window_id")
    if window_id is not None:
        for sig in manifold_state.get("signals", []):
            if int(sig.get("id", sig.get("index", -1))) == int(window_id):
                return sig
    token = invalid_action.get("token")
    if token:
        scores = manifold_state.get("string_scores", {})
        payload = scores.get(token)
        if payload is not None:
            return payload
    signature = invalid_action.get("signature")
    if signature:
        for sig in manifold_state.get("signals", []):
            if sig.get("signature") == signature:
                return sig
    raise ValueError("Unable to derive metrics for invalid action; supply metrics, window_id, token, or signature")


def suggest_twin_action(
    invalid_action: Mapping[str, Any],
    manifold_state: Mapping[str, Any],
    *,
    top_k: int = 3,
    max_distance: Optional[float] = None,
    match_signature: bool = False,
    top_tokens: int = 5,
) -> List[TwinSuggestion]:
    """Return structural twin candidates from the manifold for an invalid step.

    Parameters
    ----------
    invalid_action:
        Mapping describing the invalid transition.  It should include at
        least one of: ``metrics`` dict, ``window_id``, ``token`` or
        ``signature``.
    manifold_state:
        STM state containing ``signals`` and optionally ``string_scores``.
    top_k:
        Maximum number of twin suggestions to return.
    max_distance:
        Optional Euclidean distance threshold applied to the 5-D metric
        vector (c, s, e, r, λ).
    match_signature:
        When ``True``, restrict candidates to those sharing the same
        signature as the invalid action.
    top_tokens:
        Number of high-frequency tokens to include per twin suggestion.
    """
    signals: Sequence[Mapping[str, Any]] = manifold_state.get("signals", [])  # type: ignore[assignment]
    if not signals:
        return []

    target_payload = _derive_target_payload(invalid_action, manifold_state)
    target_vector = np.asarray(metric_vector(target_payload), dtype=np.float64)

    target_signature = invalid_action.get("signature") or target_payload.get("signature")

    string_scores: Mapping[str, Mapping[str, Any]] = manifold_state.get("string_scores", {})  # type: ignore[assignment]
    window_tokens: MutableMapping[int, List[str]] = {}
    for token, payload in string_scores.items():
        for window_id in payload.get("window_ids", []):
            try:
                wid = int(window_id)
            except (TypeError, ValueError):
                continue
            window_tokens.setdefault(wid, []).append(token)

    suggestions: List[TwinSuggestion] = []
    for sig in signals:
        window_id = int(sig.get("id", sig.get("index", 0)))
        signature = sig.get("signature")
        if match_signature and target_signature is not None and signature != target_signature:
            continue
        candidate_vector = np.asarray(metric_vector(sig), dtype=np.float64)
        distance = float(np.linalg.norm(target_vector - candidate_vector))
        if max_distance is not None and distance > max_distance:
            continue
        metrics = _ensure_metrics(sig)
        tokens = sorted(window_tokens.get(window_id, []))[:top_tokens]
        suggestions.append(
            TwinSuggestion(
                window_id=window_id,
                signature=signature if isinstance(signature, str) else None,
                distance=distance,
                metrics=metrics,
                tokens=tokens,
            )
        )

    suggestions.sort(key=lambda cand: cand.distance)
    if top_k >= 0:
        suggestions = suggestions[:top_k]
    return suggestions
