"""Cross-corpus similarity utilities for the Sep Text Manifold."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .filters import (
    MetricVector,
    compute_metric_quantiles,
    flatten_metrics,
    metric_matches,
    metric_vector,
    parse_metric_filter,
    requested_percentiles,
)

try:  # Optional dependency – mirrors index builder behaviour
    import hnswlib  # type: ignore
except ImportError as exc:  # pragma: no cover - optional path exercised in tests
    hnswlib = None  # type: ignore
    _HNSW_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when dependency available
    _HNSW_IMPORT_ERROR = None


@dataclass(frozen=True)
class SimilarityCandidate:
    """Candidate string from the source corpus to project into the target manifold."""

    text: str
    metrics: Dict[str, float]
    occurrences: int
    patternability: float
    connector: float
    window_ids: Sequence[int]
    vector: MetricVector


def load_state(path: Path | str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def select_candidates(
    state: Mapping[str, Any],
    *,
    profile: str | None = None,
    min_connector: float = 0.0,
    min_patternability: float = 0.0,
    min_occurrences: int = 1,
    sort_key: str = "patternability",
    limit: Optional[int] = None,
) -> List[SimilarityCandidate]:
    scores: Mapping[str, Mapping[str, Any]] = state.get("string_scores", {})  # type: ignore[assignment]
    constraints = parse_metric_filter(profile)

    percentile_requests = requested_percentiles(constraints)
    metric_values: Dict[str, List[float]] = {}
    for payload in scores.values():
        metrics = flatten_metrics(payload)
        for key, value in metrics.items():
            metric_values.setdefault(key, []).append(float(value))
    quantiles = compute_metric_quantiles(metric_values, percentile_requests)

    ranked: List[Tuple[float, SimilarityCandidate]] = []
    for text, payload in scores.items():
        occurrences = int(payload.get("occurrences", 0))
        if occurrences < min_occurrences:
            continue
        metrics = flatten_metrics(payload)
        if metrics["patternability"] < min_patternability:
            continue
        if metrics["connector"] < min_connector:
            continue
        if not metric_matches(metrics, constraints, quantiles=quantiles):
            continue
        vector = metric_vector(metrics)
        candidate = SimilarityCandidate(
            text=text,
            metrics=metrics,
            occurrences=occurrences,
            patternability=metrics["patternability"],
            connector=metrics["connector"],
            window_ids=tuple(payload.get("window_ids", ())),
            vector=vector,
        )
        key = metrics.get(sort_key, 0.0)
        ranked.append((float(key), candidate))

    ranked.sort(key=lambda item: item[0], reverse=True)
    candidates = [cand for _, cand in ranked]
    if limit is not None and limit >= 0:
        candidates = candidates[:limit]
    return candidates


def _load_ann_index(index_path: Path, meta_path: Path):
    if hnswlib is None:
        raise ImportError(
            "hnswlib is required for ANN-backed similarity search. Install with 'pip install hnswlib'."
        ) from _HNSW_IMPORT_ERROR
    if not index_path.exists():
        raise FileNotFoundError(f"ANN index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"ANN meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    dim = int(meta.get("dim", 5))
    index = hnswlib.Index(space="l2", dim=dim)
    index.load_index(str(index_path))
    index.set_ef(int(meta.get("ef", 100)))
    return index


def _ann_query(index, vector: MetricVector, k: int) -> List[Tuple[int, float]]:
    data = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    labels, distances = index.knn_query(data, k=k)
    return [(int(lbl), float(dist)) for lbl, dist in zip(labels[0], distances[0])]


def _fallback_query(
    signals: Sequence[Mapping[str, Any]],
    vector: MetricVector,
    *,
    k: int,
) -> List[Tuple[int, float]]:
    if not signals:
        return []
    target = np.asarray(vector, dtype=np.float32)
    distances: List[Tuple[int, float]] = []
    for sig in signals:
        candidate_vec = np.asarray(metric_vector(sig), dtype=np.float32)
        dist = float(np.linalg.norm(candidate_vec - target))
        distances.append((int(sig.get("id", 0)), dist))
    distances.sort(key=lambda item: item[1])
    return distances[:k]


def cross_corpus_similarity(
    source_state: Mapping[str, Any],
    *,
    profile: str | None = None,
    min_connector: float = 0.0,
    min_patternability: float = 0.0,
    min_occurrences: int = 1,
    sort_key: str = "patternability",
    limit: Optional[int] = None,
    ann_index_path: Optional[Path] = None,
    ann_meta_path: Optional[Path] = None,
    target_state: Optional[Mapping[str, Any]] = None,
    k: int = 50,
    max_distance: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute similarity hits from a source manifold into a target manifold."""

    candidates = select_candidates(
        source_state,
        profile=profile,
        min_connector=min_connector,
        min_patternability=min_patternability,
        min_occurrences=min_occurrences,
        sort_key=sort_key,
        limit=limit,
    )

    signals = list(target_state.get("signals", [])) if target_state else []
    signal_lookup = {int(sig.get("id", idx)): sig for idx, sig in enumerate(signals)}

    use_ann = ann_index_path is not None and ann_meta_path is not None
    ann = None
    if use_ann:
        try:
            ann = _load_ann_index(ann_index_path, ann_meta_path)
        except Exception:
            ann = None
            if not signals:
                raise

    results: List[Dict[str, Any]] = []
    for candidate in candidates:
        if ann is not None:
            neighbours = _ann_query(ann, candidate.vector, k=k)
        else:
            if not signals:
                raise ValueError("Target state with signals is required when ANN index is unavailable.")
            neighbours = _fallback_query(signals, candidate.vector, k=k)
        matches: List[Dict[str, Any]] = []
        for window_id, distance in neighbours:
            if max_distance is not None and distance > max_distance:
                continue
            payload: Dict[str, Any] = {"window_id": window_id, "distance": distance}
            sig = signal_lookup.get(window_id)
            if sig is not None:
                payload["metrics"] = flatten_metrics(sig)
                payload["signature"] = sig.get("signature")
                payload["window_start"] = sig.get("window_start")
                payload["window_end"] = sig.get("window_end")
            matches.append(payload)
        results.append(
            {
                "string": candidate.text,
                "occurrences": candidate.occurrences,
                "patternability": candidate.patternability,
                "connector": candidate.connector,
                "metrics": candidate.metrics,
                "window_ids": list(candidate.window_ids),
                "matches": matches,
            }
        )

    return {
        "candidate_count": len(candidates),
        "results": results,
    }
