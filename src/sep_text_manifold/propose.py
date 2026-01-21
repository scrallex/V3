"""Bridge-string proposal utilities for the Sep Text Manifold."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .filters import (
    compute_metric_quantiles,
    metric_matches,
    parse_metric_filter,
    requested_percentiles,
)

METRIC_KEYS = ("coherence", "stability", "entropy", "rupture")


@dataclass(frozen=True)
class Candidate:
    """Normalised data about a candidate string."""

    text: str
    metrics: Dict[str, float]
    occurrences: int
    patternability: float
    connector: float
    window_ids: Sequence[int]
    themes: Sequence[int]

    @property
    def vector(self) -> Tuple[float, float, float, float]:
        return tuple(self.metrics.get(key, 0.0) for key in METRIC_KEYS)

    @property
    def alpha_ratio(self) -> float:
        if not self.text:
            return 0.0
        alpha = sum(1 for ch in self.text if ch.isalpha())
        return alpha / len(self.text)

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class Proposal:
    string: str
    score: float
    diagnostics: Dict[str, float]
    candidate: Candidate


def load_state(path: Path | str) -> Dict[str, Any]:
    """Load a saved manifold analysis state."""
    return json.loads(Path(path).read_text())


def build_candidates(state: Mapping[str, Any]) -> Tuple[List[Candidate], Dict[str, Candidate]]:
    """Transform ``string_scores`` into ``Candidate`` objects."""
    string_scores: Mapping[str, Mapping[str, Any]] = state.get("string_scores", {})  # type: ignore[assignment]
    themes: List[Iterable[str]] = state.get("themes", [])  # type: ignore[assignment]
    theme_lookup: Dict[str, List[int]] = {}
    for idx, members in enumerate(themes):
        for s in members:
            theme_lookup.setdefault(s, []).append(idx)
    candidates: List[Candidate] = []
    index: Dict[str, Candidate] = {}
    for text, payload in string_scores.items():
        profile = payload.get("metrics") or {}
        metrics = {key: float(profile.get(key, payload.get(key, 0.0))) for key in METRIC_KEYS}
        lambda_value = float(
            profile.get(
                "lambda_hazard",
                payload.get("lambda_hazard", profile.get("rupture", payload.get("rupture", metrics.get("rupture", 0.0)))),
            )
        )
        metrics["lambda_hazard"] = lambda_value
        candidate = Candidate(
            text=text,
            metrics=metrics,
            occurrences=int(payload.get("occurrences", 0)),
            patternability=float(payload.get("patternability", 0.0)),
            connector=float(payload.get("connector", 0.0)),
            window_ids=tuple(payload.get("window_ids", ())),
            themes=tuple(theme_lookup.get(text, ())),
        )
        candidates.append(candidate)
        index[text] = candidate
    return candidates, index


def centroid(vectors: Iterable[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    vectors = list(vectors)
    if not vectors:
        return (0.0, 0.0, 0.0, 0.0)
    sums = [0.0, 0.0, 0.0, 0.0]
    for vec in vectors:
        for i, value in enumerate(vec):
            sums[i] += value
    count = float(len(vectors))
    return tuple(value / count for value in sums)  # type: ignore[return-value]


def l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def novelty_score(candidate: Candidate) -> float:
    """Simple novelty heuristic favouring lower-frequency strings."""
    return 1.0 / (1.0 + math.log(candidate.occurrences + 1.0))


def compute_score(candidate: Candidate, target: Tuple[float, ...]) -> Tuple[float, Dict[str, float]]:
    vec = candidate.vector
    distance = l2_distance(vec, target)
    pattern_score = 1.0 / (1.0 + distance * 3.0)
    connector = candidate.connector
    novelty = novelty_score(candidate)
    total = 0.55 * pattern_score + 0.30 * connector + 0.15 * novelty
    diagnostics = {
        "distance": distance,
        "pattern_component": pattern_score,
        "connector": connector,
        "novelty": novelty,
    }
    return total, diagnostics


def propose_from_state(
    state: Mapping[str, Any],
    *,
    seeds: Sequence[str],
    k: int = 25,
    min_connector: float = 0.0,
    min_patternability: float = 0.0,
    target_profile: Optional[str] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Generate bridge-string proposals from a loaded state dict."""
    candidates, index = build_candidates(state)
    seed_objects = [index[s] for s in seeds if s in index]
    if not seed_objects:
        raise ValueError("None of the provided seeds are present in the analysis state")
    seed_vectors = [seed.vector for seed in seed_objects]
    target = centroid(seed_vectors)
    profile_constraints = parse_metric_filter(target_profile)
    percentile_requests = requested_percentiles(profile_constraints)
    metric_values: Dict[str, List[float]] = {}
    for cand in candidates:
        for key, value in cand.metrics.items():
            metric_values.setdefault(key, []).append(float(value))
    quantiles = compute_metric_quantiles(metric_values, percentile_requests)
    exclude_set = set(exclude or ()) | set(seeds)
    proposals: List[Proposal] = []
    for candidate in candidates:
        if candidate.text in exclude_set:
            continue
        if candidate.connector < min_connector:
            continue
        if candidate.patternability < min_patternability:
            continue
        if not metric_matches(candidate.metrics, profile_constraints, quantiles=quantiles):
            continue
        score, diagnostics = compute_score(candidate, target)
        diagnostics.update({key: candidate.metrics.get(key, 0.0) for key in METRIC_KEYS})
        diagnostics["occurrences"] = candidate.occurrences
        diagnostics["patternability"] = candidate.patternability
        proposals.append(Proposal(candidate.text, score, diagnostics, candidate))
    proposals.sort(key=lambda item: item.score, reverse=True)
    trimmed = proposals[:k]
    return {
        "target": {
            "seeds": list(seeds),
            "vector": target,
            "constraints": profile_constraints,
        },
        "proposals": [
            {
                "string": item.string,
                "score": item.score,
                "metrics": {key: item.candidate.metrics.get(key, 0.0) for key in METRIC_KEYS},
                "patternability": item.candidate.patternability,
                "connector": item.candidate.connector,
                "occurrences": item.candidate.occurrences,
                "themes": list(item.candidate.themes),
                "diagnostics": item.diagnostics,
            }
            for item in trimmed
        ],
    }


def propose(
    state_path: Path | str,
    *,
    seeds: Sequence[str],
    k: int = 25,
    min_connector: float = 0.0,
    min_patternability: float = 0.0,
    target_profile: Optional[str] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper that loads state then delegates to ``propose_from_state``."""
    state = load_state(state_path)
    return propose_from_state(
        state,
        seeds=seeds,
        k=k,
        min_connector=min_connector,
        min_patternability=min_patternability,
        target_profile=target_profile,
        exclude=exclude,
    )
