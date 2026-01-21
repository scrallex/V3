"""Common helpers for parsing and applying metric filters.

These utilities centralise the parsing of simple comparison
expressions such as ``coh>=0.8,ent<=0.35`` and the evaluation of
metric dictionaries against those constraints.  They are shared by the
CLI, proposal engine and similarity workflows so that all entry points
interpret filters consistently.
"""

from __future__ import annotations

import operator
import re
from typing import Any, Dict, Mapping, Tuple, Iterable, List, DefaultDict

import numpy as np

# Canonical metric keys exposed by the manifold analysis.
METRIC_FIELDS = {
    "coherence",
    "stability",
    "entropy",
    "rupture",
    "lambda_hazard",
    "patternability",
    "connector",
}

# Short-hand aliases accepted in filter expressions.
TARGET_ALIASES = {
    "c": "coherence",
    "coh": "coherence",
    "coherence": "coherence",
    "s": "stability",
    "stab": "stability",
    "stability": "stability",
    "e": "entropy",
    "ent": "entropy",
    "entropy": "entropy",
    "r": "rupture",
    "rup": "rupture",
    "rupture": "rupture",
    "lambda": "lambda_hazard",
    "λ": "lambda_hazard",
    "hazard": "lambda_hazard",
    "lambda_hazard": "lambda_hazard",
    "pattern": "patternability",
    "patternability": "patternability",
    "connector": "connector",
}

COMPARE_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


ThresholdValue = float | Tuple[str, float]
FilterSpec = Dict[str, Tuple[str, ThresholdValue]]
MetricVector = Tuple[float, float, float, float, float]


def parse_metric_filter(spec: str | None) -> FilterSpec:
    """Parse a metric filter expression into comparison constraints.

    The expression is a comma-separated list of simple comparisons,
    e.g. ``"coh>=0.8, ent<=0.35"``.  Keys are matched against
    :data:`TARGET_ALIASES`.  The result maps the canonical metric name
    to a tuple of ``(operator_token, threshold)``.
    """

    if not spec:
        return {}
    constraints: FilterSpec = {}
    for raw_clause in spec.split(","):
        clause = raw_clause.strip()
        if not clause:
            continue
        for op_token in (">=", "<=", ">", "<"):
            if op_token in clause:
                left, right = clause.split(op_token, 1)
                key = TARGET_ALIASES.get(left.strip().lower())
                if key is None:
                    raise ValueError(f"Unknown metric in filter: '{left.strip()}'")
                try:
                    value_str = right.strip().upper()
                    value: ThresholdValue
                    match = re.fullmatch(r"P(\d+(?:\.\d+)?)", value_str)
                    if match:
                        percentile = float(match.group(1)) / 100.0
                        value = ("percentile", percentile)
                    else:
                        value = float(value_str)
                except ValueError as exc:  # pragma: no cover - validation guard
                    raise ValueError(f"Invalid numeric value in filter clause '{clause}'") from exc
                constraints[key] = (op_token, value)
                break
        else:  # pragma: no cover - malformed filter
            raise ValueError(f"Could not parse filter clause '{clause}'")
    return constraints


def metric_matches(
    metrics: Mapping[str, float],
    constraints: FilterSpec,
    *,
    quantiles: Mapping[str, Dict[float, float]] | None = None,
) -> bool:
    """Return ``True`` if *metrics* satisfy all *constraints*."""

    if not constraints:
        return True
    for key, (op_token, threshold_spec) in constraints.items():
        comparator = COMPARE_OPS[op_token]
        value = metrics.get(key)
        if value is None:
            return False
        threshold: float
        if isinstance(threshold_spec, tuple) and threshold_spec[0] == "percentile":
            pct = threshold_spec[1]
            if quantiles is None or key not in quantiles or pct not in quantiles[key]:
                return False
            threshold = quantiles[key][pct]
        else:
            threshold = float(threshold_spec)
        if not comparator(float(value), threshold):
            return False
    return True


def normalise_metric_key(key: str) -> str:
    """Map *key* (possibly an alias) to its canonical metric name."""

    canonical = TARGET_ALIASES.get(key.strip().lower())
    if canonical is None:
        raise KeyError(key)
    return canonical


def flatten_metrics(payload: Mapping[str, Any]) -> Dict[str, float]:
    """Return a dictionary containing all canonical metric fields as floats."""

    metrics = dict(payload.get("metrics", {}))
    metrics["coherence"] = float(metrics.get("coherence", payload.get("coherence", 0.0)))
    metrics["stability"] = float(metrics.get("stability", payload.get("stability", 0.0)))
    metrics["entropy"] = float(metrics.get("entropy", payload.get("entropy", 0.0)))
    metrics["rupture"] = float(metrics.get("rupture", payload.get("rupture", 0.0)))
    metrics["lambda_hazard"] = float(
        payload.get(
            "lambda_hazard",
            metrics.get("lambda_hazard", metrics.get("rupture", payload.get("rupture", 0.0))),
        )
    )
    metrics["patternability"] = float(payload.get("patternability", metrics.get("patternability", 0.0)))
    metrics["connector"] = float(payload.get("connector", metrics.get("connector", 0.0)))
    return metrics


def metric_vector(payload: Mapping[str, Any]) -> MetricVector:
    """Return the 5-D vector used by ANN indices (c, s, e, r, λ)."""

    metrics = flatten_metrics(payload)
    return (
        metrics["coherence"],
        metrics["stability"],
        metrics["entropy"],
        metrics["rupture"],
        metrics["lambda_hazard"],
    )


def requested_percentiles(constraints: FilterSpec) -> Dict[str, List[float]]:
    requests: Dict[str, List[float]] = {}
    for metric, (_, value) in constraints.items():
        if isinstance(value, tuple) and value[0] == "percentile":
            requests.setdefault(metric, []).append(value[1])
    return requests


def compute_metric_quantiles(
    metric_values: Mapping[str, Iterable[float]],
    requests: Mapping[str, Iterable[float]],
) -> Dict[str, Dict[float, float]]:
    lookup: Dict[str, Dict[float, float]] = {}
    for metric, percentiles in requests.items():
        values_iter = metric_values.get(metric)
        if values_iter is None:
            continue
        arr = np.asarray([float(v) for v in values_iter if v is not None and not np.isnan(float(v))])
        if arr.size == 0:
            continue
        pct_map: Dict[float, float] = {}
        for pct in sorted(set(percentiles)):
            pct_map[pct] = float(np.nanquantile(arr, pct))
        lookup[metric] = pct_map
    return lookup
