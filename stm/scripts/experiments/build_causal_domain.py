"""Lightweight causal blending used by SCORE compatibility tests."""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping


def blend_metrics(metrics: Mapping[str, float], *, history: Iterable[Mapping[str, float]] = ()) -> MutableMapping[str, float]:
    """Return a smoothed version of the supplied metrics.

    The real project applies richer temporal models; here we compute a simple
    average across the current window and any historical records.
    """

    accumulator: MutableMapping[str, float] = {key: float(value) for key, value in metrics.items()}
    history = list(history)
    if not history:
        return accumulator
    counts: MutableMapping[str, int] = {key: 1 for key in accumulator}
    for entry in history:
        for key, value in entry.items():
            accumulator[key] = accumulator.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    for key, total in accumulator.items():
        accumulator[key] = total / max(1, counts.get(key, 1))
    return accumulator


__all__ = ["blend_metrics"]
