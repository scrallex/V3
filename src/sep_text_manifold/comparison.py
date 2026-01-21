"""Comparison utilities between STM feedback and VAL signals."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable, List, Mapping, MutableSet, Sequence, Tuple, Union

NumberLike = Union[int, float]
EventLike = Union[NumberLike, Mapping[str, NumberLike]]


def _extract_index(event: EventLike) -> int:
    if isinstance(event, Mapping):
        for key in ("index", "step", "window_id", "window", "position"):
            if key in event:
                return int(event[key])
        raise KeyError("Event mapping missing recognised index key")
    return int(event)


def _normalise_events(events: Iterable[EventLike]) -> List[int]:
    return sorted(_extract_index(event) for event in events)


@dataclass(frozen=True)
class AlignmentResult:
    agreement: int
    stm_only: int
    val_only: int
    matched_pairs: List[Tuple[int, int]]
    precision: float
    recall: float
    f1: float


def stm_val_alignment(
    stm_feedback: Iterable[EventLike],
    val_feedback: Iterable[EventLike],
    *,
    tolerance: int = 0,
) -> AlignmentResult:
    """Measure how STM alerts align with VAL feedback events.

    Parameters
    ----------
    stm_feedback:
        Iterable of STM alert indices (ints) or mappings containing an
        ``index``/``step``/``window_id`` field.
    val_feedback:
        Iterable of VAL feedback events in the same format as
        ``stm_feedback``.
    tolerance:
        Maximum absolute difference between indices for the events to be
        treated as matching.  Defaults to ``0`` (exact match).
    """
    stm_indices = _normalise_events(stm_feedback)
    val_indices = _normalise_events(val_feedback)

    matched_pairs: List[Tuple[int, int]] = []
    matched_stm: MutableSet[int] = set()
    matched_val: MutableSet[int] = set()

    for val_idx in val_indices:
        for stm_idx in stm_indices:
            if stm_idx in matched_stm:
                continue
            if abs(stm_idx - val_idx) <= tolerance:
                matched_pairs.append((stm_idx, val_idx))
                matched_stm.add(stm_idx)
                matched_val.add(val_idx)
                break

    agreement = len(matched_pairs)
    stm_only = len(stm_indices) - len(matched_stm)
    val_only = len(val_indices) - len(matched_val)

    precision = agreement / len(stm_indices) if stm_indices else 0.0
    recall = agreement / len(val_indices) if val_indices else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return AlignmentResult(
        agreement=agreement,
        stm_only=stm_only,
        val_only=val_only,
        matched_pairs=matched_pairs,
        precision=precision,
        recall=recall,
        f1=f1,
    )


@dataclass(frozen=True)
class LeadTimeResult:
    coverage: float
    leads: List[int]
    mean: float
    median: float
    maximum: int
    minimum: int


def detection_lead_time(
    stm_alerts: Iterable[EventLike],
    val_failures: Iterable[EventLike],
) -> LeadTimeResult:
    """Compute the lead time (in steps) between STM alerts and VAL failures.

    Positive lead numbers indicate STM alerted before the corresponding
    VAL failure.  Negative numbers indicate STM lagged behind VAL.
    """
    stm_indices = _normalise_events(stm_alerts)
    val_indices = _normalise_events(val_failures)
    if not val_indices:
        return LeadTimeResult(coverage=0.0, leads=[], mean=0.0, median=0.0, maximum=0, minimum=0)

    leads: List[int] = []
    for val_idx in val_indices:
        prior = [idx for idx in stm_indices if idx <= val_idx]
        if prior:
            candidate = max(prior)
            leads.append(val_idx - candidate)
            continue
        future = [idx for idx in stm_indices if idx > val_idx]
        if future:
            candidate = min(future)
            leads.append(val_idx - candidate)

    if leads:
        cov = len(leads) / len(val_indices)
        return LeadTimeResult(
            coverage=cov,
            leads=leads,
            mean=mean(leads),
            median=median(leads),
            maximum=max(leads),
            minimum=min(leads),
        )
    return LeadTimeResult(coverage=0.0, leads=[], mean=0.0, median=0.0, maximum=0, minimum=0)
