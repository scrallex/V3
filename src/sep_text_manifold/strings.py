"""
String extraction and aggregation for the Sep Text Manifold.

This module provides utilities to extract tokens from text and to
aggregate quantum metrics over the occurrences of each string.  It
aligns each token back onto the byte windows produced by the manifold
builder and computes mean values of coherence, stability, entropy and
rupture for each string.

These functions are intentionally simple and can be refined.  For
example, you may wish to support n‑grams, phrase detection or more
sophisticated tokenisation, and to compute additional statistics
(standard deviation, 95th percentiles, etc.).
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class StringOccurrence:
    """Represents a single occurrence of a string in a document."""
    string: str
    file_id: str
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int


def extract_strings(text: str, file_id: str) -> List[StringOccurrence]:
    """Extract simple word tokens from *text*.

    This function uses a regular expression to find sequences of
    alphanumeric characters.  It returns a list of ``StringOccurrence``
    objects with both character and byte offsets.  Byte offsets are
    calculated relative to the UTF‑8 encoding of the entire text.
    """
    occurrences: List[StringOccurrence] = []
    # Precompute cumulative byte lengths for each character index to
    # avoid repeatedly encoding prefixes.  `byte_offsets[i]` will be
    # the number of bytes in text[:i].
    # Build mapping from char index to byte index.  We step through the
    # encoded bytes and note when the next character begins.
    byte_offsets = [0] * (len(text) + 1)
    byte_index = 0
    for char_index, ch in enumerate(text):
        byte_offsets[char_index] = byte_index
        byte_index += len(ch.encode("utf-8"))
    byte_offsets[len(text)] = byte_index
    # Use regex to extract words
    for match in re.finditer(r"\b\w+\b", text, flags=re.UNICODE):
        word = match.group(0)
        cs = match.start()
        ce = match.end()
        bs = byte_offsets[cs]
        be = byte_offsets[ce]
        occurrences.append(StringOccurrence(word.lower(), file_id, cs, ce, bs, be))
    return occurrences


def aggregate_string_metrics(
    occurrences: Iterable[StringOccurrence],
    signals: List[Dict[str, object]],
    *,
    window_bytes: int,
    stride: int,
    aggregated_fields: Optional[Iterable[str]] = None,
    min_token_length: int = 1,
    min_alpha_ratio: float = 0.0,
    drop_numeric: bool = False,
    min_occurrences: int = 1,
) -> Dict[str, Dict[str, object]]:
    """Aggregate manifold metrics over occurrences of each string.

    Parameters
    ----------
    occurrences:
        Iterable of ``StringOccurrence`` objects.
    signals:
        List of signal dictionaries returned from `build_manifold`.  Each
        signal must contain an ``index`` (the end byte position of the
        window) and a ``metrics`` dict.
    window_bytes:
        Width of each window in bytes – must match what was passed to
        `build_manifold`.
    stride:
        Stride between windows in bytes – must match what was passed to
        `build_manifold`.
    aggregated_fields:
        Optional list of metric field names to aggregate.  Defaults to
        ``["coherence", "stability", "entropy", "rupture"]``.
    min_token_length:
        Minimum length (in characters) a token must have to be included.
    min_alpha_ratio:
        Minimum fraction of alphabetic characters required for a token to
        be included.  ``0.0`` disables the check.
    drop_numeric:
        When ``True`` excludes tokens that are purely numeric.
    min_occurrences:
        Minimum number of occurrences required for a token to be kept in
        the final result.

    Returns
    -------
    Dict[str, Dict[str, object]]
        A mapping from string to a dictionary containing:
        ``metrics`` – the mean metric values over covering windows;
        ``occurrences`` – how many times the string was observed; and
        ``window_ids`` – the sorted list of window identifiers that
        covered at least one occurrence.
    """
    if aggregated_fields is None:
        aggregated_fields = ("coherence", "stability", "entropy", "rupture")
    else:
        aggregated_fields = tuple(aggregated_fields)
    window_ids: List[int] = []
    window_starts: List[int] = []
    window_ends: List[int] = []
    window_metrics: List[Dict[str, float]] = []
    for fallback_id, sig in enumerate(signals):
        end = int(sig.get("window_end", sig.get("index", 0)))
        start = int(sig.get("window_start", end - window_bytes))
        if start < 0:
            start = 0
        window_ids.append(int(sig.get("id", fallback_id)))
        window_starts.append(start)
        window_ends.append(end)
        window_metrics.append(sig.get("metrics", {}))
    metric_samples: DefaultDict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    window_hits: DefaultDict[str, Set[int]] = defaultdict(set)
    occurrence_counts: DefaultDict[str, int] = defaultdict(int)
    token_filters: Dict[str, bool] = {}
    window_count = len(window_ids)
    for occ in occurrences:
        string = occ.string
        if string not in token_filters:
            keep = True
            if len(string) < min_token_length:
                keep = False
            if keep and drop_numeric and string.isdigit():
                keep = False
            if keep and min_alpha_ratio > 0.0:
                alpha = sum(1 for ch in string if ch.isalpha())
                ratio = alpha / len(string) if string else 0.0
                if ratio < min_alpha_ratio:
                    keep = False
            token_filters[string] = keep
        if not token_filters[string]:
            continue
        occurrence_counts[string] += 1
        if window_count == 0:
            continue
        last_cover_idx = bisect_right(window_starts, occ.byte_start) - 1
        first_cover_idx = bisect_left(window_ends, occ.byte_end)
        if first_cover_idx >= window_count or last_cover_idx < 0 or first_cover_idx > last_cover_idx:
            continue
        for idx in range(first_cover_idx, last_cover_idx + 1):
            if window_starts[idx] <= occ.byte_start and window_ends[idx] >= occ.byte_end:
                wid = window_ids[idx]
                window_hits[string].add(wid)
                metrics = window_metrics[idx]
                for field in aggregated_fields:
                    value = metrics.get(field)
                    if value is not None:
                        metric_samples[string][field].append(float(value))
    result: Dict[str, Dict[str, object]] = {}
    for s, field_values in metric_samples.items():
        metric_means: Dict[str, float] = {}
        for field, values in field_values.items():
            if values:
                metric_means[field] = sum(values) / len(values)
        occ_count = occurrence_counts.get(s, 0)
        if occ_count < min_occurrences:
            continue
        result[s] = {
            "metrics": metric_means,
            "occurrences": occ_count,
            "window_ids": sorted(window_hits.get(s, set())),
        }
    for s, count in occurrence_counts.items():
        if count < min_occurrences:
            continue
        if s not in result:
            result[s] = {
                "metrics": {},
                "occurrences": count,
                "window_ids": [],
            }
    return result
