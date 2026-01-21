"""Dilution metrics for structural manifolds."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Tuple, List

__all__ = [
    "path_dilution",
    "signal_dilution",
    "semantic_dilution",
    "compute_dilution_metrics",
]


def _entropy_from_counts(counts: Iterable[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _normalise_entropy(entropy: float, cardinality: int) -> float:
    if cardinality <= 1:
        return 0.0
    max_entropy = math.log2(cardinality)
    if max_entropy <= 0.0:
        return 0.0
    value = entropy / max_entropy
    return max(0.0, min(1.0, value))


def path_dilution(signature_sequence: Sequence[str], window_index: int) -> float:
    """Return the normalised entropy of the next-signature distribution.

    Parameters
    ----------
    signature_sequence:
        Ordered list of repetition signatures for each manifold window.
    window_index:
        Index into ``signature_sequence`` representing the current window.

    Returns
    -------
    float
        A value in ``[0, 1]`` where ``0`` indicates that the signature is
        followed by a single, highly predictable successor and ``1``
        indicates that the successor space is maximally diverse.
    """
    if window_index < 0 or window_index >= len(signature_sequence):
        raise IndexError("window_index out of range")
    current_signature = signature_sequence[window_index]
    if not signature_sequence:
        return 0.0
    followers: Counter[str] = Counter()
    for idx, signature in enumerate(signature_sequence[:-1]):
        if signature == current_signature:
            followers[signature_sequence[idx + 1]] += 1
    if not followers:
        return 0.0
    entropy = _entropy_from_counts(followers.values())
    return _normalise_entropy(entropy, len(followers))


def signal_dilution(foreground_tokens: Iterable[str]) -> float:
    """Return a diversity score over structural tokens in the foreground.

    The score is defined as the normalised Shannon entropy of the token
    distribution.  Windows dominated by a single token yield low dilution
    (approaching ``0``) whereas windows containing a broad mix of tokens
    yield scores closer to ``1``.
    """
    counts: Counter[str] = Counter(token for token in foreground_tokens if token)
    if not counts:
        return 0.0
    entropy = _entropy_from_counts(counts.values())
    return _normalise_entropy(entropy, len(counts))


def semantic_dilution(structural_signatures: Sequence[str], semantic_tokens: Sequence[str]) -> float:
    """Return ``1 -`` the normalised mutual information between signals and semantics.

    Parameters
    ----------
    structural_signatures:
        Sequence of structural signatures (e.g. repetition hashes) aligned
        with ``semantic_tokens``.  Each position represents one
        structural-semantic co-occurrence.
    semantic_tokens:
        Semantic labels extracted from the same positions as
        ``structural_signatures``.

    Returns
    -------
    float
        Dilution score in ``[0, 1]``.  ``0`` indicates perfectly aligned
        semantics (high mutual information), while ``1`` indicates
        semantics that are independent of structural signatures.
    """
    if len(structural_signatures) != len(semantic_tokens):
        raise ValueError("structural_signatures and semantic_tokens must have the same length")
    if not structural_signatures:
        return 0.0
    joint_counts: Counter[Tuple[str, str]] = Counter(zip(structural_signatures, semantic_tokens))
    structural_counts: Counter[str] = Counter(structural_signatures)
    semantic_counts: Counter[str] = Counter(semantic_tokens)
    total = len(structural_signatures)
    mutual_information = 0.0
    for (signature, token), joint_count in joint_counts.items():
        if joint_count <= 0:
            continue
        p_joint = joint_count / total
        p_sig = structural_counts[signature] / total
        p_sem = semantic_counts[token] / total
        if p_joint <= 0.0 or p_sig <= 0.0 or p_sem <= 0.0:
            continue
        mutual_information += p_joint * math.log2(p_joint / (p_sig * p_sem))
    if mutual_information <= 0.0:
        return 1.0
    h_struct = _entropy_from_counts(structural_counts.values())
    h_sem = _entropy_from_counts(semantic_counts.values())
    max_mi = min(h_struct, h_sem)
    if max_mi <= 0.0:
        return 1.0
    normalised = mutual_information / max_mi
    normalised = max(0.0, min(1.0, normalised))
    return 1.0 - normalised


def compute_dilution_metrics(
    signals: Sequence[Mapping[str, Any]],
    token_windows: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[float], List[float], float]:
    """Compute dilution series for the provided manifold signals.

    Parameters
    ----------
    signals:
        Sequence of signal dictionaries (as produced by
        ``build_manifold``).  Each entry must contain ``id`` and
        ``signature`` fields.
    token_windows:
        Mapping from token string to payload containing a ``window_ids``
        iterable (either the raw string profiles or the enriched string
        scores).  Tokens listed against a window contribute to its
        foreground set when computing signal dilution.

    Returns
    -------
    tuple
        ``(path_series, signal_series, semantic_dilution_score)`` where
        the series are aligned with ``signals``.
    """
    if not signals:
        return [], [], 0.0
    signature_sequence = [str(sig.get("signature", "")) for sig in signals]
    window_id_to_index: MutableMapping[int, int] = {}
    for idx, sig in enumerate(signals):
        try:
            window_id = int(sig.get("id", idx))
        except (TypeError, ValueError):
            window_id = idx
        window_id_to_index[window_id] = idx
    window_tokens: MutableMapping[int, List[str]] = defaultdict(list)
    struct_labels: List[str] = []
    semantic_labels: List[str] = []
    for token, payload in token_windows.items():
        if not isinstance(payload, Mapping):
            continue
        window_ids = payload.get("window_ids")
        if not isinstance(window_ids, Iterable):
            continue
        for raw_wid in window_ids:
            try:
                window_id = int(raw_wid)
            except (TypeError, ValueError):
                continue
            if window_id not in window_id_to_index:
                continue
            window_tokens[window_id].append(token)
            signature = signature_sequence[window_id_to_index[window_id]]
            struct_labels.append(signature)
            semantic_labels.append(token)
    path_series = [path_dilution(signature_sequence, idx) for idx in range(len(signature_sequence))]
    signal_series: List[float] = []
    for idx, sig in enumerate(signals):
        try:
            window_id = int(sig.get("id", idx))
        except (TypeError, ValueError):
            window_id = idx
        signal_series.append(signal_dilution(window_tokens.get(window_id, [])))
    semantic_score = 1.0
    if struct_labels:
        semantic_score = semantic_dilution(struct_labels, semantic_labels)
    else:
        semantic_score = 1.0
    return path_series, signal_series, semantic_score
