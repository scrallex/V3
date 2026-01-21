"""
Manifold construction for the Sep Text Manifold project.

This module defines functions to build a manifold – an ordered list of
dynamic fingerprints – from a continuous byte stream.  Each
fingerprint summarises a sliding window of bytes using the quantum
metrics defined in `encode.py` and includes a repetition signature and
hazard estimate.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .encode import encode_window, signature_from_metrics


def sliding_windows(
    data: bytes,
    window_bytes: int,
    stride: int,
    *,
    include_partial_tail: bool = True,
) -> Iterator[Tuple[int, int, bytes]]:
    """Generate overlapping windows from a bytes object.

    Parameters
    ----------
    data:
        The byte sequence to cut into windows.
    window_bytes:
        The width of each window in bytes.
    stride:
        How many bytes to advance between windows.  A stride smaller
        than `window_bytes` results in overlapping windows.
    """
    if window_bytes <= 0 or stride <= 0:
        raise ValueError("window_bytes and stride must be positive")
    n = len(data)
    if n == 0:
        return
    # If the corpus is shorter than the window size, emit a single
    # partial window covering the entire payload.  Downstream code
    # treats the start index as zero in this case.
    if n <= window_bytes:
        yield (0, n, data)
        return
    last_end = 0
    for end in range(window_bytes, n + 1, stride):
        start = end - window_bytes
        yield (start, end, data[start:end])
        last_end = end
    if include_partial_tail and last_end < n:
        start = max(0, n - window_bytes)
        yield (start, n, data[start:n])


def build_manifold(data: bytes, *, window_bytes: int = 2048, stride: int = 1024, signature_precision: int = 2, max_signals: Optional[int] = None) -> List[Dict[str, object]]:
    """Build a manifold from a contiguous byte sequence.

    The returned list contains a dict for each window containing
    indices and metrics.  This function is a high‑level Python
    analogue of `buildManifold` in the SEP C++ code【739285426356909†L110-L169】.

    Parameters
    ----------
    data:
        The entire corpus as a bytes object.  If you wish to treat
        multiple files as a single stream, concatenate them with
        separators before calling this function.
    window_bytes:
        Number of bytes in each window.
    stride:
        Number of bytes between successive window ends.
    signature_precision:
        Decimal precision for bucketing metrics into repetition
        signatures.
    max_signals:
        Optional cap on the number of signals returned.  If provided
        and the data would produce more signals, only the most recent
        `max_signals` windows are emitted.

    Returns
    -------
    List[dict]
        A list of signal dictionaries.  Each dictionary contains:

        - **id** – a monotonically increasing integer identifying the
          window within the manifold.
        - **window_start** – the starting byte index (inclusive) of the
          window within the original data.
        - **window_end**/**index** – the ending byte index (exclusive)
          of the window within the original data.  ``index`` is kept as
          an alias for backwards compatibility.
        - **metrics** – a dictionary of coherence, stability, entropy
          and rupture.
        - **signature** – a repetition signature string computed from
          the metrics.
        - **lambda_hazard** – hazard lambda equal to the rupture value.
    """
    signals: List[Dict[str, object]] = []
    for window_id, (window_start, window_end, window) in enumerate(
        sliding_windows(data, window_bytes, stride),
        start=0,
    ):
        metrics = encode_window(window)
        sig = signature_from_metrics(
            metrics["coherence"], metrics["stability"], metrics["entropy"], precision=signature_precision
        )
        signals.append(
            {
                "id": window_id,
                "window_start": window_start,
                "window_end": window_end,
                "index": window_end,
                "metrics": metrics,
                "signature": sig,
                "lambda_hazard": metrics.get("lambda_hazard", 0.0),
            }
        )
    # If max_signals is set, keep only the most recent signals
    if max_signals is not None and len(signals) > max_signals:
        signals = signals[-max_signals:]
    return signals
