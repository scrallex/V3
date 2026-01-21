"""Utilities for following the manifold append-only log."""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, Optional

from .binary_log import RECORD_STRUCT


def decode_record(blob: bytes) -> Dict[str, float]:
    """Decode a binary manifold record into a dictionary."""
    (
        file_id,
        window_index,
        byte_start,
        window_bytes,
        stride_bytes,
        coherence,
        stability,
        entropy,
        rupture,
        lambda_hazard,
        sig_c,
        sig_s,
        sig_e,
        _reserved,
        _flags,
    ) = RECORD_STRUCT.unpack(blob)
    signature = f"c{sig_c/1000:.3f}_s{sig_s/1000:.3f}_e{sig_e/1000:.3f}"
    return {
        "file_id": int(file_id),
        "window_id": int(window_index),
        "window_start": int(byte_start),
        "window_bytes": int(window_bytes),
        "stride_bytes": int(stride_bytes),
        "coherence": float(coherence),
        "stability": float(stability),
        "entropy": float(entropy),
        "rupture": float(rupture),
        "lambda_hazard": float(lambda_hazard),
        "signature": signature,
    }


def follow_log(
    path: str | Path,
    *,
    poll_interval: float = 0.1,
    from_start: bool = False,
) -> Iterator[Dict[str, float]]:
    """Yield decoded records as they are appended to the binary log."""
    log_path = Path(path)
    record_size = RECORD_STRUCT.size
    with log_path.open("rb") as fh:
        if not from_start:
            fh.seek(0, 2)
        while True:
            blob = fh.read(record_size)
            if len(blob) < record_size:
                time.sleep(poll_interval)
                fh.seek(fh.tell())
                continue
            yield decode_record(blob)
