"""Append-only manifold log helpers."""

from __future__ import annotations

import os
import struct
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

RECORD_STRUCT = struct.Struct("<QQQIIfffffHHHHI")


@dataclass
class ManifoldRecord:
    file_id: int
    window_index: int
    byte_start: int
    window_bytes: int
    stride_bytes: int
    coherence: float
    stability: float
    entropy: float
    rupture: float
    lambda_hazard: float
    sig_c: int
    sig_s: int
    sig_e: int
    reserved: int = 0
    flags: int = 0

    def pack(self) -> bytes:
        return RECORD_STRUCT.pack(
            self.file_id & 0xFFFFFFFFFFFFFFFF,
            self.window_index & 0xFFFFFFFFFFFFFFFF,
            self.byte_start & 0xFFFFFFFFFFFFFFFF,
            self.window_bytes & 0xFFFFFFFF,
            self.stride_bytes & 0xFFFFFFFF,
            float(self.coherence),
            float(self.stability),
            float(self.entropy),
            float(self.rupture),
            float(self.lambda_hazard),
            self.sig_c & 0xFFFF,
            self.sig_s & 0xFFFF,
            self.sig_e & 0xFFFF,
            self.reserved & 0xFFFF,
            self.flags & 0xFFFFFFFF,
        )


class BinaryLogWriter(AbstractContextManager["BinaryLogWriter"]):
    """Simple append-only binary writer for manifold records."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("ab")

    def append(self, record: ManifoldRecord) -> None:
        self._fh.write(record.pack())

    def close(self) -> None:  # pragma: no cover - public API
        if not self._fh.closed:
            self._fh.close()

    def __exit__(self, exc_type, exc, exc_tb) -> Optional[bool]:  # pragma: no cover - context cleanup
        self.close()
        return None
