#!/usr/bin/env python3
"""Utilities for recovering structural manifold prototypes on demand."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, Iterable, Set
import sys

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
TOP_ROOT = REPO_ROOT.parent
for candidate in (MODULE_DIR, REPO_ROOT, TOP_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from manifold_compression_eval import iter_text_documents
from src.manifold.codec import encode_text


def collect_prototypes(
    signatures: Iterable[str],
    *,
    text_root: Path,
    window_bytes: int,
    stride_bytes: int,
    precision: int,
    max_documents: int | None = None,
) -> Dict[str, bytes]:
    """Scan ``text_root`` until the given signatures have payloads."""

    remaining: Set[str] = {sig for sig in signatures if sig}
    prototypes: Dict[str, bytes] = {}
    if not remaining:
        return prototypes

    for doc_index, (_, text) in enumerate(iter_text_documents(text_root)):
        if max_documents is not None and doc_index >= max_documents:
            break
        encoding = encode_text(
            text,
            window_bytes=window_bytes,
            stride_bytes=stride_bytes,
            precision=precision,
            deduplicate=False,
        )
        for span in encoding.spans:
            sig = span.signature
            if sig in remaining:
                chunk = span.payload
                if chunk is None:
                    continue
                prototypes[sig] = chunk
                remaining.remove(sig)
                if not remaining:
                    return prototypes
    return prototypes


def encode_prototypes(prototypes: Dict[str, bytes]) -> Dict[str, str]:
    """Base64 encode prototype payloads for JSON storage."""

    return {sig: base64.b64encode(payload).decode("ascii") for sig, payload in prototypes.items()}


def decode_prototypes(payload: Dict[str, str]) -> Dict[str, bytes]:
    """Inverse of :func:`encode_prototypes`."""

    return {sig: base64.b64decode(blob.encode("ascii")) for sig, blob in payload.items()}


def load_prototype_cache(path: Path) -> Dict[str, bytes]:
    """Load a prototype cache stored as JSON (signature -> base64 chunk)."""

    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return decode_prototypes(data)


def save_prototype_cache(path: Path, prototypes: Dict[str, bytes]) -> None:
    """Persist prototypes to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = encode_prototypes(prototypes)
    path.write_text(json.dumps(encoded, indent=2), encoding="utf-8")


def ensure_prototypes(
    signatures: Iterable[str],
    *,
    cache_path: Path | None,
    text_root: Path,
    window_bytes: int,
    stride_bytes: int,
    precision: int,
    max_documents: int | None = None,
) -> Dict[str, bytes]:
    """Load prototypes from cache and backfill missing entries by scanning text."""

    cache: Dict[str, bytes] = load_prototype_cache(cache_path) if cache_path else {}
    missing = {sig for sig in signatures if sig and sig not in cache}
    if missing:
        recovered = collect_prototypes(
            missing,
            text_root=text_root,
            window_bytes=window_bytes,
            stride_bytes=stride_bytes,
            precision=precision,
            max_documents=max_documents,
        )
        cache.update(recovered)
        if cache_path:
            save_prototype_cache(cache_path, cache)
    return cache


__all__ = [
    "collect_prototypes",
    "decode_prototypes",
    "encode_prototypes",
    "ensure_prototypes",
    "load_prototype_cache",
    "save_prototype_cache",
]
