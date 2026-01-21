"""Utilities for building manifold indices (signature postings + ANN)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:  # Optional dependency – required for ANN index
    import hnswlib  # type: ignore
except ImportError as exc:  # pragma: no cover - handled by CLI validation
    hnswlib = None  # type: ignore
    _HNSW_IMPORT_ERROR = exc
else:
    _HNSW_IMPORT_ERROR = None


def _load_state(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _ensure_hnsw() -> None:
    if hnswlib is None:
        raise ImportError(
            "hnswlib is required for building ANN index. Install with 'pip install hnswlib' or 'pip install .[index]'."
        ) from _HNSW_IMPORT_ERROR


def build_signature_postings(
    state: Dict[str, object], *, q: int = 3
) -> Dict[Tuple[str, ...], List[int]]:
    """Return mapping from signature q-gram to list of window IDs (sorted)."""
    signals: Sequence[Dict[str, object]] = sorted(state.get("signals", []), key=lambda s: s["id"])  # type: ignore[assignment]
    postings: Dict[Tuple[str, ...], set[int]] = defaultdict(set)
    if not signals:
        return {}
    signatures: List[str] = [str(sig.get("signature", "")) for sig in signals]
    ids: List[int] = [int(sig["id"]) for sig in signals]
    n = len(signatures)
    if n < q:
        key = tuple(signatures)
        postings[key].add(ids[-1])
    else:
        for i in range(n - q + 1):
            key = tuple(signatures[i : i + q])
            postings[key].add(ids[i + q - 1])
    return {k: sorted(v) for k, v in postings.items()}


def build_ann_index(
    state: Dict[str, object],
    *,
    index_path: Path,
    meta_path: Path,
    ef_construction: int = 200,
    M: int = 32,
    ef_runtime: int = 100,
) -> None:
    """Build and persist HNSW index from manifold window metrics."""
    _ensure_hnsw()
    signals: Sequence[Dict[str, object]] = state.get("signals", [])  # type: ignore[assignment]
    if not signals:
        raise ValueError("State file does not contain 'signals'. Re-run ingest with --store-signals enabled.")
    vecs: List[List[float]] = []
    ids: List[int] = []
    for sig in signals:
        metrics = sig.get("metrics", {})
        vecs.append(
            [
                float(metrics.get("coherence", 0.0)),
                float(metrics.get("stability", 0.0)),
                float(metrics.get("entropy", 1.0)),
                float(metrics.get("rupture", 0.0)),
                float(sig.get("lambda_hazard", metrics.get("rupture", 0.0))),
            ]
        )
        ids.append(int(sig["id"]))
    vecs_arr = np.asarray(vecs, dtype=np.float32)
    ids_arr = np.asarray(ids, dtype=np.int64)
    index = hnswlib.Index(space="l2", dim=vecs_arr.shape[1])
    index.init_index(max_elements=vecs_arr.shape[0], ef_construction=ef_construction, M=M)
    index.add_items(vecs_arr, ids_arr)
    index.set_ef(ef_runtime)
    index.save_index(str(index_path))
    meta = {"dim": int(vecs_arr.shape[1]), "count": int(vecs_arr.shape[0]), "ef": ef_runtime, "M": M}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_indices(
    *,
    state_path: Path,
    postings_path: Path,
    ann_path: Path,
    ann_meta_path: Path,
    q: int = 3,
) -> None:
    """Public entry to build both signature postings and ANN index."""
    state = _load_state(state_path)
    postings = build_signature_postings(state, q=q)
    postings_path.parent.mkdir(parents=True, exist_ok=True)
    postings_path.write_text(
        json.dumps({"\u001f".join(k): v for k, v in postings.items()}, indent=2),
        encoding="utf-8",
    )
    build_ann_index(state, index_path=ann_path, meta_path=ann_meta_path)
