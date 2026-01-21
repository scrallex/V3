from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:  # Optional dependency – index build ensures availability
    import hnswlib  # type: ignore
except ImportError as exc:  # pragma: no cover - gracefully handled by callers
    hnswlib = None  # type: ignore
    _HNSW_IMPORT_ERROR = exc
else:
    _HNSW_IMPORT_ERROR = None


class SeenEngine:
    """Routes trigger strings to foreground/deferred manifold windows."""

    def __init__(self, base_path: Path | str = Path("analysis")) -> None:
        self.base = Path(base_path)
        state_path = self.base / "score_state_native.json"
        router_path = self.base / "router_config.json"
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}. Run 'stm ingest --store-signals' first.")
        if not router_path.exists():
            raise FileNotFoundError(f"Router config not found: {router_path}.")
        self.state: Dict[str, object] = json.loads(state_path.read_text())
        self.router_path = router_path
        self.router = self._load_router()

        # Window map (id -> signal)
        ordered_signals = sorted(self.state.get("signals", []), key=lambda s: s["id"])  # type: ignore[arg-type]
        self.windows: Dict[int, Dict[str, object]] = {int(sig["id"]): sig for sig in ordered_signals}
        self.id_order: List[int] = [int(sig["id"]) for sig in ordered_signals]
        self.signature_sequence: List[str] = [sig.get("signature", "") for sig in ordered_signals]
        self.id_to_pos: Dict[int, int] = {wid: idx for idx, wid in enumerate(self.id_order)}
        # Map string -> score payload
        self.strings: Dict[str, Dict[str, object]] = self.state.get("string_scores", {})  # type: ignore[assignment]

        # Signature postings (q-gram -> window ids)
        postings_file = self.base / "signature_postings.json"
        if not postings_file.exists():
            raise FileNotFoundError(
                f"Signature postings not found: {postings_file}. Run 'stm index build' after ingest."
            )
        postings_raw = json.loads(postings_file.read_text())
        self.postings: Dict[Tuple[str, ...], set[int]] = {
            tuple(key.split("\u001f")): set(map(int, ids)) for key, ids in postings_raw.items()
        }

        # ANN index
        ann_meta = self.base / "ann.meta"
        ann_index = self.base / "ann.hnsw"
        if not ann_meta.exists() or not ann_index.exists():
            raise FileNotFoundError(
                f"ANN index/meta missing ({ann_index}, {ann_meta}). Run 'stm index build' first."
            )
        if hnswlib is None:
            raise ImportError(
                "hnswlib is required for the seen router. Install with 'pip install hnswlib' or 'pip install .[index]'."
            ) from _HNSW_IMPORT_ERROR
        meta = json.loads(ann_meta.read_text())
        dim = int(meta["dim"])
        self.ann = hnswlib.Index(space="l2", dim=dim)
        self.ann.load_index(str(ann_index))
        self.ann.set_ef(int(meta.get("ef", 100)))

    # --- helpers ---------------------------------------------------------

    def _load_router(self) -> Dict[str, object]:
        return json.loads(self.router_path.read_text())

    def locate_trigger_windows(self, trigger: str) -> set[int]:
        entry = self.strings.get(trigger)
        if not entry:
            return set()
        return set(entry.get("window_ids", []))

    def window_vector(self, wid: int) -> np.ndarray | None:
        sig = self.windows.get(wid)
        if not sig:
            return None
        metrics = sig.get("metrics", {})
        return np.array([
            float(metrics.get("coherence", 0.0)),
            float(metrics.get("stability", 0.0)),
            float(metrics.get("entropy", 1.0)),
            float(metrics.get("rupture", 0.0)),
            float(sig.get("lambda_hazard", metrics.get("rupture", 0.0))),
        ], dtype=np.float32)

    def centroid(self, wids: Iterable[int]) -> np.ndarray | None:
        vecs = [self.window_vector(wid) for wid in wids]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            return None
        return np.vstack(vecs).mean(axis=0)

    def qgram_hits(self, seeds: Iterable[int], q: int = 3, limit: int = 200) -> set[int]:
        counts: Dict[int, int] = {}
        n = len(self.signature_sequence)
        if n == 0:
            return set()
        if n < q:
            key = tuple(self.signature_sequence)
            for wid in self.postings.get(key, set()):
                counts[wid] = counts.get(wid, 0) + 1
            return set(wid for wid, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit])
        for wid in seeds:
            pos = self.id_to_pos.get(wid)
            if pos is None:
                continue
            start_min = max(0, pos - q + 1)
            start_max = min(pos, n - q)
            if start_min > start_max:
                start_min = start_max = pos
            for start in range(start_min, start_max + 1):
                qg = tuple(self.signature_sequence[start : start + q])
                for hit in self.postings.get(qg, set()):
                    counts[hit] = counts.get(hit, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return set(wid for wid, _ in ranked[:limit])

    def ann_hits(self, target: np.ndarray, k: int = 200, max_dist: float = 0.20) -> set[int]:
        if target is None:
            return set()
        labels, distances = self.ann.knn_query(target.reshape(1, -1), k=k)
        return {int(lbl) for lbl, dist in zip(labels[0], distances[0]) if dist <= max_dist}

    def route(self, candidates: Iterable[int]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        router = self._load_router()
        cfg = router["router"]["foreground"]  # type: ignore[index]
        min_coh = float(cfg.get("min_coh", 0.8))
        max_ent = float(cfg.get("max_ent", 0.35))
        min_stab = float(cfg.get("min_stab", 0.0))
        foreground: List[Dict[str, object]] = []
        deferred: List[Dict[str, object]] = []
        for wid in candidates:
            sig = self.windows.get(wid)
            if not sig:
                continue
            metrics = sig.get("metrics", {})
            item = {
                "window_id": wid,
                "coherence": metrics.get("coherence", 0.0),
                "entropy": metrics.get("entropy", 1.0),
                "stability": metrics.get("stability", 0.0),
                "rupture": metrics.get("rupture", 0.0),
                "signature": sig.get("signature"),
            }
            if (
                item["coherence"] >= min_coh
                and item["entropy"] <= max_ent
                and item["stability"] >= min_stab
            ):
                foreground.append(item)
            else:
                deferred.append(item)
        foreground.sort(key=lambda x: (-x["coherence"], x["entropy"]))
        deferred.sort(key=lambda x: (-x["coherence"], x["entropy"]))
        return foreground[:100], deferred[:300]

    # --- public entry ----------------------------------------------------

    def seen(self, trigger: str) -> Dict[str, List[Dict[str, object]]]:
        seeds = self.locate_trigger_windows(trigger)
        if not seeds:
            return {"foreground": [], "deferred": []}
        centroid = self.centroid(seeds)
        router = self._load_router()
        trig_cfg = router["router"]["triggers"]  # type: ignore[index]
        max_dist = float(trig_cfg.get("max_ann_dist", 0.20))
        q_hits = self.qgram_hits(seeds, q=int(trig_cfg.get("min_sig_qgrams", 2)))
        ann_hits = self.ann_hits(centroid, max_dist=max_dist) if centroid is not None else set()
        candidates = seeds | q_hits | ann_hits
        fg, df = self.route(candidates)
        return {"foreground": fg, "deferred": df}

    def update_window(self, record: Dict[str, object]) -> None:
        """Incrementally add a new window from the manifold log."""
        if hnswlib is None:
            raise ImportError(
                "hnswlib is required for router updates. Install with 'pip install hnswlib'."
            ) from _HNSW_IMPORT_ERROR
        wid = int(record["window_id"])
        metrics = {
            "coherence": float(record.get("coherence", 0.0)),
            "stability": float(record.get("stability", 0.0)),
            "entropy": float(record.get("entropy", 1.0)),
            "rupture": float(record.get("rupture", 0.0)),
        }
        signal = {
            "id": wid,
            "window_start": record.get("window_start"),
            "window_end": record.get("window_end"),
            "index": record.get("window_end"),
            "metrics": metrics,
            "signature": record.get("signature"),
            "lambda_hazard": record.get("lambda_hazard", metrics["rupture"]),
        }
        self.windows[wid] = signal
        self.id_order.append(wid)
        self.id_to_pos[wid] = len(self.id_order) - 1
        signature = str(record.get("signature", ""))
        self.signature_sequence.append(signature)

        # Update q-gram postings
        router = self._load_router()
        q = int(router["router"]["triggers"].get("min_sig_qgrams", 2))  # type: ignore[index]
        if q <= 0:
            q = 1
        n = len(self.signature_sequence)
        start_min = max(0, n - q)
        for start in range(start_min, n):
            if start + q > n:
                continue
            key = tuple(self.signature_sequence[start : start + q])
            self.postings.setdefault(key, set()).add(wid)

        # Update ANN index
        vec = self.window_vector(wid)
        if vec is not None:
            current = self.ann.get_current_count()
            max_elements = self.ann.get_max_elements()
            if current >= max_elements:
                self.ann.resize_index(max_elements + max(1, max_elements // 2))
            self.ann.add_items(vec.reshape(1, -1), np.array([wid], dtype=np.int64))


_engine: SeenEngine | None = None


def get_engine() -> SeenEngine:
    global _engine
    if _engine is None:
        _engine = SeenEngine()
    return _engine
