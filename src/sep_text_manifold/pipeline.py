from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterable, List, Optional

from .ingest import ingest_directory
from .manifold import build_manifold
from .scoring import connector_score, patternability_score
from .strings import StringOccurrence, aggregate_string_metrics, extract_strings
from .themes import build_theme_graph, compute_graph_metrics, detect_themes
from .binary_log import BinaryLogWriter, ManifoldRecord
from .dilution import compute_dilution_metrics




def _extract_file_payload(entry: tuple[str, str, str]) -> tuple[str, str, int, bytes, List[StringOccurrence]]:
    """Helper for parallel string extraction during ingestion."""
    file_id, path, text = entry
    occurrences = extract_strings(text, file_id)
    data_bytes = text.encode("utf-8")
    return file_id, path, len(text), data_bytes, occurrences

@dataclass
class AnalysisSettings:
    directory: str
    window_bytes: int
    stride: int
    extensions: Optional[List[str]]
    min_token_length: int
    min_alpha_ratio: float
    drop_numeric: bool
    min_occurrences: int
    cap_tokens_per_window: int
    graph_min_pmi: float
    graph_max_degree: Optional[int]
    theme_min_size: int
    log_file: Optional[str]
    workers: int
    graph_metric_mode: str
    betweenness_sample: Optional[int]
    max_full_betweenness_nodes: Optional[int]


@dataclass
class FileDigest:
    file_id: str
    path: str
    byte_count: int
    char_count: int
    token_count: int


@dataclass
class AnalysisResult:
    settings: AnalysisSettings
    files: List[FileDigest]
    occurrences: List[StringOccurrence]
    signals: List[Dict[str, Any]]
    string_profiles: Dict[str, Dict[str, Any]]
    string_scores: Dict[str, Dict[str, Any]]
    graph_metrics: Dict[str, Dict[str, float]]
    themes: List[List[str]]
    dilution_summary: Dict[str, Any]

    @property
    def corpus_size_bytes(self) -> int:
        return sum(f.byte_count for f in self.files)

    @property
    def token_count(self) -> int:
        return sum(f.token_count for f in self.files)

    def to_state(
        self,
        *,
        include_signals: bool = False,
        include_occurrences: bool = False,
        include_profiles: bool = False,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "settings": asdict(self.settings),
            "files": [asdict(f) for f in self.files],
            "corpus_size_bytes": self.corpus_size_bytes,
            "token_count": self.token_count,
            "string_scores": self.string_scores,
            "themes": self.themes,
            "graph_metrics": self.graph_metrics,
            "dilution_summary": self.dilution_summary,
        }
        if include_signals:
            state["signals"] = self.signals
        if include_occurrences:
            state["occurrences"] = [asdict(o) for o in self.occurrences]
        if include_profiles:
            state["string_profiles"] = self.string_profiles
        return state

    def summary(self, *, top: int = 10) -> Dict[str, Any]:
        return compute_summary(
            self.string_scores,
            signals=self.signals,
            themes=self.themes,
            corpus_size_bytes=self.corpus_size_bytes,
            token_count=self.token_count,
            file_count=len(self.files),
            dilution_summary=self.dilution_summary,
            top=top,
        )



def analyse_directory(
    directory: str,
    *,
    window_bytes: int = 2048,
    stride: int = 1024,
    extensions: Optional[Iterable[str]] = None,
    verbose: bool = False,
    min_token_length: int = 1,
    min_alpha_ratio: float = 0.0,
    drop_numeric: bool = False,
    min_occurrences: int = 1,
    cap_tokens_per_window: int = 80,
    graph_min_pmi: float = 0.0,
    graph_max_degree: Optional[int] = None,
    theme_min_size: int = 1,
    log_file: Optional[str] = None,
    workers: int = 1,
    graph_metric_mode: str = "full",
    betweenness_sample: Optional[int] = None,
    max_full_betweenness_nodes: Optional[int] = None,
) -> AnalysisResult:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"{directory} is not a directory")

    ext_list: Optional[List[str]] = list(extensions) if extensions is not None else None
    entries = list(ingest_directory(str(root), extensions=ext_list))

    try:
        worker_count = int(workers)
    except (TypeError, ValueError):
        worker_count = 1
    worker_count = max(1, worker_count)

    processed: List[tuple[str, str, int, bytes, List[StringOccurrence]]] = []
    if worker_count == 1:
        for entry in entries:
            processed.append(_extract_file_payload(entry))
    else:
        chunksize = max(1, len(entries) // (worker_count * 4)) if entries else 1
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            for item in pool.map(_extract_file_payload, entries, chunksize=chunksize):
                processed.append(item)

    tokenised_occurrences: List[StringOccurrence] = []
    files: List[FileDigest] = []
    corpus_bytes = bytearray()
    current_offset = 0

    for file_id, path, char_count, data_bytes, occs in processed:
        for occ in occs:
            occ.byte_start += current_offset
            occ.byte_end += current_offset
        tokenised_occurrences.extend(occs)
        corpus_bytes.extend(data_bytes)
        corpus_bytes.append(0)
        if verbose:
            print(
                f"[stm] processed {file_id} ({len(data_bytes)} bytes, {len(occs)} tokens)",
                flush=True,
            )
        files.append(
            FileDigest(
                file_id=file_id,
                path=path,
                byte_count=len(data_bytes),
                char_count=char_count,
                token_count=len(occs),
            )
        )
        current_offset += len(data_bytes) + 1

    if verbose:
        print(f"[stm] building manifold ({len(corpus_bytes)} bytes)", flush=True)
    signals = build_manifold(bytes(corpus_bytes), window_bytes=window_bytes, stride=stride)
    if verbose:
        print(f"[stm] manifold windows: {len(signals)}", flush=True)

    if log_file:
        if verbose:
            print(f"[stm] writing manifold log to {log_file}", flush=True)
        with BinaryLogWriter(log_file) as writer:
            for sig in signals:
                metrics = sig.get("metrics", {})
                coherence = float(metrics.get("coherence", 0.0))
                stability = float(metrics.get("stability", 0.0))
                entropy = float(metrics.get("entropy", 0.0))
                rupture = float(metrics.get("rupture", 0.0))
                lambda_hazard = float(sig.get("lambda_hazard", metrics.get("lambda_hazard", rupture)))
                sig_c = int(round(max(0.0, min(1.0, coherence)) * 1000))
                sig_s = int(round(max(0.0, min(1.0, stability)) * 1000))
                sig_e = int(round(max(0.0, min(1.0, entropy)) * 1000))
                record = ManifoldRecord(
                    file_id=0,
                    window_index=int(sig.get("id", 0)),
                    byte_start=int(sig.get("window_start", sig.get("index", 0) - window_bytes)),
                    window_bytes=window_bytes,
                    stride_bytes=stride,
                    coherence=coherence,
                    stability=stability,
                    entropy=entropy,
                    rupture=rupture,
                    lambda_hazard=lambda_hazard,
                    sig_c=sig_c,
                    sig_s=sig_s,
                    sig_e=sig_e,
                )
                writer.append(record)

    if verbose:
        print("[stm] aggregating string metrics", flush=True)
    string_profiles = aggregate_string_metrics(
        tokenised_occurrences,
        signals,
        window_bytes=window_bytes,
        stride=stride,
        min_token_length=min_token_length,
        min_alpha_ratio=min_alpha_ratio,
        drop_numeric=drop_numeric,
        min_occurrences=min_occurrences,
    )
    if verbose:
        print(f"[stm] strings with profiles: {len(string_profiles)}", flush=True)

    string_scores: Dict[str, Dict[str, Any]] = {}
    for s, profile in string_profiles.items():
        metrics = profile.get("metrics", {})
        p_score = patternability_score(
            metrics.get("coherence", 0.0),
            metrics.get("stability", 0.0),
            metrics.get("entropy", 0.0),
            metrics.get("rupture", 0.0),
        )
        entry: Dict[str, Any] = {
            "metrics": metrics,
            "occurrences": profile.get("occurrences", 0),
            "window_ids": profile.get("window_ids", []),
            "patternability": p_score,
        }
        for field, value in metrics.items():
            entry[field] = value
        string_scores[s] = entry

    if verbose:
        print("[stm] computing theme graph", flush=True)
    occurrence_counts = {s: prof.get("occurrences", 0) for s, prof in string_profiles.items()}
    graph = build_theme_graph(
        {s: prof.get("window_ids", []) for s, prof in string_profiles.items()},
        max_members_per_window=cap_tokens_per_window,
        min_pmi=graph_min_pmi,
        max_degree=graph_max_degree,
        occurrence_counts=occurrence_counts,
        total_windows=len(signals),
    )
    graph_metrics = compute_graph_metrics(
        graph,
        mode=graph_metric_mode,
        betweenness_sample=betweenness_sample,
        max_full_nodes=max_full_betweenness_nodes,
    )
    themes = [
        sorted(list(t))
        for t in detect_themes(graph)
        if len(t) >= theme_min_size
    ]
    if verbose:
        print(f"[stm] themes detected: {len(themes)}", flush=True)

    for s, entry in string_scores.items():
        gm = graph_metrics.get(s, {})
        c_score = connector_score(
            gm.get("betweenness", 0.0),
            gm.get("bridging", 0.0),
            0.0,
            gm.get("theme_entropy_neighbors", 0.0),
            gm.get("redundant_degree", 0.0),
        )
        entry["connector"] = c_score
        entry["graph_metrics"] = gm
    path_dilutions, signal_dilutions, semantic_dilution_score = compute_dilution_metrics(
        signals,
        string_scores,
    )
    for idx, sig in enumerate(signals):
        path_value = path_dilutions[idx] if idx < len(path_dilutions) else 0.0
        signal_value = signal_dilutions[idx] if idx < len(signal_dilutions) else 0.0
        sig.setdefault("dilution", {})
        sig["dilution"].update(
            {
                "path": path_value,
                "signal": signal_value,
            }
        )

    def _mean(values: Iterable[float]) -> float:
        values = list(values)
        if not values:
            return 0.0
        return sum(values) / len(values)

    dilution_summary = {
        "window_count": len(signals),
        "path_mean": _mean(path_dilutions),
        "path_max": max(path_dilutions) if path_dilutions else 0.0,
        "signal_mean": _mean(signal_dilutions),
        "signal_max": max(signal_dilutions) if signal_dilutions else 0.0,
        "semantic_dilution": semantic_dilution_score,
        "context_certainty": max(0.0, min(1.0, 1.0 - _mean(path_dilutions))),
        "signal_clarity": max(0.0, min(1.0, 1.0 - _mean(signal_dilutions))),
        "semantic_clarity": max(0.0, min(1.0, 1.0 - semantic_dilution_score)),
    }
    settings = AnalysisSettings(
        directory=str(root.resolve()),
        window_bytes=window_bytes,
        stride=stride,
        extensions=ext_list,
        min_token_length=min_token_length,
        min_alpha_ratio=min_alpha_ratio,
        drop_numeric=drop_numeric,
        min_occurrences=min_occurrences,
        cap_tokens_per_window=cap_tokens_per_window,
        graph_min_pmi=graph_min_pmi,
        graph_max_degree=graph_max_degree,
        theme_min_size=theme_min_size,
        log_file=log_file,
        workers=worker_count,
        graph_metric_mode=graph_metric_mode,
        betweenness_sample=betweenness_sample,
        max_full_betweenness_nodes=max_full_betweenness_nodes,
    )
    return AnalysisResult(
        settings=settings,
        files=files,
        occurrences=tokenised_occurrences,
        signals=signals,
        string_profiles=string_profiles,
        string_scores=string_scores,
        graph_metrics=graph_metrics,
        themes=themes,
        dilution_summary=dilution_summary,
    )


def compute_summary(
    string_scores: Dict[str, Dict[str, Any]],
    *,
    signals: Optional[List[Dict[str, Any]]] = None,
    themes: Optional[List[Iterable[str]]] = None,
    corpus_size_bytes: Optional[int] = None,
    token_count: Optional[int] = None,
    file_count: Optional[int] = None,
    dilution_summary: Optional[Dict[str, Any]] = None,
    top: int = 10,
) -> Dict[str, Any]:
    if top <= 0:
        top = 10
    strings_sorted = sorted(
        ((s, data) for s, data in string_scores.items()),
        key=lambda item: item[1].get("patternability", 0.0),
        reverse=True,
    )
    connectors_sorted = sorted(
        ((s, data) for s, data in string_scores.items()),
        key=lambda item: item[1].get("connector", 0.0),
        reverse=True,
    )
    top_patterns = [
        {
            "string": s,
            "patternability": data.get("patternability", 0.0),
            "occurrences": data.get("occurrences", 0),
        }
        for s, data in strings_sorted[:top]
    ]
    top_connectors = [
        {
            "string": s,
            "connector": data.get("connector", 0.0),
            "occurrences": data.get("occurrences", 0),
        }
        for s, data in connectors_sorted[:top]
    ]
    window_count = len(signals) if signals is not None else None
    avg_window_metrics: Dict[str, float] = {}
    if signals:
        totals: Dict[str, float] = {}
        count = 0
        for sig in signals:
            metrics = sig.get("metrics", {})
            if not metrics:
                continue
            count += 1
            for field, value in metrics.items():
                totals[field] = totals.get(field, 0.0) + float(value)
        if count:
            avg_window_metrics = {name: value / count for name, value in totals.items()}
    summary: Dict[str, Any] = {
        "string_count": len(string_scores),
        "top_patternable_strings": top_patterns,
        "top_connectors": top_connectors,
        "theme_count": len(themes or []),
    }
    if corpus_size_bytes is not None:
        summary["corpus_size_bytes"] = corpus_size_bytes
    if token_count is not None:
        summary["token_count"] = token_count
    if file_count is not None:
        summary["file_count"] = file_count
    if window_count is not None:
        summary["window_count"] = window_count
    if avg_window_metrics:
        summary["mean_window_metrics"] = avg_window_metrics
    if dilution_summary:
        summary["dilution"] = dict(dilution_summary)
    return summary
