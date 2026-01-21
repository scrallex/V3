"""
Command line interface for the Sep Text Manifold project.

This module uses ``argparse`` to expose a small set of commands for
ingesting a corpus, analysing it and exploring the results.  The CLI
flows through the entire pipeline: reading files, building the
manifold, extracting strings, aggregating metrics, computing scores
and detecting themes.

Results are stored in a JSON file (`stm_state.json` by default) so
that subsequent commands can operate on cached analysis rather than
recomputing everything.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from statistics import mean, median
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .pipeline import analyse_directory, compute_summary
from .propose import propose as run_proposer, propose_from_state, load_state as load_analysis_state
from .index_builder import build_indices
from .similar import cross_corpus_similarity
from .filters import (
    flatten_metrics,
    metric_matches,
    parse_metric_filter,
    requested_percentiles,
    compute_metric_quantiles,
)
from .dilution import compute_dilution_metrics


def _load_state(state_file: Path) -> Dict[str, Any]:
    if state_file.exists():
        with state_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(state_file: Path, data: Dict[str, Any]) -> None:
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _print_summary(summary: Dict[str, Any]) -> None:
    if not summary:
        print("No summary available.")
        return
    file_count = summary.get("file_count")
    token_count = summary.get("token_count")
    corpus_size = summary.get("corpus_size_bytes")
    window_count = summary.get("window_count")
    print("Summary")
    print("-------")
    if file_count is not None:
        print(f"Files analysed: {file_count}")
    if token_count is not None:
        print(f"Tokens extracted: {token_count}")
    if corpus_size is not None:
        print(f"Corpus size (bytes): {corpus_size}")
    if window_count is not None:
        print(f"Manifold windows: {window_count}")
    print(f"Unique strings scored: {summary.get('string_count', 0)}")
    top_patterns = summary.get("top_patternable_strings", [])
    if top_patterns:
        print("\nTop patternable strings:")
        for entry in top_patterns:
            label = entry.get("string", "<unknown>")
            print(
                f"  {label:<24} pattern={entry.get('patternability', 0.0):.3f}"
                f" occ={entry.get('occurrences', 0)}"
            )
    top_connectors = summary.get("top_connectors", [])
    if top_connectors:
        print("\nTop connectors:")
        for entry in top_connectors:
            label = entry.get("string", "<unknown>")
            print(
                f"  {label:<24} connector={entry.get('connector', 0.0):.3f}"
                f" occ={entry.get('occurrences', 0)}"
            )
    if summary.get("theme_count") is not None:
        print(f"\nTheme count: {summary['theme_count']}")
    mean_metrics = summary.get("mean_window_metrics")
    if mean_metrics:
        print("\nMean window metrics:")
        for name, value in mean_metrics.items():
            print(f"  {name}: {value:.3f}")
    dilution = summary.get("dilution")
    if dilution:
        print("\nDilution metrics:")
        print(
            f"  context certainty: {dilution.get('context_certainty', 0.0):.3f}"
            f"  signal clarity: {dilution.get('signal_clarity', 0.0):.3f}"
            f"  semantic clarity: {dilution.get('semantic_clarity', 0.0):.3f}"
        )
        print(
            f"  path mean: {dilution.get('path_mean', 0.0):.3f}"
            f"  path max: {dilution.get('path_max', 0.0):.3f}"
            f"  signal mean: {dilution.get('signal_mean', 0.0):.3f}"
            f"  signal max: {dilution.get('signal_max', 0.0):.3f}"
        )


def _collect_seeds(args: argparse.Namespace) -> List[str]:
    seeds: List[str] = []
    if getattr(args, "seeds", None):
        seeds.extend(s.strip() for s in args.seeds.split(",") if s.strip())
    seed_file = getattr(args, "seed_file", None)
    if seed_file:
        with Path(seed_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                token = line.strip()
                if token:
                    seeds.append(token)
    unique_seeds = []
    for s in seeds:
        if s not in unique_seeds:
            unique_seeds.append(s)
    if not unique_seeds:
        raise ValueError("No seeds provided. Use --seeds or --seed-file.")
    return unique_seeds


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a directory of text files (or adapter output) and analyse it."""
    source_path = Path(args.directory)
    cleanup_dir: Optional[Path] = None
    directory_path: Path
    extensions = args.extensions
    adapter_name = getattr(args, "adapter", None)
    if adapter_name:
        from stm_adapters import get_adapter

        adapter = get_adapter(adapter_name)
        tmp_dir = Path(tempfile.mkdtemp(prefix="stm_adapter_"))
        tokens_path = adapter.run(source_path, tmp_dir)
        directory_path = tokens_path.parent
        if not extensions:
            extensions = ["txt"]
        if not args.keep_adapter_output:
            cleanup_dir = tmp_dir
    else:
        directory_path = source_path

    window_bytes = args.window_bytes
    stride = args.stride
    state_file = Path(args.output)
    result = analyse_directory(
        str(directory_path),
        window_bytes=window_bytes,
        stride=stride,
        extensions=extensions,
        verbose=args.verbose,
        min_token_length=args.min_token_len,
        min_alpha_ratio=args.alpha_ratio,
        drop_numeric=args.drop_numeric,
        min_occurrences=args.min_occ,
        cap_tokens_per_window=args.cap_tokens_per_win,
        graph_min_pmi=args.graph_min_pmi,
        graph_max_degree=args.graph_max_degree,
        theme_min_size=args.theme_min_size,
        log_file=args.log_file,
    )
    summary = result.summary(top=args.summary_top)
    state = result.to_state(
        include_signals=args.store_signals,
        include_occurrences=args.store_occurrences,
        include_profiles=args.store_profiles,
    )
    state["summary"] = summary
    state["dilution_summary"] = result.dilution_summary
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    state["generated_at"] = generated_at
    _save_state(state_file, state)
    print(f"Analysis complete.  Saved state to {state_file}")
    print()
    _print_summary(summary)
    if cleanup_dir is not None:
        shutil.rmtree(cleanup_dir, ignore_errors=True)


def cmd_strings(args: argparse.Namespace) -> None:
    """List strings sorted by patternability."""
    state = _load_state(Path(args.input))
    scores: Dict[str, Dict[str, Any]] = state.get("string_scores", {})
    if not scores:
        print("No string scores found.  Run 'stm ingest' first.")
        return
    try:
        constraints = parse_metric_filter(getattr(args, "filter", None))
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    percentile_requests = requested_percentiles(constraints)
    metric_values: Dict[str, List[float]] = {}
    for entry in scores.values():
        metrics = flatten_metrics(entry)
        for key, value in metrics.items():
            metric_values.setdefault(key, []).append(float(value))
    quantiles = compute_metric_quantiles(metric_values, percentile_requests)
    items: List[Tuple[str, Dict[str, Any], Dict[str, float]]] = []
    for text, entry in scores.items():
        metrics = flatten_metrics(entry)
        if not metric_matches(metrics, constraints, quantiles=quantiles):
            continue
        items.append((text, entry, metrics))
    if not items:
        print("No strings matched the specified filters.")
        return
    if args.sort == "connector":
        items.sort(key=lambda item: item[1].get("connector", item[2].get("connector", 0.0)), reverse=True)
    else:
        items.sort(key=lambda item: item[1].get("patternability", item[2].get("patternability", 0.0)), reverse=True)
    top = args.top or len(items)
    for i, (text, entry, metrics) in enumerate(items[:top], start=1):
        patternability = float(entry.get("patternability", metrics.get("patternability", 0.0)))
        connector = float(entry.get("connector", metrics.get("connector", 0.0)))
        occurrences = int(entry.get("occurrences", 0))
        print(
            f"{i:3d}. {text:<24} pattern={patternability:.3f}"
            f" connector={connector:.3f} occ={occurrences}"
        )


def cmd_themes(args: argparse.Namespace) -> None:
    """List themes and their member strings."""
    state = _load_state(Path(args.input))
    themes: List[List[str]] = state.get("themes", [])
    if not themes:
        print("No themes found.  Run 'stm ingest' first.")
        return
    for i, members in enumerate(themes, start=1):
        print(f"Theme {i} ({len(members)} strings): {', '.join(sorted(members)[:20])}{'...' if len(members) > 20 else ''}")


def cmd_summary(args: argparse.Namespace) -> None:
    """Display a summary of the latest analysis."""
    state = _load_state(Path(args.input))
    if not state:
        print("No analysis state found.  Run 'stm ingest' first.")
        return
    top_n = args.top or 10
    summary = compute_summary(
        state.get("string_scores", {}),
        signals=state.get("signals"),
        themes=state.get("themes"),
        corpus_size_bytes=state.get("corpus_size_bytes"),
        token_count=state.get("token_count"),
        file_count=len(state.get("files", [])) or state.get("summary", {}).get("file_count"),
        top=top_n,
    )
    _print_summary(summary)


def cmd_propose(args: argparse.Namespace) -> None:
    try:
        seeds = _collect_seeds(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    result = run_proposer(
        args.input,
        seeds=seeds,
        k=args.k,
        min_connector=args.min_connector,
        min_patternability=args.min_patternability,
        target_profile=args.target_profile,
    )
    proposals = result["proposals"]
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Generated {len(proposals)} proposals (showing up to {args.top}).")
    for idx, item in enumerate(proposals[: args.top], start=1):
        print(
            f"{idx:3d}. {item['string']:<24} score={item['score']:.3f}"
            f" pattern={item['patternability']:.3f} connector={item['connector']:.3f}"
            f" occ={item['occurrences']}"
        )


def cmd_similar(args: argparse.Namespace) -> None:
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Source state not found: {source_path}")
        return
    source_state = _load_state(source_path)
    if not source_state.get("string_scores"):
        print("Source state is missing 'string_scores'. Run 'stm ingest' with --store-profiles.")
        return

    target_state: Optional[Dict[str, Any]] = None
    target_state_path: Optional[Path] = None
    if args.target_state:
        target_state_path = Path(args.target_state)
        if not target_state_path.exists():
            print(f"Target state not found: {target_state_path}")
            return
        target_state = _load_state(target_state_path)

    ann_path = Path(args.target_ann) if args.target_ann else None
    meta_path: Optional[Path] = Path(args.target_meta) if args.target_meta else None
    if ann_path and meta_path is None:
        guess = ann_path.with_suffix(".meta")
        if guess.exists():
            meta_path = guess
        else:
            alt = ann_path.parent / (ann_path.stem + ".meta")
            if alt.exists():
                meta_path = alt

    try:
        result = cross_corpus_similarity(
            source_state,
            profile=args.profile,
            min_connector=args.min_connector,
            min_patternability=args.min_patternability,
            min_occurrences=args.min_occurrences,
            sort_key=args.sort,
            limit=args.limit,
            ann_index_path=ann_path if ann_path and meta_path else None,
            ann_meta_path=meta_path if ann_path and meta_path else None,
            target_state=target_state,
            k=args.k,
            max_distance=args.max_distance,
        )
    except Exception as exc:  # pragma: no cover - defensive CLI surface
        print(f"Error: {exc}")
        return

    payload = {
        "source": str(source_path),
        "target_ann": str(ann_path) if ann_path else None,
        "target_meta": str(meta_path) if meta_path else None,
        "target_state": str(target_state_path) if target_state_path else None,
        "profile": args.profile,
        "parameters": {
            "min_connector": args.min_connector,
            "min_patternability": args.min_patternability,
            "min_occurrences": args.min_occurrences,
            "sort": args.sort,
            "limit": args.limit,
            "k": args.k,
            "max_distance": args.max_distance,
        },
    }
    payload.update(result)

    output = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


def cmd_index_build(args: argparse.Namespace) -> None:
    state_path = Path(args.state)
    postings_path = Path(args.postings)
    ann_path = Path(args.ann)
    ann_meta_path = Path(args.ann_meta)
    try:
        build_indices(
            state_path=state_path,
            postings_path=postings_path,
            ann_path=ann_path,
            ann_meta_path=ann_meta_path,
            q=args.q,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Error building indices: {exc}")
        return
    print(
        "Index build complete:\n"
        f"  postings → {postings_path}\n  ann → {ann_path}\n  meta → {ann_meta_path}"
    )


def _top_strings_for_theme(
    state: Mapping[str, Any],
    theme_index: int,
    *,
    metric: str = "connector",
    limit: int = 5,
) -> List[str]:
    theme_members: List[List[str]] = state.get("themes", [])  # type: ignore[assignment]
    if theme_index < 0 or theme_index >= len(theme_members):
        raise ValueError(f"Theme index {theme_index} out of range")
    members = theme_members[theme_index]
    scores = state.get("string_scores", {})
    ranked: List[Tuple[float, str]] = []
    for text in members:
        payload = scores.get(text)
        if not payload:
            continue
        ranked.append((float(payload.get(metric, 0.0)), text))
    ranked.sort(reverse=True)
    return [text for _, text in ranked[:limit]]


def cmd_discover(args: argparse.Namespace) -> None:
    state = load_analysis_state(args.input)
    seeds: List[str]
    if args.seeds or args.seed_file:
        try:
            seeds = _collect_seeds(args)
        except ValueError as exc:
            print(f"Error: {exc}")
            return
    elif args.mode == "cross-theme" and args.theme_a is not None and args.theme_b is not None:
        theme_seed_count = args.theme_seed_count
        theme_a_seeds = _top_strings_for_theme(
            state, args.theme_a, metric=args.theme_metric, limit=theme_seed_count
        )
        theme_b_seeds = _top_strings_for_theme(
            state, args.theme_b, metric=args.theme_metric, limit=theme_seed_count
        )
        seeds = theme_a_seeds + theme_b_seeds
    else:
        raise ValueError(
            "Discover mode requires --seeds/--seed-file or --theme-a/--theme-b with mode cross-theme."
        )
    result = propose_from_state(
        state,
        seeds=seeds,
        k=args.k,
        min_connector=args.min_connector,
        min_patternability=args.min_patternability,
        target_profile=args.target_profile,
        )
    proposals = result["proposals"]
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Seeds: {', '.join(seeds)}")
    print(f"Generated {len(proposals)} proposals (showing up to {args.top}).")
    for idx, item in enumerate(proposals[: args.top], start=1):
        print(
            f"{idx:3d}. {item['string']:<24} score={item['score']:.3f}"
            f" pattern={item['patternability']:.3f} connector={item['connector']:.3f}"
            f" occ={item['occurrences']}"
        )


def cmd_dilution(args: argparse.Namespace) -> None:
    state = _load_state(Path(args.input))
    signals: List[Dict[str, Any]] = state.get("signals") or []  # type: ignore[assignment]
    if not signals:
        print(
            "State does not include per-window signals. Rerun 'stm ingest' with "
            "--store-signals to enable dilution analysis."
        )
        return
    token_mapping: Mapping[str, Any] = state.get("string_scores") or state.get("string_profiles") or {}
    if not token_mapping:
        print(
            "State file is missing string-to-window mappings. Include string scores or profiles "
            "when generating the analysis state."
        )
        return
    path_series, signal_series, semantic_score = compute_dilution_metrics(signals, token_mapping)  # type: ignore[arg-type]
    if not path_series and not signal_series:
        print("Unable to compute dilution metrics for the supplied state.")
        return
    window_info: List[Dict[str, Any]] = []
    for idx, sig in enumerate(signals):
        try:
            window_id = int(sig.get("id", idx))
        except (TypeError, ValueError):
            window_id = idx
        window_info.append(
            {
                "index": idx,
                "id": window_id,
                "start": sig.get("window_start"),
                "end": sig.get("window_end", sig.get("index")),
                "path": path_series[idx] if idx < len(path_series) else 0.0,
                "signal": signal_series[idx] if idx < len(signal_series) else 0.0,
            }
        )
    top_k = args.top or 10
    path_sorted = sorted(window_info, key=lambda item: item["path"], reverse=True)
    signal_sorted = sorted(window_info, key=lambda item: item["signal"], reverse=True)

    path_mean = mean(path_series) if path_series else 0.0
    signal_mean = mean(signal_series) if signal_series else 0.0
    path_median = median(path_series) if path_series else 0.0
    signal_median = median(signal_series) if signal_series else 0.0

    path_max = path_sorted[0]["path"] if path_sorted else 0.0
    signal_max = signal_sorted[0]["signal"] if signal_sorted else 0.0

    print("Dilution overview")
    print("-----------------")
    print(f"Windows analysed: {len(window_info)}")
    print(f"Path dilution mean={path_mean:.3f} median={path_median:.3f} max={path_max:.3f}")
    print(f"Signal dilution mean={signal_mean:.3f} median={signal_median:.3f} max={signal_max:.3f}")
    print(f"Semantic dilution={semantic_score:.3f} (clarity={max(0.0, min(1.0, 1.0 - semantic_score)):.3f})")

    def _print_top(label: str, items: List[Dict[str, Any]], key: str) -> None:
        print(f"\nTop windows by {label} dilution:")
        for entry in items[:top_k]:
            print(
                f"  window {entry['id']} (idx {entry['index']}): {key}={entry[key]:.3f}"
                f" range=({entry['start']},{entry['end']})"
            )

    if path_sorted:
        _print_top("path", path_sorted, "path")
    if signal_sorted:
        _print_top("signal", signal_sorted, "signal")

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="stm", description="Sep Text Manifold CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Ingest command
    p_ingest = subparsers.add_parser("ingest", help="Ingest and analyse a directory of text files")
    p_ingest.add_argument("directory", help="Directory containing text files (or raw file when using --adapter)")
    p_ingest.add_argument("--window-bytes", dest="window_bytes", type=int, default=2048, help="Size of sliding window in bytes")
    p_ingest.add_argument("--stride", dest="stride", type=int, default=1024, help="Stride between windows in bytes")
    p_ingest.add_argument("--extensions", nargs="*", help="Optional list of file extensions to include (e.g. txt md)")
    p_ingest.add_argument("--output", default="stm_state.json", help="Path to output analysis state JSON file")
    p_ingest.add_argument("--store-signals", action="store_true", help="Persist per-window metrics in the output state")
    p_ingest.add_argument("--store-occurrences", action="store_true", help="Persist raw string occurrences in the output state")
    p_ingest.add_argument("--store-profiles", action="store_true", help="Persist aggregated string profiles in the output state")
    p_ingest.add_argument("--summary-top", dest="summary_top", type=int, default=10, help="Number of top items to include in the summary output")
    p_ingest.add_argument("--min-token-len", dest="min_token_len", type=int, default=1, help="Minimum token length to include in scoring")
    p_ingest.add_argument("--alpha-ratio", dest="alpha_ratio", type=float, default=0.0, help="Minimum alphabetic ratio required for tokens")
    p_ingest.add_argument("--drop-numeric", action="store_true", help="Exclude purely numeric tokens")
    p_ingest.add_argument("--min-occ", dest="min_occ", type=int, default=1, help="Minimum occurrences per token to retain")
    p_ingest.add_argument("--cap-tokens-per-win", dest="cap_tokens_per_win", type=int, default=80, help="Maximum tokens per window considered for graph edges")
    p_ingest.add_argument("--graph-min-pmi", dest="graph_min_pmi", type=float, default=0.0, help="Minimum PMI required for graph edges")
    p_ingest.add_argument("--graph-max-degree", dest="graph_max_degree", type=int, help="Maximum degree allowed per node in the theme graph")
    p_ingest.add_argument("--theme-min-size", dest="theme_min_size", type=int, default=1, help="Minimum number of members required for a theme")
    p_ingest.add_argument("--log-file", dest="log_file", help="Optional path to append-only manifold log")
    p_ingest.add_argument("--verbose", action="store_true", help="Print progress information during analysis")
    p_ingest.add_argument("--adapter", help="Adapter name for preprocessing telemetry (e.g. nasa_themis)")
    p_ingest.add_argument("--keep-adapter-output", action="store_true", help="Preserve intermediate adapter artefacts")
    p_ingest.set_defaults(func=cmd_ingest)
    # Strings command
    p_strings = subparsers.add_parser("strings", help="List strings by patternability")
    p_strings.add_argument("--input", default="stm_state.json", help="Path to analysis state JSON file")
    p_strings.add_argument("--top", type=int, help="Number of top strings to display")
    p_strings.add_argument("--sort", choices=("pattern", "connector"), default="pattern", help="Sort key to use")
    p_strings.add_argument(
        "--filter",
        dest="filter",
        help="Metric filter expression, e.g. coh>=0.8,ent<=0.35",
    )
    p_strings.set_defaults(func=cmd_strings)
    # Themes command
    p_themes = subparsers.add_parser("themes", help="List detected themes")
    p_themes.add_argument("--input", default="stm_state.json", help="Path to analysis state JSON file")
    p_themes.set_defaults(func=cmd_themes)
    # Summary command
    p_summary = subparsers.add_parser("summary", help="Show aggregated analysis statistics")
    p_summary.add_argument("--input", default="stm_state.json", help="Path to analysis state JSON file")
    p_summary.add_argument("--top", type=int, help="Number of top strings/connectors to display")
    p_summary.set_defaults(func=cmd_summary)
    # Propose command
    p_propose = subparsers.add_parser("propose", help="Generate bridge-string proposals")
    p_propose.add_argument("--input", default="stm_state.json", help="Path to analysis state JSON file")
    p_propose.add_argument("--seeds", help="Comma-separated list of seed strings")
    p_propose.add_argument("--seed-file", help="File containing one seed string per line")
    p_propose.add_argument("--k", type=int, default=25, help="Number of proposals to compute")
    p_propose.add_argument("--top", type=int, default=10, help="Number of proposals to display")
    p_propose.add_argument("--min-connector", type=float, default=0.0, help="Minimum connector score filter")
    p_propose.add_argument("--min-patternability", type=float, default=0.0, help="Minimum patternability filter")
    p_propose.add_argument("--target-profile", help="Target metric profile constraints, e.g. coh>=0.7,ent<=0.3")
    p_propose.add_argument("--output", help="Optional path to write proposals JSON")
    p_propose.set_defaults(func=cmd_propose)
    # Dilution command
    p_dilution = subparsers.add_parser("dilution", help="Inspect path/signal/semantic dilution metrics")
    p_dilution.add_argument("--input", default="stm_state.json", help="Path to analysis state JSON file")
    p_dilution.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of windows to list per ranking",
    )
    p_dilution.set_defaults(func=cmd_dilution)
    # Similar command
    p_similar = subparsers.add_parser(
        "similar",
        help="Project high-quality strings from a source state into a target manifold via ANN or direct search",
    )
    p_similar.add_argument("--source", default="stm_state.json", help="Source analysis state JSON file")
    p_similar.add_argument("--target-ann", help="Target ANN index built via 'stm index build'")
    p_similar.add_argument(
        "--target-meta",
        help="Metadata JSON accompanying the ANN index (defaults to <target-ann>.meta or sibling .meta)",
    )
    p_similar.add_argument(
        "--target-state",
        help="Optional target state JSON. Required when ANN index is unavailable and enriches match output when present.",
    )
    p_similar.add_argument("--profile", help="Metric filter expression for source strings, e.g. coh>=0.8,ent<=0.35")
    p_similar.add_argument("--min-connector", type=float, default=0.0, help="Minimum connector score in source")
    p_similar.add_argument(
        "--min-patternability", type=float, default=0.0, help="Minimum patternability score in source"
    )
    p_similar.add_argument(
        "--min-occurrences", type=int, default=1, help="Minimum source occurrences required per string"
    )
    p_similar.add_argument(
        "--sort",
        choices=("patternability", "connector"),
        default="patternability",
        help="Sort key used to prioritise source strings",
    )
    p_similar.add_argument("--limit", type=int, help="Maximum number of source strings to project")
    p_similar.add_argument("--k", type=int, default=50, help="Number of target windows to retrieve per string")
    p_similar.add_argument(
        "--max-distance",
        type=float,
        help="Optional distance ceiling when retrieving neighbours (applies to ANN and fallback search)",
    )
    p_similar.add_argument("--output", help="Optional path to write JSON payload")
    p_similar.set_defaults(func=cmd_similar)
    # Discover command
    p_discover = subparsers.add_parser("discover", help="Assistive discovery workflow built on bridge proposals")
    p_discover.add_argument("--input", default="stm_state.json", help="Path to analysis state JSON file")
    p_discover.add_argument("--mode", choices=("cross-theme", "custom"), default="cross-theme", help="Discovery strategy")
    p_discover.add_argument("--theme-a", type=int, help="First theme index (0-based) for cross-theme mode")
    p_discover.add_argument("--theme-b", type=int, help="Second theme index (0-based) for cross-theme mode")
    p_discover.add_argument("--theme-seed-count", type=int, default=5, help="Number of strings to sample from each theme")
    p_discover.add_argument("--theme-metric", choices=("connector", "patternability"), default="connector", help="Ranking metric for theme seed selection")
    p_discover.add_argument("--seeds", help="Comma-separated seeds (overrides theme selection)")
    p_discover.add_argument("--seed-file", help="File containing seed strings (overrides theme selection)")
    p_discover.add_argument("--k", type=int, default=25, help="Number of proposals to compute")
    p_discover.add_argument("--top", type=int, default=10, help="Number of proposals to display")
    p_discover.add_argument("--min-connector", type=float, default=0.0, help="Minimum connector score filter")
    p_discover.add_argument("--min-patternability", type=float, default=0.0, help="Minimum patternability filter")
    p_discover.add_argument("--target-profile", help="Target metric profile constraints, e.g. coh>=0.7,ent<=0.3")
    p_discover.add_argument("--output", help="Optional path to write discovery JSON")
    p_discover.set_defaults(func=cmd_discover)
    # Index command
    p_index = subparsers.add_parser("index", help="Index management (postings/ANN)")
    index_sub = p_index.add_subparsers(dest="index_command", required=True)
    p_index_build = index_sub.add_parser("build", help="Build signature postings and ANN index")
    p_index_build.add_argument("--state", required=True, help="State JSON produced by 'stm ingest --store-signals'")
    p_index_build.add_argument("--postings", default="analysis/signature_postings.json", help="Output path for signature postings JSON")
    p_index_build.add_argument("--ann", default="analysis/ann.hnsw", help="Output path for ANN index file")
    p_index_build.add_argument("--ann-meta", default="analysis/ann.meta", help="Output path for ANN metadata JSON")
    p_index_build.add_argument("--q", type=int, default=3, help="Signature q-gram length")
    p_index_build.set_defaults(func=cmd_index_build)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
