#!/usr/bin/env python3
"""Regenerate span gate exports using offline candle CSVs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable
from multiprocessing import Pool, cpu_count

REPO_ROOT = Path(__file__).resolve().parents[2]

os.environ.setdefault("DISABLE_REGIME_METRICS", "1")
sys.path.insert(0, str(REPO_ROOT))
from scripts.tools import backfill_gate_history  # noqa: E402


def _load_spans(span_catalog: Path) -> Iterable[dict]:
    payload = json.loads(span_catalog.read_text(encoding="utf-8"))
    return payload.get("spans", [])


def _run(arg_list) -> None:
    print(f"[cmd] backfill_gate_history {' '.join(arg_list)}")
    result = backfill_gate_history.main(arg_list)
    if result != 0:
        raise RuntimeError("backfill_gate_history returned non-zero status")


def _process_instrument_span(args_tuple):
    """Worker function for parallel processing."""
    instrument, span, mode, target_dir, extra, candles_dir, profile = args_tuple
    span_id = span["span_id"]
    start = span["start"]
    end = span["end"]
    
    export_json = target_dir / f"gates_{span_id}_{instrument}.jsonl"
    export_summary = target_dir / f"roc_summary_{span_id}_{instrument}.json"
    
    arg_list = [
        "--start",
        start,
        "--end",
        end,
        "--profile",
        str(profile),
        "--candles-dir",
        str(candles_dir),
        "--export-json",
        str(export_json),
        "--export-roc-summary",
        str(export_summary),
        "--granularity",
        "M1",
        "--roc-horizons",
        "5,15,30,60,240",
        "--instruments",
        instrument,
    ]
    arg_list.extend(extra)
    
    try:
        _run(arg_list)
        return f"{span_id}:{instrument}:{mode} OK"
    except Exception as e:
        return f"{span_id}:{instrument}:{mode} FAILED: {e}"


def run_backfills(
    spans: Iterable[dict],
    *,
    candles_dir: Path,
    profile: Path,
    gameplan_root: Path,
    instruments: str | None,
    selected_spans: set[str] | None,
    workers: int,
) -> None:
    subsets_dir = gameplan_root / "subsets"
    isolation_dir = gameplan_root / "isolation"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    isolation_dir.mkdir(parents=True, exist_ok=True)

    # Build instrument list
    from scripts.trading.portfolio_manager import StrategyProfile
    strategy_profile = StrategyProfile.load(profile)
    if instruments:
        inst_list = [item.strip().upper() for item in instruments.split(",") if item.strip()]
    else:
        inst_list = sorted(strategy_profile.instruments.keys())

    for span in spans:
        span_id = span["span_id"]
        if selected_spans and span_id not in selected_spans:
            continue
        
        print(f"\n[span] {span_id} ({span['start']} -> {span['end']}) - processing {len(inst_list)} instruments")

        for mode, target_dir, extra in (
            ("rolling", subsets_dir, []),
            ("isolation", isolation_dir, ["--isolation"]),
        ):
            print(f"[span] {span_id} mode={mode} - starting parallel processing with {workers} workers", flush=True)
            
            # Build work items for parallel processing
            work_items = [
                (inst, span, mode, target_dir, extra, candles_dir, profile)
                for inst in inst_list
            ]
            
            if workers > 1:
                with Pool(processes=workers) as pool:
                    results = pool.map(_process_instrument_span, work_items)
                for result in results:
                    print(f"  {result}")
            else:
                # Sequential fallback
                results = [_process_instrument_span(item) for item in work_items]
                for result in results:
                    print(f"  {result}")
            
            # Merge instrument-specific outputs into span-level files
            merged_gates = target_dir / f"gates_{span_id}.jsonl"
            merged_summary = target_dir / f"roc_summary_{span_id}.json"
            
            _merge_outputs(target_dir, span_id, inst_list, merged_gates, merged_summary)
            
            print(f"[span] {span_id} mode={mode} completed and merged")


def _merge_outputs(target_dir: Path, span_id: str, instruments: list[str], 
                   merged_gates: Path, merged_summary: Path) -> None:
    """Merge per-instrument outputs into span-level files."""
    import json
    
    # Merge JSONL gate files
    with merged_gates.open("w", encoding="utf-8") as out:
        for inst in instruments:
            inst_gates = target_dir / f"gates_{span_id}_{inst}.jsonl"
            if inst_gates.exists():
                out.write(inst_gates.read_text(encoding="utf-8"))
                inst_gates.unlink()  # Clean up
    
    # Merge ROC summaries
    combined_summary = {
        "generated_at": None,
        "horizons": [],
        "regimes": {},
    }
    
    for inst in instruments:
        inst_summary = target_dir / f"roc_summary_{span_id}_{inst}.json"
        if inst_summary.exists():
            data = json.loads(inst_summary.read_text(encoding="utf-8"))
            if not combined_summary["generated_at"]:
                combined_summary["generated_at"] = data.get("generated_at")
                combined_summary["horizons"] = data.get("horizons", [])
            
            # Merge regime stats
            for regime, horizon_data in data.get("regimes", {}).items():
                if regime not in combined_summary["regimes"]:
                    combined_summary["regimes"][regime] = {}
                
                for horizon, stats in horizon_data.items():
                    if horizon not in combined_summary["regimes"][regime]:
                        combined_summary["regimes"][regime][horizon] = {
                            "count": 0,
                            "avg_roc_pct": 0.0,
                            "positive_pct": 0.0,
                        }
                    
                    existing = combined_summary["regimes"][regime][horizon]
                    new_count = existing["count"] + stats["count"]
                    existing["avg_roc_pct"] = (
                        (existing["avg_roc_pct"] * existing["count"] + stats["avg_roc_pct"] * stats["count"]) 
                        / new_count
                    )
                    existing["positive_pct"] = (
                        (existing["positive_pct"] * existing["count"] + stats["positive_pct"] * stats["count"]) 
                        / new_count
                    )
                    existing["count"] = new_count
            
            inst_summary.unlink()  # Clean up
    
    merged_summary.write_text(json.dumps(combined_summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate span gates from offline candle CSVs")
    parser.add_argument("--candles-dir", required=True, help="Directory with candle CSVs (from fetch_history.py)")
    parser.add_argument("--gameplan-root", default="docs/evidence/roc_history/gameplan", help="Gameplan root directory")
    parser.add_argument("--span-catalog", default="docs/evidence/roc_history/gameplan/span_catalog.json", help="Span catalog JSON")
    parser.add_argument("--profile", default="config/echo_strategy.yaml", help="Strategy profile path")
    parser.add_argument("--instruments", help="Optional comma separated instrument list override")
    parser.add_argument(
        "--span-id",
        action="append",
        help="Span ID to process (can be provided multiple times or comma separated)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, cpu_count()),
        help="Number of parallel workers (default: min(4, cpu_count))",
    )
    args = parser.parse_args()

    spans = list(_load_spans(Path(args.span_catalog)))
    if not spans:
        raise SystemExit("No spans found in catalog")
    selected = set()
    if args.span_id:
        for chunk in args.span_id:
            for item in chunk.split(","):
                item = item.strip()
                if item:
                    selected.add(item)

    run_backfills(
        spans,
        candles_dir=Path(args.candles_dir),
        profile=Path(args.profile),
        gameplan_root=Path(args.gameplan_root),
        instruments=args.instruments,
        selected_spans=selected or None,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
