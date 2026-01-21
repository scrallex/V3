#!/usr/bin/env python3
"""Aggregate regime ROC behaviour across horizons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

from scripts.research.roc_utils import iter_summary_files, load_summary, parse_week_label


def _aggregate(
    directory: Path,
    *,
    regimes: Sequence[str],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    totals: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"count": 0.0, "sum": 0.0, "positive": 0.0}))
    for summary_file in iter_summary_files(directory):
        payload = load_summary(summary_file)
        regime_block = payload.get("regimes") or {}
        horizons = payload.get("horizons") or []
        for regime in regimes:
            stats = regime_block.get(regime)
            if not isinstance(stats, Mapping):
                continue
            for horizon in horizons:
                horizon_stats = stats.get(str(horizon))
                if not isinstance(horizon_stats, Mapping):
                    continue
                count = float(horizon_stats.get("count") or 0.0)
                avg = float(horizon_stats.get("avg_roc_pct") or 0.0)
                positive = float(horizon_stats.get("positive_pct") or 0.0) * count
                agg = totals[regime][int(horizon)]
                agg["count"] += count
                agg["sum"] += avg * count
                agg["positive"] += positive
    return totals


def _finalise(totals: Mapping[str, Mapping[int, Mapping[str, float]]]) -> Dict[str, Dict[int, Dict[str, float]]]:
    output: Dict[str, Dict[int, Dict[str, float]]] = {}
    for regime, horizon_map in totals.items():
        output[regime] = {}
        for horizon, payload in horizon_map.items():
            count = payload["count"]
            if count <= 0:
                continue
            avg = payload["sum"] / count
            positive = payload["positive"] / count
            output[regime][horizon] = {
                "count": count,
                "avg_roc_pct": avg,
                "positive_pct": positive,
            }
    return output


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, payload: Mapping[str, Mapping[int, Mapping[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["regime", "horizon", "count", "avg_roc_pct", "positive_pct"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for regime, horizon_map in payload.items():
            for horizon, stats in sorted(horizon_map.items()):
                writer.writerow(
                    {
                        "regime": regime,
                        "horizon": horizon,
                        "count": stats["count"],
                        "avg_roc_pct": stats["avg_roc_pct"],
                        "positive_pct": stats["positive_pct"],
                    }
                )


def run(args: argparse.Namespace) -> int:
    regimes = [item.strip() for item in args.regimes.split(",") if item.strip()]
    totals = _aggregate(Path(args.roc_dir), regimes=regimes)
    result = _finalise(totals)
    _write_json(Path(args.output_json), result)
    if args.output_csv:
        _write_csv(Path(args.output_csv), result)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ROC performance across horizons")
    parser.add_argument("--roc-dir", default="docs/evidence/roc_history")
    parser.add_argument("--regimes", default="mean_revert,neutral,chaotic")
    parser.add_argument("--output-json", default="docs/evidence/roc_horizon_rollup.json")
    parser.add_argument("--output-csv", default="docs/evidence/roc_horizon_rollup.csv")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
