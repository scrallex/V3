#!/usr/bin/env python3
"""Summarise semantic regime discovery runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank semantic regimes from discovery output")
    parser.add_argument("summary", type=Path, help="CSV emitted by semantic_regime_discovery")
    parser.add_argument(
        "--sort",
        default="sharpe",
        choices=["sharpe", "pnl", "return_pct", "win_rate"],
        help="Metric used to rank regimes (default: sharpe)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of regimes to emit per instrument",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="Minimum trades required to consider a regime",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _metric_value(row: Dict[str, str], key: str) -> float:
    raw = row.get(key)
    if raw is None or raw == "":
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _eligible(row: Dict[str, str], min_trades: int) -> bool:
    try:
        trades = int(float(row.get("trades", 0)))
    except ValueError:
        return False
    return trades >= min_trades


def _rank(rows: List[Dict[str, str]], key: str, top: int, min_trades: int) -> Dict[str, List[Tuple[str, float, float]]]:
    bucket: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)
    for row in rows:
        if not _eligible(row, min_trades):
            continue
        instrument = row.get("instrument", "").upper()
        regime = row.get("regime", "").strip()
        score = _metric_value(row, key)
        pnl = _metric_value(row, "pnl")
        if not instrument or not regime:
            continue
        bucket[instrument].append((regime, score, pnl))
    ranked: Dict[str, List[Tuple[str, float, float]]] = {}
    for instrument, entries in bucket.items():
        sorted_entries = sorted(entries, key=lambda item: item[1], reverse=True)
        ranked[instrument] = sorted_entries[:top]
    return ranked


def main() -> int:
    args = _parse_args()
    rows = _load_rows(args.summary)
    ranked = _rank(rows, args.sort, args.top, args.min_trades)
    for instrument, entries in ranked.items():
        print(f"Instrument: {instrument}")
        if not entries:
            print("  (no regimes met the criteria)")
            continue
        for idx, (regime, score, pnl) in enumerate(entries, start=1):
            score_label = "nan" if score != score else f"{score:.3f}"
            pnl_label = "nan" if pnl != pnl else f"{pnl:,.2f}"
            print(f"  {idx}. {regime} | {args.sort}={score_label} | pnl={pnl_label}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI tool
    raise SystemExit(main())
