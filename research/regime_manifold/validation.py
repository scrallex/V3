#!/usr/bin/env python3
"""Validation harness for the market regime manifold codec.

Usage examples:

    python research/regime_manifold/validation.py \\
        --source csv:data/EUR_USD_M1.csv --instrument EUR_USD

    python research/regime_manifold/validation.py \\
        --source valkey:EUR_USD:2024-10-01:2024-10-07 --redis ${VALKEY_URL}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None

from scripts.trading.candle_utils import candle_from_payload
from .codec import Candle, MarketManifoldCodec, window_summary, windows_to_jsonl


def _load_from_csv(path: Path) -> List[Candle]:
    candles: List[Candle] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            candles.append(candle_from_payload(row))
    candles.sort(key=lambda c: c.timestamp_ms)
    return candles


def _load_from_valkey(spec: str, redis_url: str) -> List[Candle]:
    if redis is None:
        raise RuntimeError("redis package is not installed; cannot load from Valkey")
    _, instrument, start_s, end_s = spec.split(":", 3)
    start = datetime.fromisoformat(start_s).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(end_s).replace(tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    key = f"md:candles:{instrument.upper()}:M1"
    client = redis.from_url(redis_url)
    rows = client.zrangebyscore(key, start_ms, end_ms)
    candles: List[Candle] = []
    for raw in rows:
        try:
            payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
        except Exception:
            continue
        try:
            candles.append(candle_from_payload(payload))
        except Exception:
            continue
    candles.sort(key=lambda c: c.timestamp_ms)
    return candles


def _regime_breakdown(windows):
    table: Dict[str, Dict[str, float]] = {}
    for window in windows:
        row = table.setdefault(window.canonical.regime, {"count": 0, "hazard_avg": 0.0, "trend_avg": 0.0})
        row["count"] += 1
        row["hazard_avg"] += window.metrics["hazard"]
        row["trend_avg"] += window.canonical.trend_strength
    for stats in table.values():
        if stats["count"]:
            stats["hazard_avg"] /= stats["count"]
            stats["trend_avg"] /= stats["count"]
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate market regime manifolds")
    parser.add_argument("--source", required=True, help="csv:/path/to/file or valkey:INSTRUMENT:YYYY-MM-DD:YYYY-MM-DD")
    parser.add_argument("--instrument", required=True, help="Instrument symbol (e.g., EUR_USD)")
    parser.add_argument("--redis", default="redis://localhost:6379/0", help="Valkey URL for valkey source")
    parser.add_argument("--window", type=int, default=64, help="Candles per window")
    parser.add_argument("--stride", type=int, default=16, help="Stride between windows")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR smoothing period")
    parser.add_argument("--dump-jsonl", help="Optional path to dump encoded windows")
    args = parser.parse_args()

    if args.source.startswith("csv:"):
        path = Path(args.source.split(":", 1)[1])
        candles = _load_from_csv(path)
    elif args.source.startswith("valkey:"):
        candles = _load_from_valkey(args.source, args.redis)
    else:
        raise SystemExit("Unsupported source; use csv: or valkey:")

    if not candles:
        raise SystemExit("No candles loaded for validation run")

    codec = MarketManifoldCodec(
        window_candles=args.window,
        stride_candles=args.stride,
        atr_period=args.atr_period,
    )
    windows = codec.encode(candles, instrument=args.instrument.upper())
    if not windows:
        raise SystemExit("Insufficient data to build manifolds")

    summary = window_summary(windows)
    breakdown = _regime_breakdown(windows)

    print(json.dumps({"summary": summary, "regimes": breakdown}, indent=2))

    if args.dump_jsonl:
        Path(args.dump_jsonl).write_text(windows_to_jsonl(windows), encoding="utf-8")
        print(f"Dumped {len(windows)} windows to {args.dump_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
