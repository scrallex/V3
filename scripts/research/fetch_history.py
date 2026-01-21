#!/usr/bin/env python3
"""Fetch historical candles from OANDA and write to CSV."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trading.oanda import OandaConnector

GRANULARITY_SECONDS = {
    "S5": 5,
    "S10": 10,
    "S15": 15,
    "S30": 30,
    "M1": 60,
    "M2": 120,
    "M4": 240,
    "M5": 300,
    "M10": 600,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D": 86400,
    "W": 604800,
    "M": 2592000,
}


def isoformat(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse(dt: str) -> datetime:
    return datetime.fromisoformat(dt.replace("Z", "+00:00")).astimezone(timezone.utc)


def _load_oanda_env() -> None:
    """Load OANDA.env if the key/account vars are missing."""

    if os.getenv("OANDA_API_KEY") and os.getenv("OANDA_ACCOUNT_ID"):
        return
    env_path = ROOT / "OANDA.env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def iter_candles(
    connector: OandaConnector,
    instrument: str,
    granularity: str,
    start: datetime,
    end: datetime,
    step: timedelta,
) -> Iterable[Dict[str, Any]]:
    current = start
    gran_seconds = GRANULARITY_SECONDS.get(granularity, 60)
    while current < end:
        chunk_end = min(current + step, end)
        candles = connector.get_candles(
            instrument,
            granularity=granularity,
            from_time=isoformat(current),
            to_time=isoformat(chunk_end),
        )
        if not candles:
            current = chunk_end
            continue
        for candle in candles:
            yield candle
        last_time = candles[-1].get("time")
        if last_time:
            current = parse(last_time) + timedelta(seconds=gran_seconds)
        else:
            current = chunk_end


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    writer: Optional[csv.DictWriter[str]] = None
    fieldnames: Optional[List[str]] = None
    with path.open("w", newline="", encoding="utf-8") as fh:
        for row in rows:
            if writer is None:
                fieldnames = sorted(row.keys())
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
            elif fieldnames is not None and set(row.keys()) != set(fieldnames):
                # align row with known headers by adding missing keys
                row = {key: row.get(key) for key in fieldnames}
            writer.writerow(row)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Download historical candles from OANDA")
    parser.add_argument("instrument", help="Instrument, e.g. EUR_USD")
    parser.add_argument("start", help="Start time (ISO 8601, e.g. 2024-01-01T00:00:00Z)")
    parser.add_argument("end", help="End time (ISO 8601)")
    parser.add_argument("--granularity", default="M5", help="OANDA granularity (default M5)")
    parser.add_argument("--output", default="history.csv", help="Output CSV path")
    args = parser.parse_args()

    _load_oanda_env()
    start_dt = parse(args.start)
    end_dt = parse(args.end)
    if end_dt <= start_dt:
        raise SystemExit("end must be after start")

    connector = OandaConnector(read_only=True)
    step_seconds = GRANULARITY_SECONDS.get(args.granularity.upper(), 300) * 5000
    step = timedelta(seconds=step_seconds)
    rows_iter = iter_candles(connector, args.instrument, args.granularity.upper(), start_dt, end_dt, step)
    total = write_csv(Path(args.output), rows_iter)
    if total == 0:
        raise SystemExit("no data returned")
    print(f"Exported {total} candles to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
