#!/usr/bin/env python3
"""Export gate payloads from Valkey to a CSV/JSONL file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

import redis


def iter_gate_last(client: redis.Redis) -> Iterable[Dict[str, Any]]:
    for key in client.scan_iter(match="gate:last:*"):
        raw = client.get(key)
        if not raw:
            continue
        try:
            payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            continue
        payload["__key"] = key.decode() if isinstance(key, bytes) else str(key)
        yield payload


def iter_gate_index(client: redis.Redis) -> Iterable[Dict[str, Any]]:
    for key in client.scan_iter(match="gate:index:*"):
        for member, score in client.zscan_iter(key):
            try:
                payload = json.loads(member)
            except Exception:
                continue
            payload["__key"] = key.decode() if isinstance(key, bytes) else str(key)
            payload.setdefault("ts_ms", score)
            yield payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Export gate payloads from Valkey")
    parser.add_argument("output", help="Output file (CSV if endswith .csv, else JSONL)")
    parser.add_argument("--redis", default="redis://localhost:6379/0", help="Valkey URL")
    args = parser.parse_args()

    client = redis.from_url(args.redis)
    rows = list(iter_gate_index(client))
    if not rows:
        rows = list(iter_gate_last(client))
    if not rows:
        print("No gate payloads found", file=sys.stderr)
        return 1

    if args.output.lower().endswith(".csv"):
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(args.output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row))
                fh.write("\n")
    print(f"Exported {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
