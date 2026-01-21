#!/usr/bin/env python3
"""Simulate bundle trades from gate history."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.bundle_rules import BundleCatalog, iter_gate_records, nearest_horizon
from scripts.tools.bundle_outcome_study import _direction, _roc_value, DEFAULT_HORIZONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate bundle trades")
    parser.add_argument("--gates", default="docs/evidence/roc_history", help="Gate directory or JSONL file")
    parser.add_argument("--bundle-config", default="config/bundle_strategy.yaml")
    parser.add_argument("--output", default="output/strand_bundles/bundle_trades.csv")
    parser.add_argument("--cost-bp", type=float, default=12.0)
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    horizons = [int(item.strip()) for item in args.horizons.split(",") if item.strip()]
    catalog = BundleCatalog.load(Path(args.bundle_config))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ts_ms",
                "instrument",
                "bundle_id",
                "action",
                "hold_minutes",
                "horizon",
                "bp",
            ],
        )
        writer.writeheader()
        for record in iter_gate_records(Path(args.gates)):
            hits, blocks, _ = catalog.evaluate_record(record)
            if not hits or blocks:
                continue
            roc_payload = record.get("roc_forward_pct")
            if not isinstance(roc_payload, dict):
                continue
            for hit in hits:
                direction = _direction(hit.rule.action)
                if direction == 0:
                    continue
                horizon = nearest_horizon(hit.rule.hold_minutes, horizons)
                if horizon is None:
                    continue
                roc_value = _roc_value(record, horizon)
                if roc_value is None:
                    continue
                bp = roc_value * 10_000.0 * direction - args.cost_bp
                writer.writerow(
                    {
                        "ts_ms": record.get("ts_ms"),
                        "instrument": record.get("instrument"),
                        "bundle_id": hit.rule.bundle_id,
                        "action": hit.rule.action,
                        "hold_minutes": hit.rule.hold_minutes,
                        "horizon": horizon,
                        "bp": round(bp, 5),
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
