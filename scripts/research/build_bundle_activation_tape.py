#!/usr/bin/env python3
"""Generate bundle activation tape from gate history."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.bundle_rules import BundleCatalog, iter_gate_records, write_activation_tape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bundle activation tape")
    parser.add_argument("--gates", default="docs/evidence/roc_history", help="Directory or file with gate JSONL exports")
    parser.add_argument("--bundle-config", default="config/bundle_strategy.yaml", help="Bundle definition file")
    parser.add_argument(
        "--output",
        default="output/strand_bundles/bundle_activation_tape.jsonl",
        help="Destination JSONL path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = BundleCatalog.load(Path(args.bundle_config))
    records = iter_gate_records(Path(args.gates))
    write_activation_tape(records, catalog, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
