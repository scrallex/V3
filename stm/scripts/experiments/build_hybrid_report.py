#!/usr/bin/env python3
"""Join manifest, optical predictions, and structural metrics into a single report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def build_index(path: Path, key_fields):
    index: Dict[Tuple[str, int], Dict[str, object]] = {}
    for record in load_jsonl(path):
        key = (str(record[key_fields[0]]), int(record[key_fields[1]]))
        index[key] = record
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid optical/structural report")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--optical", type=Path, required=True)
    parser.add_argument("--structural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_records = list(load_jsonl(args.manifest))
    optical_index = build_index(args.optical, ("doc_id", "page_id"))
    structural_index = build_index(args.structural, ("doc_id", "page_id"))

    report = []
    for record in manifest_records:
        key = (str(record["doc_id"]), int(record["page_id"]))
        optical = optical_index.get(key)
        structural = structural_index.get(key)
        report.append(
            {
                "doc_id": key[0],
                "page_id": key[1],
                "image_path": record["image_path"],
                "text_path": record.get("text_path"),
                "optical": optical,
                "structural": structural,
            }
        )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
