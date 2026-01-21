#!/usr/bin/env python3
"""Run a sweep over manifold compression parameters and record summaries."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import List

from manifold_compression_eval import evaluate_manifold


def parse_int_list(value: str) -> List[int]:
    if not value:
        raise argparse.ArgumentTypeError("empty list")
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep manifold compression parameters")
    parser.add_argument("--text-root", type=Path, required=True, help="Root directory of UTF-8 text files")
    parser.add_argument(
        "--precisions",
        type=parse_int_list,
        default=[1, 2, 3],
        help="Comma-separated precisions (default: 1,2,3)",
    )
    parser.add_argument(
        "--windows",
        type=parse_int_list,
        default=[128, 256, 512],
        help="Comma-separated window sizes in bytes (default: 128,256,512)",
    )
    parser.add_argument(
        "--strides",
        type=parse_int_list,
        default=[96, 192, 256],
        help="Comma-separated stride sizes in bytes (default: 96,192,256)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file summarising the sweep")
    parser.add_argument("--use-native", action="store_true", help="Prefer the native manifold kernel if available")
    args = parser.parse_args()

    text_root = args.text_root.resolve()
    if not text_root.exists():
        raise FileNotFoundError(f"text root not found: {text_root}")

    results = []
    for precision, window_bytes, stride_bytes in product(args.precisions, args.windows, args.strides):
        if stride_bytes > window_bytes:
            continue  # invalid configuration
        summary = evaluate_manifold(
            text_root=text_root,
            window_bytes=window_bytes,
            stride_bytes=stride_bytes,
            precision=precision,
            use_native=args.use_native,
        )
        results.append(summary)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps({"runs": len(results), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
