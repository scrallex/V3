#!/usr/bin/env python3
"""
Compute manifold structural metrics for text transcripts referenced in a manifest.

Usage (from repository root):
    PYTHONPATH=score/src python scripts/experiments/run_structural_metrics.py \
        --manifest data/manifests/optical_structural/fox_subset.jsonl \
        --output output/metrics/fox_structural.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from sep_text_manifold import encode
from sep_text_manifold import native


def load_manifest(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_predictions(path: Path, field: str) -> Dict[Tuple[str, int], Dict[str, object]]:
    predictions: Dict[Tuple[str, int], Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["doc_id"]), int(record["page_id"]))
            predictions[key] = record
            if field not in record:
                raise KeyError(f"decoded field '{field}' missing for {key}")
    return predictions


def window_bytes(data: bytes, window: int, stride: int) -> Iterator[bytes]:
    n = len(data)
    if n == 0:
        return
    if n <= window:
        yield data
        return
    for start in range(0, n - window + 1, stride):
        yield data[start : start + window]
    tail_start = n - window
    if tail_start % stride != 0:
        yield data[tail_start:]


def aggregate_metrics(metrics_list: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    sums: Dict[str, float] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    count = float(len(metrics_list))
    return {key: sums[key] / count for key in sums}


def compute_structural_metrics(data: bytes, window_bytes_len: int, stride_bytes_len: int) -> Tuple[Dict[str, float], int]:
    metrics_list = [encode.encode_window(chunk) for chunk in window_bytes(data, window_bytes_len, stride_bytes_len)]
    return aggregate_metrics(metrics_list), len(metrics_list)


def subtract_metrics(reference: Dict[str, float], comparison: Dict[str, float]) -> Dict[str, float]:
    delta: Dict[str, float] = {}
    keys = set(reference) | set(comparison)
    for key in keys:
        delta[key] = float(comparison.get(key, 0.0)) - float(reference.get(key, 0.0))
    return delta


def process_record(
    record: Dict[str, object],
    window_bytes_len: int,
    stride_bytes_len: int,
    decoded_lookup: Dict[Tuple[str, int], Dict[str, object]] | None,
    decoded_field: str,
) -> Dict[str, object]:
    text_path = (Path.cwd() / Path(record["text_path"])).resolve()
    with text_path.open("r", encoding="utf-8") as fh:
        reference_bytes = fh.read().encode("utf-8")

    reference_metrics, reference_windows = compute_structural_metrics(reference_bytes, window_bytes_len, stride_bytes_len)

    decoded_section: Dict[str, object] | None = None
    delta_metrics: Dict[str, float] | None = None
    decoded_windows = 0

    if decoded_lookup:
        key = (str(record["doc_id"]), int(record["page_id"]))
        decoded_entry = decoded_lookup.get(key)
        if decoded_entry:
            decoded_text = str(decoded_entry[decoded_field])
            decoded_bytes = decoded_text.encode("utf-8")
            decoded_metrics, decoded_windows = compute_structural_metrics(decoded_bytes, window_bytes_len, stride_bytes_len)
            decoded_section = {
                "metrics": decoded_metrics,
                "windows": decoded_windows,
                "text": decoded_text,
                "metadata": {k: v for k, v in decoded_entry.items() if k not in {"doc_id", "page_id", decoded_field}},
            }
            delta_metrics = subtract_metrics(reference_metrics, decoded_metrics)

    result = {
        "doc_id": record["doc_id"],
        "page_id": record["page_id"],
        "reference": {"metrics": reference_metrics, "windows": reference_windows},
    }
    if decoded_section:
        result["decoded"] = decoded_section
    if delta_metrics is not None:
        result["delta"] = delta_metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute structural metrics for manifest entries")
    parser.add_argument("--manifest", type=Path, required=True, help="Input JSONL manifest")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL metrics file")
    parser.add_argument("--window-bytes", type=int, default=2048, help="Window size in bytes")
    parser.add_argument("--stride-bytes", type=int, default=1024, help="Stride in bytes")
    parser.add_argument("--use-native", action="store_true", help="Prefer the native manifold kernel when available")
    parser.add_argument("--decoded", type=Path, help="Optional JSONL with decoded OCR outputs")
    parser.add_argument(
        "--decoded-field",
        type=str,
        default="text",
        help="Field name in decoded JSONL that stores the OCR text (default: text)",
    )
    args = parser.parse_args()

    if args.use_native:
        native.set_use_native(True)

    manifest_records = load_manifest(args.manifest.resolve())
    decoded_lookup = None
    if args.decoded:
        decoded_lookup = load_predictions(args.decoded.resolve(), args.decoded_field)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for record in manifest_records:
            result = process_record(record, args.window_bytes, args.stride_bytes, decoded_lookup, args.decoded_field)
            fh.write(json.dumps(result, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
