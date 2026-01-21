#!/usr/bin/env python3
"""Run manifold compression evaluation across benchmark datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.manifold_compression_eval import evaluate_manifold  # noqa: E402


@dataclass
class DatasetConfig:
    label: str
    text_root: Path


def parse_dataset_args(dataset_args: Iterable[str]) -> List[DatasetConfig]:
    configs: List[DatasetConfig] = []
    for arg in dataset_args:
        if "=" not in arg:
            raise ValueError(f"Dataset argument must be LABEL=PATH, received: {arg}")
        label, path_str = arg.split("=", 1)
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Text root not found for dataset '{label}': {path}")
        configs.append(DatasetConfig(label=label, text_root=path))
    if not configs:
        raise ValueError("At least one --dataset entry is required.")
    return configs


def flatten_summary(label: str, summary: Dict[str, object]) -> Dict[str, object]:
    token_metrics = summary.get("token_metrics", {})
    character_metrics = summary.get("character_metrics", {})
    verification = summary.get("verification", {})
    return {
        "label": label,
        "documents": summary.get("documents"),
        "window_bytes": summary.get("window_bytes"),
        "stride_bytes": summary.get("stride_bytes"),
        "precision": summary.get("precision"),
        "tokenizer": summary.get("tokenizer_name"),
        "compression_ratio": summary.get("compression_ratio"),
        "token_accuracy": token_metrics.get("token_accuracy"),
        "token_precision": token_metrics.get("token_precision"),
        "token_recall": token_metrics.get("token_recall"),
        "token_f1": token_metrics.get("token_f1"),
        "token_compression_unique": token_metrics.get("token_compression_unique"),
        "token_compression_stream": token_metrics.get("token_compression_stream"),
        "character_accuracy": character_metrics.get("character_accuracy"),
        "normalized_edit_distance": character_metrics.get("normalized_edit_distance"),
        "verification_precision": verification.get("precision"),
        "verification_fpr": verification.get("false_positive_rate"),
    }


def write_csv(path: Path, records: Iterable[Dict[str, object]]) -> None:
    records = list(records)
    if not records:
        return
    fieldnames = list(records[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch manifold compression benchmark evaluation.")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset specification in the form LABEL=TEXT_ROOT. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "benchmark_runs")
    parser.add_argument("--window-bytes", type=int, default=512)
    parser.add_argument("--stride-bytes", type=int, default=384)
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument(
        "--tokenizer-trust-remote-code",
        action="store_true",
        help="Allow remote code when loading the tokenizer (required for some custom tokenizers).",
    )
    parser.add_argument(
        "--json-text-key",
        type=str,
        default="text",
        help="Field name to read when ingesting JSON/JSONL corpora (default: text).",
    )
    parser.add_argument("--use-native", action="store_true", help="Use the native manifold kernel if available.")
    parser.add_argument("--max-documents", type=int, help="Optional cap on number of documents per dataset.")
    parser.add_argument(
        "--document-offset",
        type=int,
        default=0,
        help="Skip the first N documents for each dataset before processing.",
    )
    args = parser.parse_args()

    configs = parse_dataset_args(args.dataset or [])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    flat_rows: List[Dict[str, object]] = []
    combined: Dict[str, Dict[str, object]] = {}

    for config in configs:
        summary = evaluate_manifold(
            text_root=config.text_root,
            window_bytes=args.window_bytes,
            stride_bytes=args.stride_bytes,
            precision=args.precision,
            tokenizer_name=args.tokenizer,
            tokenizer_trust_remote_code=args.tokenizer_trust_remote_code,
            max_documents=args.max_documents,
            use_native=args.use_native,
            json_text_key=args.json_text_key,
            document_offset=args.document_offset,
        )
        json_path = args.output_dir / f"{config.label}.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        flat_rows.append(flatten_summary(config.label, summary))
        combined[config.label] = summary

    write_csv(args.output_dir / "summary.csv", flat_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
