#!/usr/bin/env python3
"""
Demonstrate hazard-gated manifold verification inspired by PortfolioManager/STM.

For each document in the corpus we:
 1. Build the compressed manifold (signature -> count + mean hazard λ).
 2. Derive a hazard threshold (default: 80th percentile) per document.
 3. Replay both in-document windows (positives) and cross-document windows (negatives)
    to measure acceptance/rejection under the hazard gate.

The output summarises acceptance rates, precision/recall, and the effective hazard
thresholds for downstream integration with PortfolioManager.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from manifold_compression_eval import build_compressed_representation, normalise_compressed


@dataclass
class GateStats:
    threshold: float
    positives: int = 0
    accepted_positives: int = 0
    negatives: int = 0
    rejected_negatives: int = 0

    @property
    def recall(self) -> float:
        return (self.accepted_positives / self.positives) if self.positives else 0.0

    @property
    def false_positive_rate(self) -> float:
        violations = self.negatives - self.rejected_negatives
        return (violations / self.negatives) if self.negatives else 0.0

    @property
    def precision(self) -> float:
        violations = self.negatives - self.rejected_negatives
        denom = self.accepted_positives + violations
        return (self.accepted_positives / denom) if denom else 1.0


def compute_gate_thresholds(
    compressed: Dict[str, Dict[str, Dict[str, float]]],
    percentile: float,
) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for doc_id, signatures in compressed.items():
        hazards = sorted(entry["hazard"] for entry in signatures.values())
        if not hazards:
            thresholds[doc_id] = 1.0
            continue
        index = max(0, min(int(len(hazards) * percentile / 100.0), len(hazards) - 1))
        thresholds[doc_id] = hazards[index]
    return thresholds


def replay_windows(
    text_root: Path,
    window_bytes: int,
    stride_bytes: int,
    precision: int,
    percentile: float,
) -> Dict[str, GateStats]:
    compressed_raw, doc_windows, _, _, _ = build_compressed_representation(
        text_root=text_root,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=precision,
    )
    compressed = normalise_compressed(compressed_raw)
    doc_signatures = {doc_id: set(bucket.keys()) for doc_id, bucket in compressed.items()}
    thresholds = compute_gate_thresholds(compressed, percentile=percentile)

    stats: Dict[str, GateStats] = {doc_id: GateStats(threshold=thresholds[doc_id]) for doc_id in doc_signatures}

    for doc_id, windows in doc_windows.items():
        stat = stats[doc_id]
        for record in windows:
            stat.positives += 1
            entry = compressed[doc_id].get(record.signature)
            if entry and record.metrics["lambda_hazard"] <= stat.threshold:
                stat.accepted_positives += 1

    for doc_id, stat in stats.items():
        signatures = compressed[doc_id]
        for other_id, other_signatures in compressed.items():
            if doc_id == other_id:
                continue
            for signature, metrics in other_signatures.items():
                hazard = metrics["hazard"]
                stat.negatives += 1
                if signature not in signatures or hazard > stat.threshold:
                    stat.rejected_negatives += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Hazard-gated manifold verification demo")
    parser.add_argument("--text-root", type=Path, required=True, help="Root directory of UTF-8 text files")
    parser.add_argument("--window-bytes", type=int, default=256, help="Sliding window size")
    parser.add_argument("--stride-bytes", type=int, default=192, help="Sliding window stride")
    parser.add_argument("--precision", type=int, default=3, help="Signature precision")
    parser.add_argument(
        "--hazard-percentile",
        type=float,
        default=80.0,
        help="Percentile of per-document hazard used as acceptance threshold",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON report")
    args = parser.parse_args()

    text_root = args.text_root.resolve()
    if not text_root.exists():
        raise FileNotFoundError(f"text root not found: {text_root}")

    stats = replay_windows(
        text_root=text_root,
        window_bytes=args.window_bytes,
        stride_bytes=args.stride_bytes,
        precision=args.precision,
        percentile=args.hazard_percentile,
    )

    overall_positives = sum(stat.positives for stat in stats.values())
    overall_accepted = sum(stat.accepted_positives for stat in stats.values())
    overall_negatives = sum(stat.negatives for stat in stats.values())
    overall_rejected = sum(stat.rejected_negatives for stat in stats.values())
    overall_precision = overall_accepted / (overall_accepted + (overall_negatives - overall_rejected)) if overall_negatives else 1.0
    overall_recall = overall_accepted / overall_positives if overall_positives else 0.0
    overall_fpr = (overall_negatives - overall_rejected) / overall_negatives if overall_negatives else 0.0

    report = {
        "text_root": str(text_root),
        "window_bytes": args.window_bytes,
        "stride_bytes": args.stride_bytes,
        "precision": args.precision,
        "hazard_percentile": args.hazard_percentile,
        "overall": {
            "positives": overall_positives,
            "accepted_positives": overall_accepted,
            "negatives": overall_negatives,
            "rejected_negatives": overall_rejected,
            "precision": overall_precision,
            "recall": overall_recall,
            "false_positive_rate": overall_fpr,
        },
        "per_document": {
            doc_id: {
                "threshold": stat.threshold,
                "positives": stat.positives,
                "accepted_positives": stat.accepted_positives,
                "negatives": stat.negatives,
                "rejected_negatives": stat.rejected_negatives,
                "precision": stat.precision,
                "recall": stat.recall,
                "false_positive_rate": stat.false_positive_rate,
            }
            for doc_id, stat in stats.items()
        },
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
