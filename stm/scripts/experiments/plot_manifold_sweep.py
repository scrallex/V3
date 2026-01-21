#!/usr/bin/env python3
"""
Plot compression ratio versus false-positive rate from manifold sweep results.

Usage:
    python scripts/experiments/plot_manifold_sweep.py \
        --input output/manifold_compression_corpus_sweep.jsonl \
        --output output/manifold_sweep_plot.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_results(path: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                results.append(json.loads(line))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compression vs FPR from manifold sweep")
    parser.add_argument("--input", type=Path, required=True, help="JSONL file produced by manifold_compression_grid.py")
    parser.add_argument("--output", type=Path, required=True, help="Destination PNG path")
    args = parser.parse_args()

    results = load_results(args.input.resolve())
    if not results:
        raise RuntimeError("No sweep results found in input file")

    grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for result in results:
        precision = int(result["precision"])
        compression_ratio = float(result["compression_ratio"])
        fpr = float(result["verification"]["false_positive_rate"])
        window = int(result["window_bytes"])
        stride = int(result["stride_bytes"])
        grouped[precision].append(
            {
                "compression_ratio": compression_ratio,
                "false_positive_rate": fpr,
                "window": window,
                "stride": stride,
            }
        )

    plt.figure(figsize=(10, 6))
    for precision, entries in sorted(grouped.items()):
        entries_sorted = sorted(entries, key=lambda x: x["compression_ratio"])
        ratios = [item["compression_ratio"] for item in entries_sorted]
        fprs = [item["false_positive_rate"] for item in entries_sorted]
        labels = [f"W{item['window']}/S{item['stride']}" for item in entries_sorted]
        plt.scatter(ratios, fprs, label=f"precision={precision}")
        for ratio, fpr, label in zip(ratios, fprs, labels):
            plt.annotate(label, (ratio, fpr), textcoords="offset points", xytext=(4, 4), fontsize=8)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Compression Ratio (log scale)")
    plt.ylabel("False Positive Rate (log scale)")
    plt.title("Manifold Compression Sweep: Ratio vs False Positive Rate")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(json.dumps({"output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
