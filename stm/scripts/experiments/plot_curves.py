#!/usr/bin/env python3
"""Plot compression vs. accuracy curves from benchmark summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(path: Path) -> pd.DataFrame:
    raw: Dict[str, Dict[str, object]] = json.loads(path.read_text())
    rows: List[Dict[str, object]] = []
    for label, payload in raw.items():
        rows.append(
            {
                "label": label,
                "documents": payload["documents"],
                "compression_ratio": payload["compression_ratio"],
                "token_compression": payload["token_metrics"]["token_compression_unique"],
                "token_accuracy": payload["token_metrics"]["token_accuracy"],
                "character_accuracy": payload["character_metrics"]["character_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    df.sort_values("token_compression", ascending=True, inplace=True)
    ax.plot(
        df["token_compression"],
        df["token_accuracy"] * 100.0,
        marker="o",
        linewidth=2,
        label="Structural Manifold",
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["label"],
            (row["token_compression"], row["token_accuracy"] * 100.0),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    ax.set_xlabel("Token compression (×)")
    ax.set_ylabel("Token accuracy (%)")
    ax.set_title("Compression vs. Accuracy")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(88, 100)
    ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compression curves from summary JSON.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to summary.json produced by benchmark_eval.py")
    parser.add_argument("--output", type=Path, required=True, help="Destination image path")
    args = parser.parse_args()
    df = load_summary(args.summary)
    plot(df, args.output)


if __name__ == "__main__":
    main()
