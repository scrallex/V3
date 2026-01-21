#!/usr/bin/env python3
"""Analyse enriched strands that include semantic tags and structural slopes."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from scripts.research.roc_utils import (
    iter_gate_files,
    load_gate_records,
    parse_week_label,
    roc_value,
    semantic_primary,
    slope_bucket,
)


def _update_stats(bucket: Dict[str, float], horizon_values: Dict[int, float | None], horizons: Sequence[int]) -> None:
    bucket["count"] = bucket.get("count", 0) + 1
    for horizon in horizons:
        value = horizon_values.get(horizon)
        if value is None:
            continue
        key_sum = f"sum_{horizon}"
        key_pos = f"pos_{horizon}"
        bucket[key_sum] = bucket.get(key_sum, 0.0) + float(value)
        if value > 0:
            bucket[key_pos] = bucket.get(key_pos, 0.0) + 1.0


def _record_horizon_values(record: Mapping[str, object], horizons: Sequence[int]) -> Dict[int, float | None]:
    return {h: roc_value(record, h) for h in horizons}


def _aggregate(
    gates_dir: Path,
    *,
    horizons: Sequence[int],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    base_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
    enriched_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
    for gate_file in iter_gate_files(gates_dir):
        parse_week_label(gate_file)  # ensure ordering (unused result)
        for record in load_gate_records(gate_file):
            strand = record.get("strand_id")
            if not strand or not isinstance(strand, str):
                continue
            horizon_values = _record_horizon_values(record, horizons)
            base_bucket = base_stats[strand]
            _update_stats(base_bucket, horizon_values, horizons)
            structure = record.get("structure") or {}
            sem_tag = semantic_primary(record.get("semantic_tags", []))
            coh_bucket = slope_bucket(structure.get("coherence_tau_slope"))
            domain_bucket = slope_bucket(structure.get("domain_wall_slope"))
            enriched_id = f"{strand}|{sem_tag}|coh={coh_bucket}|dw={domain_bucket}"
            enriched_bucket = enriched_stats[enriched_id]
            enriched_bucket["parent"] = strand
            enriched_bucket["semantic_tag"] = sem_tag
            enriched_bucket["coherence_bucket"] = coh_bucket
            enriched_bucket["domain_bucket"] = domain_bucket
            _update_stats(enriched_bucket, horizon_values, horizons)
    return base_stats, enriched_stats


def _normalise_stats(
    stats: Mapping[str, Mapping[str, float]],
    *,
    horizons: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    for key, payload in stats.items():
        count = float(payload.get("count", 0.0) or 0.0)
        if count <= 0:
            continue
        row: Dict[str, float] = {"count": count}
        for horizon in horizons:
            sum_key = f"sum_{horizon}"
            pos_key = f"pos_{horizon}"
            avg_key = f"avg_{horizon}"
            pos_share_key = f"positive_{horizon}"
            if sum_key in payload:
                row[avg_key] = payload[sum_key] / count
            if pos_key in payload:
                row[pos_share_key] = payload[pos_key] / count
        for extra in ("parent", "semantic_tag", "coherence_bucket", "domain_bucket"):
            if extra in payload:
                row[extra] = payload[extra]
        output[key] = row
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown(
    path: Path,
    enriched: Mapping[str, Mapping[str, float]],
    base: Mapping[str, Mapping[str, float]],
    *,
    horizon: int,
    top_n: int,
) -> None:
    sortable = []
    for key, payload in enriched.items():
        parent_id = payload.get("parent")
        parent_stats = base.get(parent_id, {})
        avg = payload.get(f"avg_{horizon}")
        parent_avg = parent_stats.get(f"avg_{horizon}")
        if avg is None or parent_avg is None:
            continue
        lift = avg - parent_avg
        sortable.append(
            (
                lift,
                {
                    "strand": key,
                    "parent": parent_id,
                    "semantic_tag": payload.get("semantic_tag"),
                    "coherence_bucket": payload.get("coherence_bucket"),
                    "domain_bucket": payload.get("domain_bucket"),
                    "count": payload.get("count"),
                    "avg": avg,
                    "parent_avg": parent_avg,
                    "lift": lift,
                },
            )
        )
    sortable.sort(key=lambda item: item[0], reverse=True)
    best = sortable[:top_n]
    worst = sortable[-top_n:] if len(sortable) >= top_n else sortable
    lines = ["# Enriched Strand Performance", "", f"## Top {len(best)} by {horizon}m ROC lift", ""]
    lines.append("| Strand | Parent | Tag | Coh | Dom | Count | Avg (bp) | Parent Avg (bp) | Lift (bp) |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for _, row in best:
        lines.append(
            f"| `{row['strand']}` | `{row['parent']}` | {row['semantic_tag']} | {row['coherence_bucket']} | "
            f"{row['domain_bucket']} | {row['count']:.0f} | {row['avg']:.4f} | {row['parent_avg']:.4f} | {row['lift']:.4f} |"
        )
    lines.extend(["", f"## Bottom {len(worst)} by {horizon}m ROC lift", ""])
    lines.append("| Strand | Parent | Tag | Coh | Dom | Count | Avg (bp) | Parent Avg (bp) | Lift (bp) |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for _, row in reversed(worst):
        lines.append(
            f"| `{row['strand']}` | `{row['parent']}` | {row['semantic_tag']} | {row['coherence_bucket']} | "
            f"{row['domain_bucket']} | {row['count']:.0f} | {row['avg']:.4f} | {row['parent_avg']:.4f} | {row['lift']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]
    base_raw, enriched_raw = _aggregate(Path(args.roc_dir), horizons=horizons)
    base_stats = _normalise_stats(base_raw, horizons=horizons)
    enriched_stats = _normalise_stats(enriched_raw, horizons=horizons)
    csv_rows = []
    fieldnames = [
        "strand",
        "parent",
        "semantic_tag",
        "coherence_bucket",
        "domain_bucket",
        "count",
    ]
    for horizon in horizons:
        fieldnames.extend([f"avg_{horizon}", f"positive_{horizon}"])
    for strand, payload in enriched_stats.items():
        row = {
            "strand": strand,
            "parent": payload.get("parent"),
            "semantic_tag": payload.get("semantic_tag"),
            "coherence_bucket": payload.get("coherence_bucket"),
            "domain_bucket": payload.get("domain_bucket"),
            "count": payload.get("count"),
        }
        for horizon in horizons:
            row[f"avg_{horizon}"] = payload.get(f"avg_{horizon}")
            row[f"positive_{horizon}"] = payload.get(f"positive_{horizon}")
        csv_rows.append(row)
    _write_csv(Path(args.output_csv), csv_rows, fieldnames)
    if args.summary_md:
        _render_markdown(Path(args.summary_md), enriched_stats, base_stats, horizon=args.summary_horizon, top_n=args.top_n)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate enriched strand performance tables")
    parser.add_argument("--roc-dir", default="docs/evidence/roc_history")
    parser.add_argument("--horizons", default="60,90,360")
    parser.add_argument("--output-csv", default="docs/evidence/enriched_strands.csv")
    parser.add_argument("--summary-md", default="docs/evidence/enriched_strands.md")
    parser.add_argument("--summary-horizon", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=10, help="Number of strands shown per ranking table")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
