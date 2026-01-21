#!/usr/bin/env python3
"""Aggregate ROC summaries across long horizons and emit tables/figures."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class WeekWindow:
    label: str
    start: datetime
    end: datetime


def _parse_window_from_name(path: Path) -> WeekWindow:
    """Parse YYYY-MM-DD_to_YYYY-MM-DD window from filename."""

    stem = path.stem  # roc_summary_YYYY-MM-DD_to_YYYY-MM-DD
    _, _, window = stem.split("_", 2)
    start_str, end_str = window.split("_to_")
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    return WeekWindow(label=f"{start_str}_to_{end_str}", start=start, end=end)


def _iter_summary_files(directory: Path) -> Iterable[Path]:
    yield from sorted(directory.glob("roc_summary_*.json"))


def _load_summary(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _aggregate_weekly_records(
    directory: Path,
    *,
    start: datetime | None,
    end: datetime | None,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for summary_file in _iter_summary_files(directory):
        window = _parse_window_from_name(summary_file)
        if start and window.start < start:
            continue
        if end and window.start > end:
            continue
        payload = _load_summary(summary_file)
        regimes = payload.get("regimes", {}) if isinstance(payload, dict) else {}
        for regime, horizons in regimes.items():
            if not isinstance(horizons, dict):
                continue
            for horizon, stats in horizons.items():
                try:
                    horizon_int = int(horizon)
                except (TypeError, ValueError):
                    continue
                count = stats.get("count", 0) if isinstance(stats, dict) else 0
                avg_pct = stats.get("avg_roc_pct", 0.0) if isinstance(stats, dict) else 0.0
                pos_pct = stats.get("positive_pct", 0.0) if isinstance(stats, dict) else 0.0
                records.append(
                    {
                        "week_start": window.start,
                        "week_label": window.label,
                        "regime": regime,
                        "horizon_min": horizon_int,
                        "count": count,
                        "avg_roc_pct": avg_pct,
                        "positive_pct": pos_pct,
                    }
                )
    if not records:
        raise SystemExit("No ROC summary files matched the provided window")
    return pd.DataFrame.from_records(records)


def _weighted_average(df: pd.DataFrame, value_col: str) -> float:
    num = (df[value_col] * df["count"]).sum()
    den = df["count"].sum()
    return num / den if den else 0.0


def _regime_horizon_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for (regime, horizon), subset in df.groupby(["regime", "horizon_min"]):
        total_count = subset["count"].sum()
        avg = _weighted_average(subset, "avg_roc_pct")
        pos = _weighted_average(subset, "positive_pct")
        grouped.append(
            {
                "regime": regime,
                "horizon_min": horizon,
                "samples": total_count,
                "avg_roc_pct": avg,
                "positive_pct": pos,
            }
        )
    result = pd.DataFrame(grouped).sort_values(["regime", "horizon_min"]).reset_index(drop=True)
    return result


def _monthly_table(df: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    monthly_df = df[df["horizon_min"] == horizon].copy()
    monthly_df["month"] = monthly_df["week_start"].dt.to_period("M").dt.to_timestamp()
    grouped = []
    for (month, regime), subset in monthly_df.groupby(["month", "regime"]):
        total = subset["count"].sum()
        avg = _weighted_average(subset, "avg_roc_pct")
        grouped.append({"month": month, "regime": regime, "avg_roc_pct": avg, "samples": total})
    result = pd.DataFrame(grouped).sort_values(["month", "regime"]).reset_index(drop=True)
    return result


def _top_bottom_months(monthly_df: pd.DataFrame, *, regime: str, top_n: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset = monthly_df[monthly_df["regime"] == regime].copy()
    subset["avg_bp"] = subset["avg_roc_pct"] * 10000.0
    subset = subset.sort_values("avg_bp")
    lows = subset.head(top_n)[["month", "avg_bp", "samples"]]
    highs = subset.tail(top_n)[["month", "avg_bp", "samples"]]
    return lows, highs


def _plot_horizon_bars(table: pd.DataFrame, output_path: Path) -> None:
    regimes = sorted(table["regime"].unique())
    fig, axes = plt.subplots(len(regimes), 1, figsize=(9, 3 * len(regimes)), sharex=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        subset = table[table["regime"] == regime]
        ax.bar(subset["horizon_min"], subset["avg_roc_pct"] * 10000.0, color="#3B82F6")
        ax.set_title(f"{regime} — Avg ROC vs Horizon")
        ax.set_ylabel("bp")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("Horizon (minutes)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_monthly_lines(monthly_df: pd.DataFrame, output_path: Path) -> None:
    pivot = (
        monthly_df.assign(month=pd.to_datetime(monthly_df["month"]))
        .pivot(index="month", columns="regime", values="avg_roc_pct")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    for regime in pivot.columns:
        ax.plot(pivot.index, pivot[regime] * 10000.0, label=regime)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("60m ROC (bp)")
    ax.set_xlabel("Month")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)


def _write_extrema_report(monthly_df: pd.DataFrame, output_path: Path, top_n: int = 3) -> None:
    records = {}
    for regime in sorted(monthly_df["regime"].unique()):
        lows, highs = _top_bottom_months(monthly_df, regime=regime, top_n=top_n)
        records[regime] = {
            "lowest": [
                {"month": row["month"].strftime("%Y-%m"), "avg_bp": row["avg_bp"], "samples": int(row["samples"])}
                for _, row in lows.iterrows()
            ],
            "highest": [
                {"month": row["month"].strftime("%Y-%m"), "avg_bp": row["avg_bp"], "samples": int(row["samples"])}
                for _, row in highs.iterrows()
            ],
        }
    _ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise ROC history across long windows")
    parser.add_argument("--roc-dir", default="docs/evidence/roc_history", help="Directory containing roc_summary_*.json files")
    parser.add_argument("--start", help="Inclusive ISO date (YYYY-MM-DD). Omit for earliest.")
    parser.add_argument("--end", help="Inclusive ISO date (YYYY-MM-DD). Omit for latest.")
    parser.add_argument("--monthly-horizon", type=int, default=60, help="Horizon (minutes) used for monthly trend plots (default 60)")
    parser.add_argument("--output-dir", default="docs/evidence/longitudinal_summary", help="Directory for tables + charts")
    parser.add_argument("--top-n", type=int, default=3, help="Number of extrema months to report per regime")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    start = datetime.fromisoformat(args.start) if args.start else None
    # inclusive end: add 1 day when filtering
    end = datetime.fromisoformat(args.end) if args.end else None
    roc_dir = Path(args.roc_dir)
    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    df = _aggregate_weekly_records(roc_dir, start=start, end=end)
    regime_table = _regime_horizon_table(df)
    monthly_table = _monthly_table(df, horizon=args.monthly_horizon)

    _save_table(regime_table, output_dir / "regime_horizon_summary.csv")
    _save_table(monthly_table, output_dir / "monthly_60m_summary.csv")
    _write_extrema_report(monthly_table, output_dir / "monthly_extrema.json", top_n=args.top_n)
    _plot_horizon_bars(regime_table, output_dir / "horizon_bars.png")
    _plot_monthly_lines(monthly_table, output_dir / "monthly_trends.png")

    print(f"[summary] wrote tables + charts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
