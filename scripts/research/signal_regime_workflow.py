#!/usr/bin/env python3
"""Plan + orchestration helper for the SEP signal regime restructuring workflow."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks  # type: ignore
except Exception:  # pragma: no cover - SciPy optional
    find_peaks = None


def _parse_window_from_name(path: Path) -> tuple[datetime, datetime]:
    stem = path.stem  # roc_summary_YYYY-MM-DD_to_YYYY-MM-DD
    _, _, window = stem.split("_", 2)
    start_str, end_str = window.split("_to_")
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    return start, end


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _serialize_dt(value: datetime) -> str:
    return value.replace(tzinfo=None).isoformat() + "Z"


def _human_label(classification: str) -> str:
    mapping = {
        "positive_impulse": "Impulse ↑",
        "negative_impulse": "Impulse ↓",
        "neutral_bridge": "Neutral Bridge",
    }
    return mapping.get(classification, classification)


@dataclass
class SpanRecord:
    span_id: str
    label: str
    classification: str
    start: datetime
    end: datetime
    duration_days: int
    weeks: int
    anchor: bool
    avg_bp: float
    zscore_mean: float
    stats: Dict[str, float]

    def to_json(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["start"] = _serialize_dt(self.start)
        payload["end"] = _serialize_dt(self.end)
        payload["label"] = _human_label(self.classification)
        return payload


class SignalRegimeWorkflow:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.roc_dir = Path(args.roc_dir)
        self.output_root = Path(args.output_root)
        self.subsets_dir = self.output_root / "subsets"
        self.isolation_dir = self.output_root / "isolation"
        self.comparisons_dir = self.output_root / "comparisons"
        self.robustness_dir = self.output_root / "robustness"
        for directory in [self.output_root, self.subsets_dir, self.isolation_dir, self.comparisons_dir, self.robustness_dir]:
            _ensure_dir(directory)
        self.horizons = args.horizons
        self.instruments = args.instruments
        self.profile = args.profile
        self.nav = args.nav
        self.min_state_weeks = args.min_state_weeks

    # ------------------------------------------------------------------
    # Phase 1: data loading
    # ------------------------------------------------------------------
    def load_history(self) -> pd.DataFrame:
        records: List[Dict[str, object]] = []
        for path in sorted(self.roc_dir.glob("roc_summary_*.json")):
            start, end = _parse_window_from_name(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
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
                            "week_start": start,
                            "week_end": end,
                            "week_label": f"{start.date()}_{end.date()}",
                            "regime": regime,
                            "horizon_min": horizon_int,
                            "count": count,
                            "avg_roc_pct": avg_pct,
                            "positive_pct": pos_pct,
                        }
                    )
        if not records:
            raise SystemExit(f"No roc_summary_*.json files found in {self.roc_dir}")
        df = pd.DataFrame(records)
        df.sort_values("week_start", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def save_master_frame(self, df: pd.DataFrame) -> Path:
        output_path = self.output_root / "processed_5y_roc.pkl"
        df.to_pickle(output_path)
        return output_path

    # ------------------------------------------------------------------
    # Phase 2: span identification
    # ------------------------------------------------------------------
    def identify_spans(self, df: pd.DataFrame) -> List[SpanRecord]:
        subset = df[(df["regime"] == self.args.meta_regime) & (df["horizon_min"] == self.args.target_horizon)].copy()
        if subset.empty:
            raise SystemExit(f"No rows found for regime={self.args.meta_regime} horizon={self.args.target_horizon}")
        subset["avg_roc_pct"] = subset["avg_roc_pct"].fillna(0.0)
        series = subset.sort_values("week_start").reset_index(drop=True)
        mu = series["avg_roc_pct"].mean()
        sigma = series["avg_roc_pct"].std() or 1e-9
        series["zscore"] = (series["avg_roc_pct"] - mu) / sigma
        series["classification"] = series["zscore"].apply(self._classify_point)
        if find_peaks is not None and len(series) >= 4:
            magnitude = np.abs(series["zscore"].to_numpy())
            prominence = float(np.percentile(magnitude, 65)) if magnitude.size else 0.5
            peak_idx, _ = find_peaks(magnitude, prominence=prominence)
            for idx in peak_idx:
                if idx < 0 or idx >= len(series):
                    continue
                series.loc[idx, "classification"] = "positive_impulse" if series.loc[idx, "zscore"] >= 0 else "negative_impulse"
        self._smooth_classifications(series)
        spans = self._group_spans(series)
        return spans

    def _classify_point(self, zscore: float) -> str:
        if zscore >= self.args.high_zscore:
            return "positive_impulse"
        if zscore <= -self.args.high_zscore:
            return "negative_impulse"
        return "neutral_bridge"

    def _group_spans(self, series: pd.DataFrame) -> List[SpanRecord]:
        spans: List[SpanRecord] = []
        current_label: Optional[str] = None
        span_start: Optional[datetime] = None
        span_end: Optional[datetime] = None
        anchor_assigned = False
        for row in series.itertuples():
            label = row.classification
            if current_label is None:
                current_label = label
                span_start = row.week_start
                span_end = row.week_end
                continue
            if label == current_label:
                span_end = row.week_end
                continue
            if span_start and span_end:
                record = self._make_span_record(series, current_label, span_start, span_end, anchor_assigned)
                if record:
                    if record.anchor:
                        anchor_assigned = True
                    spans.append(record)
            current_label = label
            span_start = row.week_start
            span_end = row.week_end
        if span_start and span_end:
            record = self._make_span_record(series, current_label, span_start, span_end, anchor_assigned)
            if record:
                spans.append(record)
        return spans

    def _smooth_classifications(self, series: pd.DataFrame) -> None:
        if self.min_state_weeks <= 1 or series.empty:
            return
        labels = series["classification"].tolist()
        idx = 0
        while idx < len(labels):
            j = idx
            while j < len(labels) and labels[j] == labels[idx]:
                j += 1
            length = j - idx
            if length < self.min_state_weeks:
                replacement = labels[idx - 1] if idx > 0 else (labels[j] if j < len(labels) else labels[idx])
                for k in range(idx, j):
                    labels[k] = replacement
            idx = j
        series["classification"] = labels

    def _make_span_record(
        self,
        series: pd.DataFrame,
        classification: Optional[str],
        span_start: datetime,
        span_end: datetime,
        anchor_assigned: bool,
    ) -> Optional[SpanRecord]:
        duration_days = max(1, (span_end - span_start).days)
        if duration_days < self.args.min_span_days:
            return None
        mask = (series["week_start"] >= span_start) & (series["week_start"] <= span_end)
        span_slice = series[mask]
        avg_bp = float(span_slice["avg_roc_pct"].mean() * 10000.0)
        zscore_mean = float(span_slice["zscore"].mean())
        stats = {
            "avg_count": float(span_slice["count"].mean()),
            "avg_positive_pct": float(span_slice["positive_pct"].mean()),
        }
        span_id = f"{span_start.date()}_{span_end.date()}"
        anchor = False
        if not anchor_assigned and duration_days >= self.args.validation_days:
            anchor = True
        return SpanRecord(
            span_id=span_id,
            label=_human_label(classification or "neutral_bridge"),
            classification=classification or "neutral_bridge",
            start=span_start,
            end=span_end,
            duration_days=duration_days,
            weeks=math.ceil(duration_days / 7),
            anchor=anchor,
            avg_bp=avg_bp,
            zscore_mean=zscore_mean,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Phase 3-5 planning helpers
    # ------------------------------------------------------------------
    def build_job_specs(self, spans: Sequence[SpanRecord]) -> Dict[str, object]:
        specs: Dict[str, object] = {"generated_at": datetime.now(timezone.utc).isoformat(), "spans": []}
        horizons_str = ",".join(str(h) for h in self.horizons)
        instrument_arg = f"--instruments {self.instruments}" if self.instruments else ""
        for span in spans:
            start_iso = span.start.replace(tzinfo=None).isoformat() + "Z"
            end_iso = span.end.replace(tzinfo=None).isoformat() + "Z"
            rolling_gate = self.subsets_dir / f"gates_{span.span_id}.jsonl"
            rolling_summary = self.subsets_dir / f"roc_summary_{span.span_id}.json"
            isolation_gate = self.isolation_dir / f"gates_{span.span_id}.jsonl"
            isolation_summary = self.isolation_dir / f"roc_summary_{span.span_id}.json"
            rolling_outcomes = self.subsets_dir / f"outcomes_{span.span_id}.json"
            isolation_outcomes = self.isolation_dir / f"outcomes_{span.span_id}.json"
            rolling_sim = self.subsets_dir / f"simulate_{span.span_id}.log"
            isolation_sim = self.isolation_dir / f"simulate_{span.span_id}.log"

            span_entry = {
                "span": span.to_json(),
                "commands": {
                    "rolling": [
                        f"python scripts/tools/backfill_gate_history.py --start {start_iso} --end {end_iso} {instrument_arg}".strip()
                        + f" --profile {self.profile} --export-json {rolling_gate} --export-roc-summary {rolling_summary}",
                        f"python scripts/tools/signal_outcome_study.py --input {rolling_gate} --price-mode embedded --horizons {horizons_str} --export-json {rolling_outcomes}",
                        f"python scripts/research/simulate_day.py {rolling_gate} --profile {self.profile} --nav {self.nav} > {rolling_sim}",
                    ],
                    "isolation": [
                        f"python scripts/tools/backfill_gate_history.py --start {start_iso} --end {end_iso} {instrument_arg}".strip()
                        + f" --profile {self.profile} --export-json {isolation_gate} --export-roc-summary {isolation_summary} --isolation",
                        f"python scripts/tools/signal_outcome_study.py --input {isolation_gate} --price-mode embedded --horizons {horizons_str} --export-json {isolation_outcomes}",
                        f"python scripts/research/simulate_day.py {isolation_gate} --profile {self.profile} --nav {self.nav} > {isolation_sim}",
                    ],
                },
                "comparison_output": str(self.comparisons_dir / f"stacking_{span.span_id}.json"),
            }
            specs["spans"].append(span_entry)
        spec_path = self.output_root / "job_specs.json"
        spec_path.write_text(json.dumps(specs, indent=2), encoding="utf-8")
        return specs

    def write_span_catalog(self, spans: Sequence[SpanRecord]) -> Path:
        catalog = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "meta_regime": self.args.meta_regime,
            "target_horizon": self.args.target_horizon,
            "spans": [span.to_json() for span in spans],
        }
        path = self.output_root / "span_catalog.json"
        path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the SEP signal regime restructuring workflow assets.")
    parser.add_argument("--roc-dir", default="docs/evidence/roc_history", help="Directory containing roc_summary_*.json exports")
    parser.add_argument("--output-root", default="docs/evidence/roc_history/gameplan", help="Directory used for derived outputs")
    parser.add_argument("--target-horizon", type=int, default=60, help="Horizon (minutes) used for span detection (default 60)")
    parser.add_argument("--meta-regime", default="mean_revert", help="Regime used for span classification (default mean_revert)")
    parser.add_argument("--high-zscore", type=float, default=0.6, help="Z-score threshold for impulse classification")
    parser.add_argument("--min-span-days", type=int, default=90, help="Minimum day count for a span to qualify")
    parser.add_argument("--min-state-weeks", type=int, default=3, help="Minimum contiguous weeks required to keep a classification label")
    parser.add_argument("--validation-days", type=int, default=182, help="Span duration that tags the validation anchor (default 26w)")
    parser.add_argument("--profile", default="config/echo_strategy.yaml", help="Strategy profile path forwarded to tooling commands")
    parser.add_argument("--instruments", help="Optional comma separated instruments forwarded to tooling commands")
    parser.add_argument("--horizons", default="5,15,30,60,240", help="Comma separated horizons for downstream studies")
    parser.add_argument("--nav", type=float, default=10000.0, help="NAV forwarded to simulate_day.py")
    return parser.parse_args(argv)


def _parse_horizons(raw: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        item = item.strip().rstrip("m")
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            continue
    return [value for value in values if value > 0]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.horizons = _parse_horizons(args.horizons)
    if not args.horizons:
        raise SystemExit("No valid horizons provided.")
    workflow = SignalRegimeWorkflow(args)
    df = workflow.load_history()
    master_path = workflow.save_master_frame(df)
    spans = workflow.identify_spans(df)
    catalog_path = workflow.write_span_catalog(spans)
    specs = workflow.build_job_specs(spans)
    print(f"[workflow] master frame -> {master_path}")
    print(f"[workflow] span catalog -> {catalog_path}")
    print(f"[workflow] job specs -> {workflow.output_root / 'job_specs.json'} (spans={len(specs['spans'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
