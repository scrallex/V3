#!/usr/bin/env python3
"""Rebuild span-level gate exports from archived weekly gate JSONL files.

This script replaces the rolling/isolation command triplets referenced in
docs/evidence/roc_history/gameplan/job_specs.json by stitching together the
already-ingested gate history. It emits per-span gate files, ROC summaries,
outcome studies, simulator logs, and stacking comparison JSON payloads without
touching OANDA.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trading.portfolio_manager import StrategyProfile
from scripts.trading.regime_manifold_service import HazardCalibrator

UTC = timezone.utc


def _parse_iso(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _ts_from_ms(ts_ms: int | float) -> datetime:
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=UTC)


def _format_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class WeeklyArchive:
    start: datetime
    end: datetime
    path: Path


def _discover_weekly_archives(roc_dir: Path) -> List[WeeklyArchive]:
    archives: List[WeeklyArchive] = []
    prefix = "gates_with_roc_"
    for path in sorted(roc_dir.glob("gates_with_roc_*.jsonl")):
        name = path.name
        if not name.startswith(prefix):
            continue
        window = name[len(prefix) : -len(".jsonl")]
        try:
            start_str, end_str = window.split("_to_")
            start = _parse_iso(f"{start_str}T00:00:00Z")
            end = _parse_iso(f"{end_str}T00:00:00Z")
        except Exception:
            continue
        archives.append(WeeklyArchive(start=start, end=end, path=path))
    archives.sort(key=lambda item: item.start)
    return archives


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _regime_label(payload: Mapping[str, object]) -> str:
    regime = payload.get("regime")
    if isinstance(regime, Mapping):
        label = regime.get("label")
        if isinstance(label, str):
            return label
    if isinstance(regime, str):
        return regime
    return "unknown"


class SpanGateBuilder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.roc_dir = Path(args.roc_dir)
        self.output_root = Path(args.output_root)
        self.subsets_dir = self.output_root / "subsets"
        self.isolation_dir = self.output_root / "isolation"
        self.comparisons_dir = self.output_root / "comparisons"
        for directory in (self.subsets_dir, self.isolation_dir, self.comparisons_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.archives = _discover_weekly_archives(self.roc_dir)
        if not self.archives:
            raise SystemExit(f"No gates_with_roc_*.jsonl files found in {self.roc_dir}")
        span_catalog = json.loads(Path(args.span_catalog).read_text(encoding="utf-8"))
        self.spans = span_catalog.get("spans") or []
        if not self.spans:
            raise SystemExit("Span catalog does not contain any spans.")
        self.profile = StrategyProfile.load(Path(args.profile))
        self.hazard_max: Dict[str, float] = {
            symbol.upper(): float(entry.hazard_max or self.profile.global_defaults.get("hazard_max", 1.0))
            for symbol, entry in self.profile.instruments.items()
        }
        self.hazard_percentile = args.hazard_percentile
        self.signature_retention_ms = args.signature_retention * 60 * 1000
        self.admit_regimes = [item.strip() for item in args.admit_regimes.split(",") if item.strip()]
        self.min_confidence = args.min_confidence
        self.horizons = args.horizons
        self.job_specs = json.loads(Path(args.job_specs).read_text(encoding="utf-8"))
        self.nav = args.nav
        self.selected_spans: set[str] = set(getattr(args, "selected_spans", set()) or set())

    # ------------------------------------------------------------------
    # Event aggregation
    # ------------------------------------------------------------------
    def _iter_span_events(self, start: datetime, end: datetime) -> Iterable[Dict[str, object]]:
        for archive in self.archives:
            if archive.end <= start or archive.start >= end:
                continue
            with archive.path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts_ms = row.get("ts_ms")
                    if ts_ms is None:
                        continue
                    ts = _ts_from_ms(float(ts_ms))
                    if start <= ts < end:
                        row["_ts_dt"] = ts
                        yield row

    def _collect_span_events(self, span_id: str, start: datetime, end: datetime) -> List[Dict[str, object]]:
        events = list(self._iter_span_events(start, end))
        events.sort(key=lambda item: float(item.get("ts_ms", 0.0)))
        if not events:
            print(f"[span:{span_id}] WARNING: no events found between {start} and {end}", file=sys.stderr)
        return events

    # ------------------------------------------------------------------
    # Isolation rebuild helpers
    # ------------------------------------------------------------------
    def _build_isolation_events(self, events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        calibrators: Dict[str, HazardCalibrator] = defaultdict(lambda: HazardCalibrator(percentile=self.hazard_percentile))
        signature_history: Dict[str, Dict[str, Deque[int]]] = defaultdict(lambda: defaultdict(deque))
        iso_events: List[Dict[str, object]] = []
        for original in events:
            row = dict(original)
            instrument = str(row.get("instrument", "")).upper()
            if not instrument:
                continue
            hazard_value = float(row.get("hazard") or row.get("lambda") or 0.0)
            cal = calibrators[instrument]
            cal.update(hazard_value)
            adaptive = cal.threshold()
            threshold = min(adaptive, self.hazard_max.get(instrument, adaptive))
            reasons: List[str] = []
            if hazard_value > threshold:
                reasons.append("hazard_exceeds_adaptive_threshold")
                if hazard_value > threshold * 1.5:
                    reasons.append("hazard_fallback_requested")
            regime_label = _regime_label(row)
            if self.admit_regimes and regime_label not in self.admit_regimes:
                reasons.append("regime_filtered")
            regime_conf = 0.0
            regime_payload = row.get("regime")
            if isinstance(regime_payload, Mapping):
                try:
                    regime_conf = float(regime_payload.get("confidence", 0.0))
                except Exception:
                    regime_conf = 0.0
            row["hazard_threshold"] = threshold
            if regime_conf < self.min_confidence:
                reasons.append("regime_confidence_low")
            ts_ms = int(row.get("ts_ms") or 0)
            signature = None
            structure = row.get("structure")
            if isinstance(structure, Mapping):
                signature = structure.get("signature")
            if isinstance(signature, str):
                history = signature_history[instrument][signature]
                history.append(ts_ms)
                while history and ts_ms - history[0] > self.signature_retention_ms:
                    history.popleft()
                row["repetitions"] = len(history)
            admit_flag = len(reasons) == 0
            row["admit"] = admit_flag
            row["reasons"] = "|".join(reasons)
            row["lambda"] = hazard_value
            row["run_mode"] = "isolation"
            row.pop("_ts_dt", None)
            iso_events.append(row)
        return iso_events

    # ------------------------------------------------------------------
    # ROC summary builder
    # ------------------------------------------------------------------
    def _build_roc_summary(
        self,
        events: Sequence[Mapping[str, object]],
        start: datetime,
        end: datetime,
    ) -> Dict[str, object]:
        regimes: Dict[str, Dict[str, Dict[str, float]]] = {}
        for row in events:
            regime_label = _regime_label(row)
            forward = row.get("roc_forward_pct")
            if not isinstance(forward, Mapping):
                continue
            for horizon in self.horizons:
                value = forward.get(str(horizon))
                if value is None:
                    continue
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                stats = regimes.setdefault(regime_label, {}).setdefault(
                    str(horizon),
                    {"count": 0.0, "sum": 0.0, "positive": 0.0},
                )
                stats["count"] += 1.0
                stats["sum"] += val
                if val > 0:
                    stats["positive"] += 1.0
        regimes_payload: Dict[str, Dict[str, object]] = {}
        for regime_label, horizon_stats in regimes.items():
            per_regime: Dict[str, object] = {}
            for horizon, stats in horizon_stats.items():
                count = stats["count"]
                if count <= 0:
                    continue
                per_regime[horizon] = {
                    "count": int(count),
                    "avg_roc_pct": stats["sum"] / count,
                    "positive_pct": stats["positive"] / count,
                }
            regimes_payload[regime_label] = per_regime
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "window": {"start": _format_iso(start), "end": _format_iso(end)},
            "horizons": self.horizons,
            "regimes": regimes_payload,
        }

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------
    def _load_outcomes(self, path: Path) -> Dict[str, object]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_comparison(
        self,
        span_id: str,
        start: datetime,
        end: datetime,
        rolling_path: Path,
        isolation_path: Path,
        output_path: Path,
    ) -> None:
        rolling = self._load_outcomes(rolling_path)
        isolation = self._load_outcomes(isolation_path)
        metrics: List[Dict[str, object]] = []
        instruments = set()
        instruments.update((rolling.get("instruments") or {}).keys())
        instruments.update((isolation.get("instruments") or {}).keys())
        for instrument in sorted(instruments):
            r_entry = (rolling.get("instruments") or {}).get(instrument, {})
            i_entry = (isolation.get("instruments") or {}).get(instrument, {})
            if isinstance(r_entry, Mapping) and "note" in r_entry:
                continue
            if isinstance(i_entry, Mapping) and "note" in i_entry:
                continue
            horizons = set()
            horizons.update((r_entry or {}).keys())
            horizons.update((i_entry or {}).keys())
            for horizon in sorted(horizons):
                r_stats = r_entry.get(horizon, {}) if isinstance(r_entry, Mapping) else {}
                i_stats = i_entry.get(horizon, {}) if isinstance(i_entry, Mapping) else {}
                if not r_stats and not i_stats:
                    continue
                metric = {
                    "instrument": instrument,
                    "horizon": horizon,
                    "rolling": {
                        "count": r_stats.get("count"),
                        "avg_return_pct": r_stats.get("avg_return_pct"),
                        "positive_pct": r_stats.get("positive_pct"),
                        "auroc_admit": r_stats.get("auroc_admit"),
                    },
                    "isolation": {
                        "count": i_stats.get("count"),
                        "avg_return_pct": i_stats.get("avg_return_pct"),
                        "positive_pct": i_stats.get("positive_pct"),
                        "auroc_admit": i_stats.get("auroc_admit"),
                    },
                }
                metric["delta"] = {
                    "count": (i_stats.get("count") or 0) - (r_stats.get("count") or 0),
                    "avg_return_pct": (i_stats.get("avg_return_pct") or 0.0) - (r_stats.get("avg_return_pct") or 0.0),
                    "positive_pct": (i_stats.get("positive_pct") or 0.0) - (r_stats.get("positive_pct") or 0.0),
                    "auroc_admit": (i_stats.get("auroc_admit") or 0.0) - (r_stats.get("auroc_admit") or 0.0),
                }
                metrics.append(metric)
        payload = {
            "span_id": span_id,
            "window": {"start": _format_iso(start), "end": _format_iso(end)},
            "rolling_outcomes": str(rolling_path),
            "isolation_outcomes": str(isolation_path),
            "metrics": metrics,
        }
        _ensure_parent(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # External command runners
    # ------------------------------------------------------------------
    def _run_command(self, cmd: List[str], log_path: Optional[Path] = None) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{Path(__file__).resolve().parents[2]}:{env.get('PYTHONPATH','')}"
        if log_path:
            _ensure_parent(log_path)
            with log_path.open("w", encoding="utf-8") as handle:
                proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(Path.cwd()), env=env)
        else:
            proc = subprocess.run(cmd, cwd=str(Path.cwd()), env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")

    def _run_outcome_study(self, gate_path: Path, output_path: Path, log_path: Path) -> None:
        cmd = [
            "python3",
            "scripts/tools/signal_outcome_study.py",
            "--input",
            str(gate_path),
            "--price-mode",
            "embedded",
            "--horizons",
            ",".join(str(h) for h in self.horizons),
            "--export-json",
            str(output_path),
        ]
        self._run_command(cmd, log_path=log_path)

    def _run_simulator(self, gate_path: Path, log_path: Path) -> None:
        cmd = [
            "python3",
            "scripts/research/simulate_day.py",
            str(gate_path),
            "--profile",
            self.args.profile,
            "--nav",
            str(self.nav),
        ]
        self._run_command(cmd, log_path=log_path)

    # ------------------------------------------------------------------
    # Main driver
    # ------------------------------------------------------------------
    def process(self) -> None:
        for span in self.spans:
            span_id = span["span_id"]
            if self.selected_spans and span_id not in self.selected_spans:
                continue
            start = _parse_iso(span["start"])
            end = _parse_iso(span["end"])
            print(f"[span:{span_id}] building events {start} -> {end}")
            events = self._collect_span_events(span_id, start, end)
            if not events:
                continue
            rolling_gate = self.subsets_dir / f"gates_{span_id}.jsonl"
            rolling_summary = self.subsets_dir / f"roc_summary_{span_id}.json"
            isolation_gate = self.isolation_dir / f"gates_{span_id}.jsonl"
            isolation_summary = self.isolation_dir / f"roc_summary_{span_id}.json"
            rolling_outcomes = self.subsets_dir / f"outcomes_{span_id}.json"
            isolation_outcomes = self.isolation_dir / f"outcomes_{span_id}.json"
            rolling_outcome_log = self.subsets_dir / f"outcomes_{span_id}.log"
            isolation_outcome_log = self.isolation_dir / f"outcomes_{span_id}.log"
            rolling_simlog = self.subsets_dir / f"simulate_{span_id}.log"
            isolation_simlog = self.isolation_dir / f"simulate_{span_id}.log"
            comparison_path = self.comparisons_dir / f"stacking_{span_id}.json"

            # Rolling gate export
            print(f"[span:{span_id}] writing rolling gate file -> {rolling_gate}")
            self._write_gate_file(events, rolling_gate, run_mode="rolling")
            rolling_summary_payload = self._build_roc_summary(events, start, end)
            _ensure_parent(rolling_summary)
            rolling_summary.write_text(json.dumps(rolling_summary_payload, indent=2), encoding="utf-8")

            # Isolation gate export
            print(f"[span:{span_id}] rebuilding isolation gate file -> {isolation_gate}")
            iso_events = self._build_isolation_events(events)
            self._write_gate_file(iso_events, isolation_gate, run_mode="isolation")
            iso_summary_payload = self._build_roc_summary(iso_events, start, end)
            _ensure_parent(isolation_summary)
            isolation_summary.write_text(json.dumps(iso_summary_payload, indent=2), encoding="utf-8")

            # Outcome studies + simulators
            print(f"[span:{span_id}] running outcome study (rolling)")
            self._run_outcome_study(rolling_gate, rolling_outcomes, rolling_outcome_log)
            print(f"[span:{span_id}] running simulator (rolling)")
            self._run_simulator(rolling_gate, rolling_simlog)

            print(f"[span:{span_id}] running outcome study (isolation)")
            self._run_outcome_study(isolation_gate, isolation_outcomes, isolation_outcome_log)
            print(f"[span:{span_id}] running simulator (isolation)")
            self._run_simulator(isolation_gate, isolation_simlog)

            # Comparison payload
            print(f"[span:{span_id}] computing comparison -> {comparison_path}")
            self._build_comparison(span_id, start, end, rolling_outcomes, isolation_outcomes, comparison_path)

    def _write_gate_file(self, events: Sequence[Mapping[str, object]], path: Path, *, run_mode: str) -> None:
        _ensure_parent(path)
        with path.open("w", encoding="utf-8") as handle:
            for event in events:
                row = dict(event)
                row["run_mode"] = run_mode
                row.pop("_ts_dt", None)
                handle.write(json.dumps(row))
                handle.write("\n")


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild span-level gate exports from archived weekly gates.")
    parser.add_argument("--roc-dir", default="docs/evidence/roc_history", help="Path to weekly gates_with_roc_*.jsonl files")
    parser.add_argument("--output-root", default="docs/evidence/roc_history/gameplan", help="Output root for span artefacts")
    parser.add_argument("--span-catalog", default="docs/evidence/roc_history/gameplan/span_catalog.json", help="Span catalog JSON")
    parser.add_argument("--job-specs", default="docs/evidence/roc_history/gameplan/job_specs.json", help="Job specs JSON (for metadata)")
    parser.add_argument("--profile", default="config/echo_strategy.yaml", help="Strategy profile for hazard caps")
    parser.add_argument("--hazard-percentile", type=float, default=0.95, help="Percentile used for isolation hazard calibrator")
    parser.add_argument("--signature-retention", type=int, default=720, help="Signature retention window (minutes)")
    parser.add_argument(
        "--admit-regimes",
        default="trend_bull,trend_bear,mean_revert,neutral,chaotic",
        help="Comma separated regime labels allowed for admission",
    )
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum regime confidence")
    parser.add_argument("--horizons", default="5,15,30,60,240", help="Comma separated horizons for ROC/outcomes")
    parser.add_argument("--nav", type=float, default=10000.0, help="NAV used for simulate_day nav argument")
    parser.add_argument(
        "--span-id",
        action="append",
        help="Optional span_id filter (can be provided multiple times or comma separated)",
    )
    args = parser.parse_args(argv)
    args.horizons = _parse_horizons(args.horizons)
    if not args.horizons:
        raise SystemExit("No valid horizons provided.")
    selected: set[str] = set()
    if args.span_id:
        for chunk in args.span_id:
            for part in chunk.split(","):
                part = part.strip()
                if part:
                    selected.add(part)
    args.selected_spans = selected
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    builder = SpanGateBuilder(args)
    builder.process()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
