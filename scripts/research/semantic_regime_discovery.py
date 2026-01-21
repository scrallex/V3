#!/usr/bin/env python3
"""Brute-force semantic regime discovery orchestrator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGIMES = (
    "highly_stable",
    "improving_stability",
    "high_rupture_event",
    "chaotic_price_action",
    "low_hazard_environment",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover profitable semantic regimes")
    parser.add_argument("--config", type=Path, required=True, help="Backtest configuration consumed by run_unified_backtest")
    parser.add_argument("--instruments", nargs="*", help="Subset of instruments to analyse (defaults to config list)")
    parser.add_argument(
        "--regime",
        action="append",
        dest="regimes",
        help="Regime tag to test (can be repeated); defaults to canonical set",
    )
    parser.add_argument("--output", type=Path, default=Path("output/semantic_regime_summary.csv"), help="CSV summary path")
    parser.add_argument("--extra-args", default="", help="Additional CLI args forwarded to run_unified_backtest")
    parser.add_argument("--make-path", default="make", help="Path to make executable")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run stdout")
    return parser.parse_args(argv)


def _config_instruments(path: Path) -> List[str]:
    content = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - config parsing fallback
            raise RuntimeError("Config is not valid JSON and PyYAML is unavailable") from exc
        payload = yaml.safe_load(content)
    if not isinstance(payload, Mapping):
        raise ValueError("Backtest config must decode to a mapping")
    if "instruments" in payload and isinstance(payload["instruments"], Iterable):
        return [str(inst).upper() for inst in payload["instruments"]]
    if "semantic_filters" in payload and isinstance(payload["semantic_filters"], Mapping):
        return [str(inst).upper() for inst in payload["semantic_filters"].keys()]
    raise ValueError("Config must define either 'instruments' or 'semantic_filters'")


def _extract_json_block(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("no JSON object could be decoded", stripped, 0)


def _call_unified_backtest(
    *,
    make_path: str,
    args: Sequence[str],
    cwd: Path,
) -> Dict[str, Any]:
    cmd = [make_path, "-C", str(cwd), "unified-backtest", f"ARGS={' '.join(args)}"]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"unified-backtest failed with code {result.returncode}:\nSTDOUT:{result.stdout}\nSTDERR:{result.stderr}"
        )
    try:
        return _extract_json_block(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Run output was not valid JSON. Command stdout/stderr attached." f"\nSTDOUT:{result.stdout}\nSTDERR:{result.stderr}"
        ) from exc


def _instrument_rows(
    summary: Mapping[str, Any],
    regime: str,
    instrument: str,
) -> Dict[str, Any]:
    instruments = summary.get("instruments")
    if not isinstance(instruments, Mapping) or instrument not in instruments:
        raise ValueError(f"Instrument '{instrument}' missing from run output")
    payload = instruments[instrument]
    if not isinstance(payload, Mapping) or payload.get("error"):
        return {
            "instrument": instrument,
            "regime": regime,
            "pnl": float("nan"),
            "return_pct": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "trades": 0,
            "win_rate": float("nan"),
            "filtered_minutes": 0,
            "tag_counts": "{}",
        }
    metrics = payload.get("metrics") if isinstance(payload, Mapping) else {}
    semantic = payload.get("semantic_summary") if isinstance(payload, Mapping) else {}
    tag_counts = semantic.get("tag_counts") if isinstance(semantic, Mapping) else {}
    return {
        "instrument": instrument,
        "regime": regime,
        "pnl": metrics.get("pnl"),
        "return_pct": metrics.get("return_pct"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_dd") or metrics.get("max_drawdown"),
        "trades": metrics.get("trades"),
        "win_rate": metrics.get("win_rate"),
        "filtered_minutes": semantic.get("filtered_minutes"),
        "tag_counts": json.dumps(tag_counts, separators=(",", ":")) if tag_counts else "{}",
    }


def _render_summary(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "instrument",
        "regime",
        "pnl",
        "return_pct",
        "sharpe",
        "max_drawdown",
        "trades",
        "win_rate",
        "filtered_minutes",
        "tag_counts",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    regimes = args.regimes or list(DEFAULT_REGIMES)
    instruments = [inst.upper() for inst in args.instruments] if args.instruments else _config_instruments(args.config)

    rows: List[Dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    base_args = [f"--config", str(args.config)]
    if args.extra_args:
        base_args.extend(args.extra_args.split())

    for instrument in instruments:
        for regime in regimes:
            run_args = list(base_args)
            run_args.extend(["--instruments", instrument, "--semantic-regime-filter", regime])
            output_path = Path("output") / "semantic_runs" / f"{instrument}_{regime}.json"
            run_args.extend(["--output", str(output_path)])
            summary = _call_unified_backtest(make_path=args.make_path, args=run_args, cwd=ROOT)
            row = _instrument_rows(summary, regime, instrument)
            rows.append(row)
            if not args.quiet:
                print(f"[{timestamp}] {instrument} @ {regime} => PnL {row['pnl']} Sharpe {row['sharpe']}")

    _render_summary(rows, args.output)
    if not args.quiet:
        print(f"Semantic regime summary written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI tool
    raise SystemExit(main())
