#!/usr/bin/env python3
"""Run a short ABA cycle for the telemetry optimizer."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

from .runtime import bootstrap_from_yaml


@dataclass(frozen=True)
class Phase:
    name: str
    minutes: float
    autotune: bool


def _log_event(root: Path, phase: str, state: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "state": state,
        "timestamp": time.time(),
    }
    path = root / "aba_markers.ndjson"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def _run_phase(phase: Phase, pipeline, marker_root: Path, stop_event: signal.Signals | None) -> None:
    pipeline.set_autotune_enabled(phase.autotune)
    status = "ENABLED" if phase.autotune else "DISABLED"
    print(f"[{time.strftime('%H:%M:%S')}] phase {phase.name}: auto-tune {status}, duration {phase.minutes:.1f} minutes")
    sys.stdout.flush()

    _log_event(marker_root, phase.name, "start")
    deadline = time.time() + phase.minutes * 60.0
    try:
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(15.0, remaining))
    finally:
        _log_event(marker_root, phase.name, "end")
        pipeline.set_autotune_enabled(False)
        print(f"[{time.strftime('%H:%M:%S')}] phase {phase.name} complete")
        sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short ABA sequence (baseline → treatment → washout).")
    parser.add_argument("--config", default="config/telemetry.yaml", help="Telemetry YAML config path")
    parser.add_argument("--baseline-min", type=float, default=20.0, help="Baseline duration in minutes (default: 20)")
    parser.add_argument("--treatment-min", type=float, default=20.0, help="Treatment duration in minutes (default: 20)")
    parser.add_argument("--washout-min", type=float, default=20.0, help="Washout duration in minutes (default: 20)")
    parser.add_argument("--marker-dir", default="logs/telemetry", help="Directory to write ABA marker NDJSON logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    marker_root = Path(args.marker_dir)

    phases = [
        Phase("A1_baseline", args.baseline_min, autotune=False),
        Phase("B_treatment", args.treatment_min, autotune=True),
        Phase("A2_washout", args.washout_min, autotune=False),
    ]

    pipeline = bootstrap_from_yaml(args.config)

    stop_flag = {"triggered": False}

    def _handle_signal(signum, frame):  # pragma: no cover - signal handler
        if not stop_flag["triggered"]:
            print("\nSignal received, stopping after current phase...")
            stop_flag["triggered"] = True
        else:
            print("Immediate termination requested.")
            pipeline.stop()
            sys.exit(1)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    pipeline.start()
    try:
        pipeline.set_autotune_enabled(False)
        # Give collectors a moment to warm up
        time.sleep(2.0)

        for phase in phases:
            if stop_flag["triggered"]:
                break
            _run_phase(phase, pipeline, marker_root, None)

        print("ABA sequence complete. Auto-tune disabled.")
    finally:
        pipeline.set_autotune_enabled(False)
        pipeline.stop()


if __name__ == "__main__":
    main()
