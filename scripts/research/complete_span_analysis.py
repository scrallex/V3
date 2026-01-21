#!/usr/bin/env python3
"""Complete remaining span analysis with memory-efficient processing for large files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_outcome_study_safe(gate_path: Path, output_path: Path, log_path: Path, horizons: List[int]) -> None:
    """Run outcome study with memory-safe fallback for large files."""
    file_size_mb = gate_path.stat().st_size / (1024 * 1024)
    
    if file_size_mb > 100:
        print(f"[WARN] Large file detected ({file_size_mb:.1f}MB), using in-process analysis")
        _process_outcomes_in_memory(gate_path, output_path, horizons)
    else:
        cmd = [
            "python3", "scripts/tools/signal_outcome_study.py",
            "--input", str(gate_path),
            "--price-mode", "embedded",
            "--horizons", ",".join(str(h) for h in horizons),
            "--export-json", str(output_path),
        ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"[ERROR] Outcome study failed for {gate_path.name}")
            raise RuntimeError(f"Outcome study failed")


def _process_outcomes_in_memory(gate_path: Path, output_path: Path, horizons: List[int]) -> None:
    """Process outcomes directly without subprocess to avoid memory issues."""
    from datetime import datetime, timezone
    
    instruments: Dict[str, Dict[int, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    line_count = 0
    with gate_path.open() as f:
        for line in f:
            line_count += 1
            if line_count % 50000 == 0:
                print(f"  Processed {line_count} events...")
            
            try:
                row = json.loads(line.strip())
            except:
                continue
            
            instrument = row.get("instrument", "").upper()
            if not instrument:
                continue
            
            admit = row.get("admit", False)
            roc_forward = row.get("roc_forward_pct", {})
            
            for h in horizons:
                roc_val = roc_forward.get(str(h))
                if roc_val is None:
                    continue
                
                try:
                    roc_val = float(roc_val)
                except:
                    continue
                
                instruments[instrument][h]["returns"].append(roc_val)
                instruments[instrument][h]["admits"].append(1 if admit else 0)
    
    print(f"  Total events: {line_count}")
    
    # Build output
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizons": horizons,
        "instruments": {}
    }
    
    for instrument, horizon_data in instruments.items():
        inst_output: Dict[str, object] = {}
        for horizon, stats in horizon_data.items():
            returns = stats["returns"]
            admits = stats["admits"]
            
            if not returns:
                continue
            
            count = len(returns)
            avg_return = sum(returns) / count if count > 0 else 0.0
            positive_count = sum(1 for r in returns if r > 0)
            positive_pct = positive_count / count if count > 0 else 0.0
            
            admit_count = sum(admits)
            
            inst_output[str(horizon)] = {
                "count": count,
                "avg_return_pct": avg_return,
                "positive_pct": positive_pct,
                "admit_count": admit_count,
                "admit_pct": admit_count / count if count > 0 else 0.0
            }
        
        output["instruments"][instrument] = inst_output
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"  Wrote outcomes to {output_path}")


def _run_simulator_safe(gate_path: Path, log_path: Path, profile: str, nav: float) -> None:
    """Run simulator with error handling."""
    file_size_mb = gate_path.stat().st_size / (1024 * 1024)
    
    if file_size_mb > 100:
        print(f"[WARN] Large file ({file_size_mb:.1f}MB), skipping simulator to conserve memory")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"Skipped due to large file size ({file_size_mb:.1f}MB)\n")
        return
    
    cmd = [
        "python3", "scripts/research/simulate_day.py",
        str(gate_path), "--profile", profile, "--nav", str(nav)
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"[WARN] Simulator failed for {gate_path.name}")


def _build_comparison(
    span_id: str,
    rolling_outcomes: Path,
    isolation_outcomes: Path,
    output_path: Path
) -> None:
    """Build comparison JSON between rolling and isolation outcomes."""
    rolling = json.loads(rolling_outcomes.read_text()) if rolling_outcomes.exists() else {}
    isolation = json.loads(isolation_outcomes.read_text()) if isolation_outcomes.exists() else {}
    
    metrics: List[Dict[str, object]] = []
    instruments = set()
    instruments.update((rolling.get("instruments") or {}).keys())
    instruments.update((isolation.get("instruments") or {}).keys())
    
    for instrument in sorted(instruments):
        r_entry = (rolling.get("instruments") or {}).get(instrument, {})
        i_entry = (isolation.get("instruments") or {}).get(instrument, {})
        
        horizons = set()
        if isinstance(r_entry, dict):
            horizons.update(r_entry.keys())
        if isinstance(i_entry, dict):
            horizons.update(i_entry.keys())
        
        for horizon in sorted(horizons):
            r_stats = r_entry.get(horizon, {}) if isinstance(r_entry, dict) else {}
            i_stats = i_entry.get(horizon, {}) if isinstance(i_entry, dict) else {}
            
            if not r_stats and not i_stats:
                continue
            
            metric = {
                "instrument": instrument,
                "horizon": horizon,
                "rolling": {
                    "count": r_stats.get("count"),
                    "avg_return_pct": r_stats.get("avg_return_pct"),
                    "positive_pct": r_stats.get("positive_pct"),
                },
                "isolation": {
                    "count": i_stats.get("count"),
                    "avg_return_pct": i_stats.get("avg_return_pct"),
                    "positive_pct": i_stats.get("positive_pct"),
                },
                "delta": {
                    "avg_return_pct": (i_stats.get("avg_return_pct") or 0.0) - (r_stats.get("avg_return_pct") or 0.0),
                    "positive_pct": (i_stats.get("positive_pct") or 0.0) - (r_stats.get("positive_pct") or 0.0),
                }
            }
            metrics.append(metric)
    
    payload = {"span_id": span_id, "metrics": metrics}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def complete_analysis(gameplan_root: Path, profile: str, nav: float, horizons: List[int]) -> None:
    """Complete missing analysis for all spans."""
    subsets_dir = gameplan_root / "subsets"
    isolation_dir = gameplan_root / "isolation"
    comparisons_dir = gameplan_root / "comparisons"
    
    # Find all spans that have gate files
    gate_files = list(subsets_dir.glob("gates_*.jsonl"))
    
    for gate_file in sorted(gate_files):
        span_id = gate_file.stem.replace("gates_", "")
        print(f"\n[{span_id}] Checking status...")
        
        # Check what's missing
        rolling_gate = subsets_dir / f"gates_{span_id}.jsonl"
        rolling_outcomes = subsets_dir / f"outcomes_{span_id}.json"
        rolling_simlog = subsets_dir / f"simulate_{span_id}.log"
        
        isolation_gate = isolation_dir / f"gates_{span_id}.jsonl"
        isolation_outcomes = isolation_dir / f"outcomes_{span_id}.json"
        isolation_outcome_log = isolation_dir / f"outcomes_{span_id}.log"
        isolation_simlog = isolation_dir / f"simulate_{span_id}.log"
        
        comparison = comparisons_dir / f"stacking_{span_id}.json"
        
        # Process rolling if needed
        if rolling_gate.exists() and not rolling_outcomes.exists():
            print(f"  Running rolling outcome study...")
            _run_outcome_study_safe(rolling_gate, rolling_outcomes, subsets_dir / f"outcomes_{span_id}.log", horizons)
        
        if rolling_gate.exists() and not rolling_simlog.exists():
            print(f"  Running rolling simulator...")
            _run_simulator_safe(rolling_gate, rolling_simlog, profile, nav)
        
        # Process isolation if needed
        if isolation_gate.exists() and not isolation_outcomes.exists():
            print(f"  Running isolation outcome study...")
            _run_outcome_study_safe(isolation_gate, isolation_outcomes, isolation_outcome_log, horizons)
        
        if isolation_gate.exists() and not isolation_simlog.exists():
            print(f"  Running isolation simulator...")
            _run_simulator_safe(isolation_gate, isolation_simlog, profile, nav)
        
        # Generate comparison if needed
        if rolling_outcomes.exists() and isolation_outcomes.exists() and not comparison.exists():
            print(f"  Building comparison...")
            _build_comparison(span_id, rolling_outcomes, isolation_outcomes, comparison)
        
        print(f"  ✓ {span_id} complete")


def main():
    parser = argparse.ArgumentParser(description="Complete remaining span analysis")
    parser.add_argument("--gameplan-root", default="docs/evidence/roc_history/gameplan")
    parser.add_argument("--profile", default="config/echo_strategy.yaml")
    parser.add_argument("--nav", type=float, default=10000.0)
    parser.add_argument("--horizons", default="5,15,30,60,240")
    args = parser.parse_args()
    
    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    gameplan_root = Path(args.gameplan_root)
    
    print(f"Completing span analysis in {gameplan_root}")
    print(f"Horizons: {horizons}")
    
    complete_analysis(gameplan_root, args.profile, args.nav, horizons)
    
    print("\n✓ All spans completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())