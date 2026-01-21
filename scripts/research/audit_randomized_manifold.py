#!/usr/bin/env python3
"""
Null Hypothesis Audit: Randomized Manifold Test
===============================================
This script tests whether the "Edge" of the Bundle Strategy (MB003) is structural
or merely a result of overfitting/luck.

Methodology:
1. Load genuine Gate events (Structure) and their Forward Returns (ROC).
2. Calculate the Real Performance (Mean return of MB003 hits).
3. "The Shuffle Test":
   - Keep the Gate Structure constant (same signals at same times).
   - Shuffle the pool of Forward Returns (destroying the Structure->Return link).
   - Recalculate Performance.
   - Repeat 1000 times to build a Null Distribution.
4. Metric:
   - Z-Score = (Real_Exp - Null_Mean) / Null_StdDev
   - If Z > 3.0, the link between Structure and Return is HIGHLY SIGNIFICANT.
   - If Z < 2.0, the strategy is indistinguishable from luck.
"""

import argparse
import json
import random
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict

# Ensure we can import from the repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.bundle_rules import get_bundle_hits

def load_gates(path: Path, horizon: str) -> List[Dict]:
    """Load gate records with ROC data."""
    gates = []
    print(f"Loading gates from {path}...")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                gate = json.loads(line)
                # We only care about gates that actually have the target horizon (e.g. 360m)
                if "roc_forward_pct" in gate and horizon in gate["roc_forward_pct"]:
                    gates.append(gate)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(gates)} valid gates.")
    return gates

def evaluate_performance(gates: List[Dict], rocs: List[float], bundle_id: str = "MB003") -> float:
    """
    Calculate mean return of the bundle using the provided ROCs.
    Note: 'gates' provides the signals, 'rocs' provides the outcomes.
    They must be aligned by index (or shuffled for the test).
    """
    total_return = 0.0
    count = 0
    
    # We assume 'rocs' is same length as 'gates'
    for i, gate in enumerate(gates):
        # Check if bundle activates
        hits, _, _ = get_bundle_hits(gate, bundle_config="config/bundle_strategy.yaml")
        
        # Check if our bundle is in the hits
        active = any(h['id'] == bundle_id for h in hits)
        
        if active:
            total_return += rocs[i]
            count += 1
            
    return (total_return / count) if count > 0 else 0.0

def run_audit(gate_path: Path, iterations: int = 1000, horizon: str = "360"):
    gates = load_gates(gate_path, horizon)
    run_audit_on_gates(gates, iterations, horizon)

def run_audit_on_gates(gates: List[Dict], iterations: int = 1000, horizon: str = "360"):
    bundle_target = "MB003" # Mean Revert Promote
    
    if not gates:
        print("No gates found.")
        return

    # Extract the "Real" ROC sequence associated with these gates
    # We convert to Basis Points (bps) for readability
    real_rocs = [float(g["roc_forward_pct"][horizon]) * 10000.0 for g in gates]
    
    # 2. Calculate Real PnL
    # We need to re-evaluate the hits because the JSON might be stale if rules changed,
    # or we just want to be sure.
    print(f"Evaluating Real Performance for {bundle_target}...")
    real_mean = evaluate_performance(gates, real_rocs, bundle_id=bundle_target)
    print(f"Real Mean Return: {real_mean:.2f} bps per trade")

    # 3. The Shuffle Test
    print(f"Running {iterations} Monte Carlo shuffles...")
    null_means = []
    
    # Pre-calculate hit indices to speed up the loop
    # We only need to know WHICH gates triggered, then we just sum their shuffled returns
    hit_indices = []
    for i, gate in enumerate(gates):
        hits, _, _ = get_bundle_hits(gate, bundle_config="config/bundle_strategy.yaml")
        if any(h['id'] == bundle_target for h in hits):
            hit_indices.append(i)
    
    num_hits = len(hit_indices)
    print(f"Bundle {bundle_target} triggered {num_hits} times ({num_hits/len(gates)*100:.2f}% coverage)")
    
    if num_hits == 0:
        print("Bundle never triggered. Cannot audit.")
        return

    # Optimization: Convert ROCs to numpy array for fast shuffling/indexing
    roc_pool = np.array(real_rocs)
    
    for _ in range(iterations):
        # Shuffle the OUTCOMES, destroying the link to STRUCTURE
        np.random.shuffle(roc_pool)
        
        # The 'strategy' picks the same indices (because structure is constant),
        # but now they point to random returns from the same era.
        # Note: We must be careful. If we just shuffle the pool, 'hit_indices' 
        # points to the slots. We want the slots to have random values.
        
        # Calculate mean of the values at the hit indices
        random_return = np.mean(roc_pool[hit_indices])
        null_means.append(random_return)

    # 4. Analysis
    null_means = np.array(null_means)
    null_mean = np.mean(null_means)
    null_std = np.std(null_means)
    
    z_score = (real_mean - null_mean) / null_std
    
    print("\n=== AUDIT RESULTS ===")
    print(f"Bundle: {bundle_target}")
    print(f"Trades: {num_hits}")
    print(f"Real Exp:      {real_mean:+.2f} bps")
    print(f"Null Exp Mean: {null_mean:+.2f} bps (Luck)")
    print(f"Null Exp Std:  {null_std:.2f} bps")
    print(f"Z-Score:       {z_score:+.2f}")
    
    print("\nInterpretation:")
    if z_score > 3.0:
        print("[PASS] EXTREMELY SIGNIFICANT. The edge is structural.")
    elif z_score > 2.0:
        print("[PASS] Significant. Likely structural.")
    elif z_score > 1.0:
        print("[WEAK] Marginal result. Could be noise.")
    else:
        print("[FAIL] Indistinguishable from random chance.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input files or directories")
    parser.add_argument("--iter", type=int, default=1000, help="Monte Carlo iterations")
    parser.add_argument("--horizon", type=str, default="360", help="Forward horizon in minutes")
    args = parser.parse_args()
    
    all_files = []
    for inp in args.inputs:
        path = Path(inp)
        if path.is_dir():
            all_files.extend(sorted(list(path.glob("gates_with_roc_*.jsonl"))))
        else:
            all_files.append(path)
            
    print(f"Processing {len(all_files)} files...")
    
    # Check for empty list
    if not all_files:
        print("No files found.")
        sys.exit(0)

    # Optimization: Process year by year or just limit to the provided list
    all_gates = []
    for f in all_files:
        all_gates.extend(load_gates(f, args.horizon))
        
    run_audit_on_gates(all_gates, args.iter, args.horizon)
