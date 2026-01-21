#!/usr/bin/env python3
"""
Global Manifold V2 Prototype
============================
Simulates the "Global Hazard" gated strategy.

Methodology:
1. Load Gate Data (Index: Time, Instrument).
2. Compute 'Global Hazard' (Mean Lambda across universe).
3. Apply V2 Logic:
   - RED (Hazard > 0.52): Quarantine (No Trades).
   - YELLOW (0.48 < Hazard <= 0.52): Quality Filter (Coherence > 0.6).
   - GREEN (Hazard <= 0.48): All Systems Go.
4. Compare V1 (Raw MB003) vs V2 (Global Gated).
"""

import argparse
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

# Import bundle logic to identify V1 hits
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.bundle_rules import get_bundle_hits

from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_file(path: Path) -> List[Dict]:
    records = []
    try:
        # Load catalog once per process implicitly via get_bundle_hits caching
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    ts_ms = row.get("ts_ms")
                    inst = row.get("instrument")
                    hazard = row.get("hazard")
                    coherence = row.get("coherence")
                    
                    # Check for V1 Hit (MB003)
                    hits, [], _ = get_bundle_hits(row, bundle_config="config/bundle_strategy.yaml")
                    is_mb003 = any(h['id'] == "MB003" for h in hits)
                    
                    # Forward Return (360m)
                    roc_360 = row.get("roc_forward_pct", {}).get("360")
                    
                    if ts_ms and inst and hazard is not None:
                        records.append({
                            "ts": pd.to_datetime(ts_ms, unit="ms"),
                            "instrument": inst,
                            "hazard": float(hazard),
                            "coherence": float(coherence) if coherence else 0.0,
                            "is_signal": is_mb003,
                            "return_360": float(roc_360) if roc_360 else 0.0
                        })
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return records

def load_universe(files: List[Path]) -> pd.DataFrame:
    print(f"Loading {len(files)} files with multiprocessing...")
    all_records = []
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_file, f): f for f in files}
        for future in as_completed(futures):
            res = future.result()
            all_records.extend(res)
            
    print(f"Loaded {len(all_records)} total records.")
    
    df = pd.DataFrame(all_records)
    if df.empty:
        return df
        
    df.sort_values("ts", inplace=True)
    return df

def run_simulation(df: pd.DataFrame):
    # 1. Compute Global Hazard Index
    # Pivot to get hazard matrix
    # Resample to hourly to align
    print("Computing Global Hazard Index...")
    hazard_matrix = df.pivot_table(index="ts", columns="instrument", values="hazard")
    
    # Forward fill to handle slight async, limit to 2 hours staleness
    hazard_matrix = hazard_matrix.resample("1h").mean().ffill(limit=2)
    
    # Global Hazard is the mean of available instruments
    global_hazard = hazard_matrix.mean(axis=1)
    global_hazard.name = "global_hazard"
    
    # 2. Join Global Hazard back to the trade list
    # We use merge_asof to match trades to the latest known global state
    df.sort_values("ts", inplace=True)
    df_sim = pd.merge_asof(df, global_hazard, on="ts", direction="backward", tolerance=pd.Timedelta("4h"))
    
    # Drop rows where we couldn't match a global hazard (start of data)
    df_sim.dropna(subset=["global_hazard"], inplace=True)
    
    # 3. Simulate V1 vs V2
    print("Simulating Strategies...")
    
    # V1: Raw MB003
    v1_trades = df_sim[df_sim["is_signal"]]
    v1_pnl = v1_trades["return_360"].sum()
    v1_count = len(v1_trades)
    v1_avg = v1_trades["return_360"].mean() * 10000 # bps
    
    # V2: Global Gated
    # Logic:
    # RED: Global > 0.52 -> BLOCK
    # YELLOW: 0.48 < Global <= 0.52 -> FILTER (Coherence > 0.6)
    # GREEN: Global <= 0.48 -> PERMIT
    
    def v2_filter(row):
        h = row["global_hazard"]
        c = row["coherence"]
        
        if h > 0.52:
            return False # Red Regime
        elif h > 0.48:
            return c > 0.60 # Yellow Regime (Quality Filter)
        else:
            return True # Green Regime
            
    v2_trades = v1_trades[v1_trades.apply(v2_filter, axis=1)]
    v2_pnl = v2_trades["return_360"].sum()
    v2_count = len(v2_trades)
    v2_avg = v2_trades["return_360"].mean() * 10000 # bps
    
    # 4. Report
    print("\n=== SIMULATION RESULTS (240-360m Horizon) ===")
    print(f"Data Points: {len(df)}")
    print(f"Global Hazard Mean: {global_hazard.mean():.4f}")
    print("-" * 40)
    print(f"STRATEGY V1 (Independent Bundles)")
    print(f"Trades: {v1_count}")
    print(f"Total Return: {v1_pnl:.4f} R")
    print(f"Avg Expectancy: {v1_avg:.2f} bps")
    print("-" * 40)
    print(f"STRATEGY V2 (Global Manifold)")
    print(f"Trades: {v2_count} ({v2_count/v1_count*100:.1f}% of V1)")
    print(f"Total Return: {v2_pnl:.4f} R")
    print(f"Avg Expectancy: {v2_avg:.2f} bps")
    print("-" * 40)
    
    improvement = (v2_avg - v1_avg)
    print(f"IMPROVEMENT: +{improvement:.2f} bps per trade")
    
    # Regimes stats
    red_count = len(global_hazard[global_hazard > 0.52])
    green_count = len(global_hazard[global_hazard <= 0.48])
    total_hours = len(global_hazard)
    print(f"\nRegime Distribution:")
    print(f"RED (Risk Off):   {red_count/total_hours*100:.1f}%")
    print(f"GREEN (Risk On):  {green_count/total_hours*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input files or directories")
    args = parser.parse_args()
    
    all_files = []
    for inp in args.inputs:
        path = Path(inp)
        if path.is_dir():
            all_files.extend(sorted(list(path.glob("gates_with_roc_*.jsonl"))))
        else:
            all_files.append(path)

    if not all_files:
        sys.exit(0)
        
    df = load_universe(all_files)
    if not df.empty:
        run_simulation(df)
