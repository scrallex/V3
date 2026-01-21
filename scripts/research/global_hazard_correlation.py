#!/usr/bin/env python3
"""
Global Hazard Correlation Study
===============================
Tests the "Zero Sum" / "Systemic Risk" hypothesis.

Methodology:
1. Load longitudinal gate history.
2. Extract 'hazard' (lambda) timeseries for each instrument.
3. Resample to a common frequency (e.g. Hourly) to handle async timestamps.
4. Compute Correlation Matrix.
5. Compute 'Global Hazard' (mean hazard across instruments).
6. Report findings.
"""

import argparse
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

def load_hazard_series_from_files(files: List[Path]) -> pd.DataFrame:
    data = []
    print(f"Processing {len(files)} files...")
    
    for f in files:
        with f.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    ts_ms = row.get("ts_ms")
                    inst = row.get("instrument")
                    hazard = row.get("hazard") or row.get("lambda")
                    
                    if ts_ms and inst and hazard is not None:
                        data.append({
                            "ts": pd.to_datetime(ts_ms, unit="ms"),
                            "instrument": inst,
                            "hazard": float(hazard)
                        })
                except:
                    continue
                    
    df = pd.DataFrame(data)
    if df.empty:
        return df
        
    # Pivot: Index=Time, Columns=Instrument, Values=Hazard
    # We create a pivot table, resampling to Hour to align data
    df.set_index("ts", inplace=True)
    pivot = df.groupby([pd.Grouper(freq="h"), "instrument"])["hazard"].mean().unstack()
    return pivot

def analyze_correlations(df: pd.DataFrame):
    print("\n=== Hazard Correlation Matrix ===")
    corr = df.corr()
    print(corr.round(2))
    
    avg_corr = corr.values[np.triu_indices_from(corr.values, 1)].mean()
    print(f"\nAverage Pairwise Correlation: {avg_corr:.3f}")
    
    # Global Hazard
    df["GLOBAL_MEAN"] = df.mean(axis=1)
    
    print("\n=== Global Hazard Stats ===")
    print(df["GLOBAL_MEAN"].describe())
    
    print("\nInterpretation:")
    if avg_corr > 0.5:
        print("[HIGH] Systemic Linkage. 'Hazard' is a global market state.")
    elif avg_corr > 0.2:
        print("[MED] Moderate Linkage. Some diversification benefit.")
    else:
        print("[LOW] Independent Manifolds. Zero Sum hypothesis is weak; local structure dominates.")

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
            
    # Load hazard series from the collected files
    # We refactor load_hazard_series to accept a list of files
    if not all_files:
        print("No files found.")
        sys.exit(0)
        
    df = load_hazard_series_from_files(all_files)
    if not df.empty:
        analyze_correlations(df)
    else:
        print("No data found.")
