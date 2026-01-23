#!/usr/bin/env python3
"""
05_debug_features.py
Check Volatility Stats in Training Data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

path = Path("/sep/research_lab/data/EUR_USD_rich.parquet")
df = pd.read_parquet(path)

# Recalculate Techs exactly like Agent
df["close"] = df["close"].astype(float)
df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
df["volatility"] = df["log_ret"].rolling(20).std()

print("--- Volatility Stats (Training Data) ---")
print(df["volatility"].describe())
print("\n--- Percentiles ---")
print(df["volatility"].quantile([0.1, 0.5, 0.9, 0.95, 0.99]))

# Check how many rows exceed the reported threshold 3.23e-05
threshold = 3.23e-05
count = (df["volatility"] > threshold).sum()
ratio = count / len(df)
print(f"\nRows > {threshold}: {count} ({ratio:.2%})")
