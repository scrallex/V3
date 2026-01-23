#!/usr/bin/env python3
"""
06_verify_silence.py
Fetch EUR_USD Data for Jan 21-23 (The Silent Period).
Run Agent Logic to see if any trades SHOULD have happened.
"""
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

FETCH_SCRIPT = "/sep/scripts/research/fetch_oanda_history.py"
BIN_PATH = "/sep/bin/manifold_generator"
MODEL_PATH = "/sep/research_lab/models/model_EUR_USD.json"
DATA_DIR = Path("/sep/research_lab/data/debug")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_recent():
    # Last 48 Hours
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=48)

    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    csv_path = DATA_DIR / "debug_raw.csv"
    if csv_path.exists():
        print("Using existing CSV.")
        return csv_path

    cmd = [
        "python3",
        FETCH_SCRIPT,
        "--instrument",
        "EUR_USD",
        "--granularity",
        "S5",
        "--start",
        start_str,
        "--end",
        end_str,
        "--output",
        str(csv_path),
    ]
    print("Fetching...")
    subprocess.run(cmd, check=True)
    return csv_path


def run_agent_logic(csv_path):
    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df["timestamp_ns"] = df["time"].astype("int64")

    # Manifold
    print("Running Manifold...")
    recs = []
    for _, row in df.iterrows():
        recs.append(
            {
                "timestamp": int(row["timestamp_ns"]) // 1_000_000,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(recs, f)
        src = f.name
    dst = src + ".out"

    subprocess.run([BIN_PATH, "--input", src, "--output", dst], check=True)

    with open(dst) as f:
        signals = json.load(f).get("signals", [])

    # Merge
    sfi_recs = []
    for s in signals:
        m = s.get("metrics", {})
        c = s.get("coeffs", {})
        sfi_recs.append(
            {
                "timestamp_ns": s["timestamp_ns"],
                "coherence": m.get("coherence", 0),
                "stability": m.get("stability", 0),
                "entropy": m.get("entropy", 0),
                "lambda_hazard": c.get("lambda_hazard", 0),
            }
        )

    df_sfi = pd.DataFrame(sfi_recs)
    df["timestamp_ns"] = df["timestamp_ns"].astype(np.int64)
    df_sfi["timestamp_ns"] = df_sfi["timestamp_ns"].astype(np.int64)

    merged = pd.merge_asof(
        df.sort_values("timestamp_ns"),
        df_sfi.sort_values("timestamp_ns"),
        on="timestamp_ns",
        direction="backward",
    )

    # Features (Agent S5 logic - NO Resampling)
    merged["close"] = merged["close"].astype(float)

    delta = merged["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    merged["rsi"] = 100 - (100 / (1 + rs))

    merged["log_ret"] = np.log(merged["close"] / merged["close"].shift(1))
    merged["volatility"] = merged["log_ret"].rolling(20).std()

    ema = merged["close"].ewm(span=60).mean()
    merged["dist_ema60"] = (merged["close"] - ema) / ema

    merged.fillna(0, inplace=True)

    # Predict
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_PATH)

    feats = [
        "stability",
        "entropy",
        "coherence",
        "lambda_hazard",
        "rsi",
        "volatility",
        "dist_ema60",
    ]
    X = merged[feats]

    probs = clf.predict_proba(X)[:, 1]

    trades = (probs > 0.55).sum()
    print(f"\n--- Simulation Results (Last 48h) ---")
    print(f"Total Candles: {len(merged)}")
    print(f"Trades Triggered: {trades}")
    print(f"Max Prob: {probs.max():.4f}")

    # DIAGNOSTIC
    print("\nDIAGNOSTIC:")
    print(f"Avg Volatility: {merged['volatility'].mean():.2e}")
    print(f"Max Volatility: {merged['volatility'].max():.2e}")
    high_vol = (merged["volatility"] > 3.23e-5).sum()
    print(f"Candles with Vol > 3.23e-5: {high_vol}")


if __name__ == "__main__":
    p = fetch_recent()
    run_agent_logic(p)
