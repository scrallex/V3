#!/usr/bin/env python3
"""
04_calc_params.py
Purpose: Sizing & Frequency Analysis.
Simulate portfolio of 7 pairs to determine Capacity and Frequency.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_DIR = Path("/sep/research_lab/models")
DATA_DIR = Path("/sep/research_lab/data")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("SizingLab")


def add_technicals(df):
    df["close"] = df["close"].astype(float)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["log_ret"].rolling(20).std()
    ema60 = df["close"].ewm(span=60).mean()
    df["dist_ema60"] = (df["close"] - ema60) / ema60
    df.fillna(0, inplace=True)
    return df


def run_simulation():
    instruments = [
        "EUR_USD",
        "GBP_USD",
        "USD_JPY",
        "USD_CHF",
        "AUD_USD",
        "USD_CAD",
        "NZD_USD",
    ]

    all_trades = []

    logger.info("loading models & data...")

    for inst in instruments:
        path = DATA_DIR / f"{inst}_rich.parquet"
        if not path.exists():
            continue

        df = pd.read_parquet(path)
        df = add_technicals(df)
        if "structure_entropy" in df.columns:
            df["entropy"] = df["structure_entropy"]

        # Mapping SFI to Features
        # Model trained on: stability, entropy, coherence, lambda_hazard, rsi, volatility, dist_ema60
        # Check col names
        if "stability" not in df.columns and "structure_stability" in df.columns:
            df["stability"] = df["structure_stability"]

        # Load Model
        try:
            clf = xgb.XGBClassifier()
            clf.load_model(MODEL_DIR / f"model_{inst}.json")
        except:
            continue

        features = [
            "stability",
            "entropy",
            "coherence",
            "lambda_hazard",
            "rsi",
            "volatility",
            "dist_ema60",
        ]

        # Check missing
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.warning(f"Skipping {inst} missing cols {missing}")
            continue

        X = df[features]
        probs = clf.predict_proba(X)[:, 1]

        timestamps = df["timestamp_ns"].values
        in_trade_until = 0

        for i in range(len(probs)):
            if timestamps[i] < in_trade_until:
                continue

            if probs[i] > 0.55:
                # Entry
                entry_time = timestamps[i]
                # 60 candles duration (assuming S5 data, 60*5s = 5min)
                # timestamps are NS. 5min = 300 * 1e9
                duration_ns = 300 * 1_000_000_000
                exit_time = entry_time + duration_ns

                all_trades.append({"inst": inst, "start": entry_time, "end": exit_time})
                in_trade_until = exit_time

    if not all_trades:
        logger.error("No trades found!")
        return

    trades_df = pd.DataFrame(all_trades).sort_values("start")

    min_ts = trades_df["start"].min()
    max_ts = trades_df["end"].max()
    days = (max_ts - min_ts) / (1e9 * 60 * 60 * 24)
    total_trades = len(trades_df)
    daily_avg = total_trades / days if days > 0 else 0

    # Concurrency
    events = []
    for _, t in trades_df.iterrows():
        events.append((t["start"], 1))
        events.append((t["end"], -1))

    events.sort()

    current_open = 0
    max_open = 0
    for ts, change in events:
        current_open += change
        max_open = max(max_open, current_open)

    logger.info("-" * 40)
    logger.info(f"Total Trades (14 Days): {total_trades}")
    logger.info(f"Daily Average: {daily_avg:.1f} trades/day")
    logger.info(f"Max Concurrent Positions: {max_open}")
    logger.info("-" * 40)

    rec_nav_unlev = 100.0 / max_open if max_open else 0
    logger.info(f"Target NAV % (No Leverage Buffer): {rec_nav_unlev:.2f}%")


if __name__ == "__main__":
    run_simulation()
