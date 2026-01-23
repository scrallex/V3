#!/usr/bin/env python3
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("GlobalGrid")


def grid_search(df, name):
    df["ema_trend"] = df["close"].ewm(span=60).mean()
    df["trend_dir"] = np.where(df["close"] > df["ema_trend"], 1, -1)
    df["ret_5m"] = np.log(df["close"].shift(-60) / df["close"])

    if "structure_stability" in df.columns:
        df["stability"] = df["structure_stability"]

    stability_thresholds = [0.80, 0.85]
    hazard_thresholds = [0.20, 0.15, 0.10, 0.05]

    results = []

    for stab in stability_thresholds:
        for haz in hazard_thresholds:
            mask = (df["stability"] > stab) & (df["lambda_hazard"] < haz)
            sig = np.zeros(len(df))
            sig[mask & (df["trend_dir"] == 1)] = 1
            sig[mask & (df["trend_dir"] == -1)] = -1

            pnl = df["ret_5m"] * sig
            trades = (np.diff(sig) != 0).sum() / 2
            tot = pnl[sig != 0].sum()
            avg = tot / trades if trades > 10 else -999

            results.append({"s": stab, "h": haz, "avg": avg, "n": int(trades)})

    results.sort(key=lambda x: x["avg"], reverse=True)
    logger.info(f"--- Top 3 for {name} ---")
    for r in results[:3]:
        logger.info(f"S:{r['s']} H:{r['h']} Avg:{r['avg']:.6f} N:{r['n']}")
    return results


def main():
    instruments = ["EUR_USD", "GBP_USD", "USD_JPY"]
    for inst in instruments:
        path = f"/sep/research_lab/data/{inst}_rich.parquet"
        if Path(path).exists():
            df = pd.read_parquet(path)
            grid_search(df, inst)


if __name__ == "__main__":
    main()
