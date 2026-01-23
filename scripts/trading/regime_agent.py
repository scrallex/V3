#!/usr/bin/env python3
"""
Regime Adaptive Agent (Live)
- Consumes S5 Data from Redis.
- Runs SFI Manifold Generator (Binary).
- Generates Technical Features (RSI, Volatility, EMA).
- Applies Gold Standard XGBoost Models (Pair-Specific).
- Publishes Safe Regime Signals.
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import redis
import xgboost as xgb

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("regime-agent")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BIN_PATH = BASE_DIR / "bin" / "manifold_generator"
MODEL_DIR = BASE_DIR / "models"

# Config
HISTORY_LEN = 3000
SFI_STEP = 1


class RegimeAgent:
    def __init__(self, redis_url, instruments, poll_interval=1.0):
        self.redis = redis.from_url(redis_url)
        self.instruments = instruments
        self.poll_interval = poll_interval
        self.running = True

        self.models = {}
        self.load_models()

        self.last_ts = {inst: 0 for inst in instruments}

    def load_models(self):
        logger.info("Loading Gold Standard Models...")
        for inst in self.instruments:
            model_path = MODEL_DIR / f"model_{inst}.json"
            if model_path.exists():
                try:
                    clf = xgb.XGBClassifier()
                    clf.load_model(model_path)
                    self.models[inst] = clf
                    logger.info(f"Loaded {inst}: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load model for {inst}: {e}")
            else:
                logger.warning(
                    f"No model found for {inst} at {model_path}. Agent will skip."
                )

    def run(self):
        logger.info(f"Starting Regime Agent (S5) for {self.instruments}...")
        while self.running:
            for inst in self.instruments:
                try:
                    if inst in self.models:
                        self.process_instrument(inst)
                except Exception as e:
                    logger.error(f"Error processing {inst}: {e}", exc_info=True)
            time.sleep(self.poll_interval)

    def stop(self):
        self.running = False
        logger.info("Stopping Agent...")

    def process_instrument(self, inst):
        key = f"md:candles:{inst}:S5"
        raw_data = self.redis.zrange(key, -HISTORY_LEN, -1)

        if not raw_data:
            return

        candles = []
        for r in raw_data:
            try:
                if isinstance(r, bytes):
                    r = r.decode()
                c = json.loads(r)
                mid = c.get("mid", {})
                if isinstance(mid, dict):
                     o = float(mid.get("o", 0))
                     h = float(mid.get("h", 0))
                     l = float(mid.get("l", 0))
                     cl = float(mid.get("c", 0))
                else:
                     # Fallback if mid is scalar
                     o = h = l = cl = float(mid)
                
                vol = float(c.get("v", 0))
                ts = int(c.get("t", 0))

                if ts > 0 and c.get("complete", True):
                    candles.append({
                        "timestamp_ns": ts * 1_000_000, 
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": cl,
                        "volume": vol,
                        "price": cl  # Keep price for backward compat in features
                    })
            except:
                continue

        if len(candles) < 100:
            return

        curr_ts = candles[-1]["timestamp_ns"]
        if curr_ts <= self.last_ts[inst]:
            return

        # Run SFI (S5)
        sfi_metrics = self.run_manifold(candles) # Passed candles now has OHLCV
        if not sfi_metrics:
            return
            
        # ... logic continues ...

    def run_manifold(self, candles):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f_in:
            flat_candles = []
            for c in candles:
                flat_candles.append(
                    {
                        "timestamp": c["timestamp_ns"] // 1_000_000,
                        "close": c["close"],
                        "open": c["open"],
                        "high": c["high"],
                        "low": c["low"],
                        "volume": c["volume"],
                    }
                )
            json.dump(flat_candles, f_in)
            src = f_in.name

        dst = src + ".out"

        try:
            subprocess.run(
                [str(BIN_PATH), "--input", src, "--output", dst],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(dst, "r") as f:
                return json.load(f).get("signals", [])
        except:
            return None
        finally:
            if os.path.exists(src):
                os.unlink(src)
            if os.path.exists(dst):
                os.unlink(dst)

    def compute_features(self, candles, sfi_signals):
        # Align
        df_p = pd.DataFrame(candles)
        recs = []
        for s in sfi_signals:
            m = s.get("metrics", {})
            c = s.get("coeffs", {})
            recs.append(
                {
                    "timestamp_ns": s["timestamp_ns"],
                    "coherence": m.get("coherence", 0),
                    "stability": m.get("stability", 0),
                    "entropy": m.get("entropy", 0),
                    "lambda_hazard": c.get("lambda_hazard", 0),
                }
            )
        df_sfi = pd.DataFrame(recs)
        df_p["timestamp_ns"] = df_p["timestamp_ns"].astype(np.int64)
        df_sfi["timestamp_ns"] = df_sfi["timestamp_ns"].astype(np.int64)

        df = pd.merge_asof(
            df_p.sort_values("timestamp_ns"),
            df_sfi.sort_values("timestamp_ns"),
            on="timestamp_ns",
            direction="backward",
        )

        # S5 Features (NO RESAMPLING)
        df["close"] = df["price"].astype(float)

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Volatility
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_ret"].rolling(20).std()

        # EMA
        ema = df["close"].ewm(span=60).mean()
        df["dist_ema60"] = (df["close"] - ema) / ema

        df.fillna(0, inplace=True)
        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis", default="redis://localhost:6379/0")
    parser.add_argument(
        "--pairs", default="EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,USD_CAD,NZD_USD"
    )
    args = parser.parse_args()

    agent = RegimeAgent(args.redis, args.pairs.split(","))

    def shutdown(sig, frame):
        agent.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    agent.run()


if __name__ == "__main__":
    main()
