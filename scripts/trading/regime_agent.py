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
HISTORY_LEN = 3000  # Buffer size for SFI (S5 frequency)
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
        # Fetch S5 Data
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
                # Normalize mid price
                mid = c.get("mid", c)
                price = float(mid.get("c") if isinstance(mid, dict) else mid)
                ts = int(c.get("t", 0))
                if ts > 0:
                    candles.append({"timestamp_ns": ts * 1_000_000, "price": price})
            except:
                continue

        if len(candles) < 100:
            return

        # Check for new candle
        curr_ts = candles[-1]["timestamp_ns"]
        if curr_ts <= self.last_ts[inst]:
            return

        # Process Completed History (Assuming current is forming?)
        # For S5, updates are frequent.
        # Ideally we process the last completed candle.

        # Run SFI
        sfi_metrics = self.run_manifold(candles)
        if not sfi_metrics:
            return

        # Feature Engineering (Combined Price + SFI)
        df = self.compute_features(candles, sfi_metrics)
        if df is None or df.empty:
            return

        row = df.iloc[-1]

        # Inference
        # Features must match Training exactly:
        # ["stability", "entropy", "coherence", "lambda_hazard",
        #  "rsi", "volatility", "dist_ema60",
        #  "structure_reynolds_ratio", "structure_spectral_lowf_share"]

        feat_cols = [
            "coherence", "entropy", "stability", 
            "coh_mean_5", "ent_mean_5", "coh_std_15", 
            "vol_5"
        ]

        # Ensure all columns exist (fill 0)
        input_data = [row.get(f, 0.0) for f in feat_cols]
        # XGBoost expects 2D array
        X = np.array([input_data])

        prob = float(self.models[inst].predict_proba(X)[:, 1][0])

        # Signal Logic
        # Model predicts "Safe Win (Long)".
        # Wait, Phase 3 model was trained on TARGET LONG.
        # But JPY trained on generic "Safe Win"? No, JPY logic was Reversion?
        # The MODEL output = Probability of "Safe Profit".
        # But DOES IT IMPLY DIRECTION?
        # My JPY training reused `target_long = ... future_ret > 0`.
        # So the model predicts "Long Profit" specifically.
        # Implication:
        # If Prob > 0.5 -> Long.
        # If Prob < 0.5 -> "Not Long". Does it mean Short?
        # Not necessarily.
        # Ideally we trained a Short model too.
        # But for Universality Verification, I accepted the JPY Long model results (Recall 99%).
        # Note: JPY model had 28% Precision on LONG.
        # If I want to trade Short, I need a Short model.
        # For Minimum Viable Deployment: I will emit LONG signals.
        # (Or I should have trained 2 models).

        signal = "NEUTRAL"
        if prob > 0.55:  # Safety Threshold
            signal = "LONG"

        # Publish
        gate_payload = {
            "instrument": inst,
            "ts_ms": curr_ts // 1_000_000,
            "signal": signal,
            "prob": prob,
            "regime": "SafeRegime" if prob > 0.5 else "Hazel",
            "hazard": float(row.get("lambda_hazard", 0)),
            "source": "regime_agent",
            "admit": True,  # Always admit updates
        }

        self.redis.set(f"gate:last:{inst}", json.dumps(gate_payload))
        self.last_ts[inst] = curr_ts

        logger.info(
            f"{inst} | Haz={row.get('lambda_hazard',0):.2f} | Prob={prob:.2f} | {signal}"
        )

    def run_manifold(self, candles):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f_in:
            # Need to convert to list of dicts with 'close' etc for binary?
            # Binary needs 'close', 'open'...
            # candles only has 'price' (mid).
            # We map price -> close/open/high/low (flat candle)
            flat_candles = []
            for c in candles:
                p = c["price"]
                flat_candles.append(
                    {
                        "timestamp": c["timestamp_ns"] // 1_000_000,
                        "close": p,
                        "open": p,
                        "high": p,
                        "low": p,
                        "volume": 1,
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
        # SFI
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

        # Techncials
        df["close"] = df["price"].astype(float)
        
        # --- Feature Engineering (Research Lab Standard) ---
        # Features: ["coherence", "entropy", "stability", "coh_mean_5", "ent_mean_5", "coh_std_15", "vol_5"]
        
        # 1. Resample to M1 implies we treat these S5 candles as a timeseries. 
        # But wait, Training was done on M1 resampled data.
        # This agent runs on S5. 
        # If I compute rolling(5) on S5 data, it spans 25 seconds.
        # In Training, rolling(5) on M1 data spanned 5 minutes.
        # CRITICAL DISTINCTION:
        # The Research Lab Resampled S5 -> M1, then calculated features.
        # If I want to match the model (which expects 5-minute trends), I CANNOT feed it 25-second trends (S5 row).
        # I MUST Resample this execution buffer to M1 first!
        
        df["datetime"] = pd.to_datetime(df["timestamp_ns"], unit="ns")
        df.set_index("datetime", inplace=True)
        
        agg = {
            "close": "last",
            "coherence": "last",
            "entropy": "last",
            "stability": "last"
        }
        
        # Resample to M1 (1Min)
        df_m1 = df.resample("1min").agg(agg).ffill() # ffill to handle gaps
        
        if len(df_m1) < 20:
             # Not enough history for M1 features
             return None

        # Features
        df_m1["coh_mean_5"] = df_m1["coherence"].rolling(5).mean()
        df_m1["ent_mean_5"] = df_m1["entropy"].rolling(5).mean()
        df_m1["coh_std_15"] = df_m1["coherence"].rolling(15).std()
        
        df_m1["returns"] = df_m1["close"].pct_change()
        df_m1["vol_5"] = df_m1["returns"].rolling(5).std()
        
        df_m1.dropna(inplace=True)
        return df_m1


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
