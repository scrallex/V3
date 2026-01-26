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
from datetime import datetime

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
HISTORY_LEN = 60000 # Matching full 3-day backtest window
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
        self.last_signal = {inst: "NEUTRAL" for inst in instruments}

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
            logger.info("Tick.")
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
            logger.info(f"{inst}: No data in Redis.")
            return

        logger.info(f"{inst}: Found {len(raw_data)} chunks in Redis.")

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
                        "price": cl  # Required by compute_features (same as close)
                    })
            except:
                continue

        if len(candles) < 100:
            if len(candles) > 0:
                 logger.info(f"{inst}: Only {len(candles)} valid candles (need 100). Waiting...")
            return

        curr_ts = candles[-1]["timestamp_ns"]
        if curr_ts <= self.last_ts[inst]:
            return

        # FORCE TRIM (Weekend Gap Mitigation)
        # Smart detection failed. We force a 15,000 candle window (approx 21h).
        # This guarantees we only see the current Monday session and exclude the 48h weekend gap.
        # This aligns with the successful Forensic Backtest (1 Day).
        # TIME-BASED TRIM (Weekend Gap Mitigation)
        # Force strict 24h window. 
        # Using count (-15000) failed because sparse data meant 15k candles spanned >3 days (including weekend).
        current_head_ts = candles[-1]["timestamp_ns"]
        cutoff_ts = current_head_ts - (24 * 3600 * 1_000_000_000)
        
        # Filter (assume sorted)
        # Find bisect point or just list comp
        candles = [c for c in candles if c["timestamp_ns"] > cutoff_ts]
        logger.info(f"{inst}: Time-Trimmed to last 24h. Remaining: {len(candles)} candles.")

        
        # Prepare Data for Manifold (Strict 5s Grid)
        candles_for_manifold = candles # Default
        
        if len(candles) > 100:
            df_r = pd.DataFrame(candles)
            df_r["dt"] = pd.to_datetime(df_r["timestamp_ns"], unit="ns")
            df_r.set_index("dt", inplace=True)
            
            # Resample
            df_resampled = df_r.resample("5s").agg({
                "timestamp_ns": "first",
                "open": "first",
                "high": "first",
                "low": "first",
                "close": "last",
                "volume": "sum",
                "price": "last"
            })
            
            # Forward Fill
            df_resampled["close"] = df_resampled["close"].ffill()
            df_resampled["open"] = df_resampled["open"].fillna(df_resampled["close"])
            df_resampled["high"] = df_resampled["high"].fillna(df_resampled["close"])
            df_resampled["low"] = df_resampled["low"].fillna(df_resampled["close"])
            df_resampled["price"] = df_resampled["price"].fillna(df_resampled["close"])
            
            # Volume REMOVED (Set to 0) to match Backtest
            df_resampled["volume"] = 0
            
            # Reconstruct Timestamp
            df_resampled["timestamp_ns"] = df_resampled.index.astype(np.int64)
            
            # Convert back to list of dicts
            candles_for_manifold = df_resampled.dropna(subset=["close"]).to_dict("records")


        # Run SFI on RESAMPLED (Continuous Physics)
        sfi_metrics = self.run_manifold(candles_for_manifold)
        if not sfi_metrics:
            logger.info(f"{inst}: Manifold returned None/Empty.")
            return

        # Compute Features on RESAMPLED (Unified 5s Time Basis)
        # We must use the same time basis (5s) for rolling windows (e.g. Volatility 20)
        # to match the Backtest/Training data. Using sparse original data (12s) distorts the window to 240s.
        try:
            df = self.compute_features(candles_for_manifold, sfi_metrics)
        except Exception as e:
            logger.error(f"{inst}: Feature computation failed: {e}")
            return

        if not sfi_metrics:
            logger.info(f"{inst}: Manifold returned None/Empty.")
            return
        logger.info(f"{inst}: Manifold Success. Metrics: {len(sfi_metrics)}")

        # Features
        try:
            df = self.compute_features(candles, sfi_metrics)
        except Exception as e:
            logger.error(f"{inst}: Feature computation failed: {e}")
            return

        if df.empty:
            logger.warning(f"{inst}: Empty DataFrame after features.")
            return

        # Inference
        feat_cols = [
            "stability",
            "entropy",
            "coherence",
            "lambda_hazard",
            "rsi",
            "volatility",
            "dist_ema60",
        ]
        
        try:
            X = df[feat_cols].values
            probs = self.models[inst].predict_proba(X)[:, 1]
            row = df.iloc[-1]
            prob = probs[-1]
            
            
            if inst == "GBP_USD":
                pass # Debug log removed
            
            # Decision
            # Decision
            current_signal = self.last_signal.get(inst, "NEUTRAL")
            new_signal = "NEUTRAL"
            
            # 1. Entry Logic (High Confidence)
            if prob > 0.65:
                if row["dist_ema60"] > 0:
                    new_signal = "LONG"
                else:
                    new_signal = "SHORT"
            
            # 2. Hold Logic (Hysteresis / Buffer Zone)
            elif prob > 0.55:
                # If we are already in a trade, hold it
                if current_signal != "NEUTRAL":
                    new_signal = current_signal
            
            # 3. Exit (Implicit) implies new_signal remains "NEUTRAL" if prob <= 0.55
            
            # Update State
            self.last_signal[inst] = new_signal
            signal = new_signal

            # DEBUG PARITY
            logger.info(f"{inst} | Prob={prob:.2f} | Haz={row.get('lambda_hazard',0):.2f} | Vol={row.get('volatility',0):.2e} | RSI={row.get('rsi',0):.1f} | Dist={row.get('dist_ema60',0):.2e}")

            logger.info(
                f"{inst} | Haz={row.get('lambda_hazard',0):.2f} | Prob={prob:.2f} | {signal}"
            )



            # Determine Regime Label for UI
            haz = float(row.get("lambda_hazard", 0))
            regime_label = "Hazel"
            if haz < 0.20: regime_label = "LowVol"
            elif haz > 0.50: regime_label = "HighVol"

            # Construct Payload (RICH STATE for Dashboard & Execution)
            payload = {
                "strategy": "Regime_S5",
                "instrument": inst,
                "signal": signal,
                "timestamp": int(row["timestamp_ns"] // 1_000_000), # Millis
                "ts_ms": int(row["timestamp_ns"] // 1_000_000),     # UI expects ts_ms
                "confidence": float(prob),
                "prob": float(prob),                                # UI expects prob
                "hazard": haz,
                "hazard_norm": haz,                                 # UI expects hazard_norm
                "price": float(row["close"]),
                "volatility": float(row.get("volatility", 0)),      # UI expects volatility
                "rsi": float(row.get("rsi", 50)),                   # UI expects rsi
                "regime": regime_label,                             # UI expects regime
                "dist_ema60": float(row.get("dist_ema60", 0))
            }

            # ALWAYS PERSIST STATE (Heartbeat for Dashboard & Execution)
            # This ensures dashboard shows live data and Execution Agent receives "NEUTRAL" to close trades.
            self.redis.set(f"gate:last:{inst}", json.dumps(payload))
            
            # Publish only significant events or keep chatter low? 
            # PubSub is cheap, let's publish all for live stream if needed, 
            # but mainly the KEY set is what matters for the dashboard/poller.
            if signal != "NEUTRAL":
                 self.redis.publish("strategies:regime:signals", json.dumps(payload))
                 logger.info(f"🚀 PUBLISHED: {payload}")
            else:
                 # Log heartbeat occasionally?
                 # logger.debug(f"{inst} Heartbeat: {signal} P={prob:.2f}")
                 pass


        except Exception as e:
            logger.error(f"{inst}: Inference failed: {e}", exc_info=True)

    def run_manifold(self, candles):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f_in:
            flat_candles = []
            for c in candles:
                flat_candles.append(
                    {
                        "timestamp": c["timestamp_ns"] // 1_000_000_000, # Nanos -> Seconds (Match deploy_models.py)
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
            
            ts = int(s["timestamp_ns"])
            if ts < 10_000_000_000: ts *= 1_000_000_000      # Seconds to Nanos
            elif ts < 10_000_000_000_000_000: ts *= 1_000    # Micros to Nanos
            
            recs.append(
                {
                    "timestamp_ns": ts,
                    "coherence": m.get("coherence", 0),
                    "stability": m.get("stability", 0),
                    "entropy": m.get("entropy", 0),
                    "lambda_hazard": c.get("lambda_hazard", 0),
                }
            )
        df_sfi = pd.DataFrame(recs)
        df_p["timestamp_ns"] = df_p["timestamp_ns"].astype(np.int64)
        if not df_sfi.empty:
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
        # logger.info(f"Features Computed. Size={len(df)}")
        if df.empty:
             if not df_p.empty and not df_sfi.empty:
                 logger.info(f"Merge Fail! Price TS: {df_p['timestamp_ns'].iloc[-1]} vs Signal TS: {df_sfi['timestamp_ns'].iloc[-1]}")
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
