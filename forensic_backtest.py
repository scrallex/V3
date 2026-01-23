
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta, timezone
import tempfile
import subprocess

# Add path for OandaConnector
sys.path.append("/sep")
from scripts.trading.oanda import OandaConnector

# Config
PAIRS = ["EUR_USD"]
LOOKBACK_DAYS = 2
MODEL_DIR = "/sep/models"
BIN_PATH = "/sep/bin/manifold_generator"
OUTPUT_DIR = "/sep/logs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
logger = logging.getLogger("forensic")

class ForensicBacktest:
    def __init__(self):
        self.connector = OandaConnector(read_only=True)
        self.model = xgb.XGBClassifier()
        # Load the LIVE model for EUR_USD
        self.model.load_model(f"{MODEL_DIR}/model_EUR_USD.json")
        logger.info("Loaded Live Model: model_EUR_USD.json")

    def fetch_data(self, instrument):
        logger.info(f"Fetching {LOOKBACK_DAYS} days of S5 data for {instrument}...")
        to_time = datetime.now(timezone.utc)
        from_time = to_time - timedelta(days=LOOKBACK_DAYS)
        
        all_candles = []
        current_from = from_time
        
        while current_from < to_time:
            candles = self.connector.get_candles(
                instrument,
                granularity="S5",
                count=5000,
                from_time=current_from.isoformat().replace("+00:00", "Z"),
                price="M"
            )
            
            if not candles:
                break
                
            for c in candles:
                t_str = c["time"]
                dt = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
                ts_ns = int(dt.timestamp() * 1_000_000_000)
                mid = c["mid"]
                
                all_candles.append({
                    "timestamp_ns": ts_ns,
                    "timestamp": int(dt.timestamp()), 
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"])
                })
            
            last_dt = datetime.fromisoformat(candles[-1]["time"].replace("Z", "+00:00"))
            if last_dt >= to_time or len(candles) < 10:
                break
            current_from = last_dt + timedelta(seconds=5)
            
        logger.info(f"Fetched {len(all_candles)} S5 candles.")
        return pd.DataFrame(all_candles)

    def run_manifold(self, df):
        logger.info("Running Manifold (S5 High Fidelity)...")
        input_data = []
        for _, row in df.iterrows():
            input_data.append({
                "timestamp": int(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"])
            })
            
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_in:
            json.dump(input_data, f_in)
            input_path = f_in.name
        output_path = input_path + ".out.json"
        
        try:
            subprocess.run([BIN_PATH, "--input", input_path, "--output", output_path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            with open(output_path, "r") as f_out:
                res = json.load(f_out)
            return res.get("signals", [])
        finally:
            if os.path.exists(input_path): os.unlink(input_path)
            if os.path.exists(output_path): os.unlink(output_path)

    def resample_and_compute_features(self, df_s5, signals):
        logger.info("Resampling S5 -> M1 & Computing RAW Features...")
        
        # 1. Merge S5 Signals
        sig_data = []
        for s in signals:
            mets = s.get("metrics", {})
            coeffs = s.get("coeffs", {})
            ts = s.get("timestamp_ns", 0)
            if ts < 10_000_000_000: ts *= 1_000_000_000 # Fix seconds
            
            sig_data.append({
                "timestamp_ns": int(ts),
                "coherence": float(mets.get("coherence", 0)),
                "entropy": float(mets.get("entropy", 0)),
                "stability": float(mets.get("stability", 0)),
                "lambda_hazard": float(coeffs.get("lambda_hazard", 0))
            })
            
        df_sig = pd.DataFrame(sig_data)
        df_merged = pd.merge_asof(
            df_s5.sort_values("timestamp_ns"),
            df_sig.sort_values("timestamp_ns"),
            on="timestamp_ns",
            direction="nearest",
            tolerance=5000000000
        )
        
        # 2. Resample to M1
        df_merged["dt"] = pd.to_datetime(df_merged["timestamp_ns"], unit="ns")
        df_merged.set_index("dt", inplace=True)
        
        agg = {
            "close": "last",
            "coherence": "last",
            "entropy": "last",
            "stability": "last",
            "lambda_hazard": "last"
        }
        df_m1 = df_merged.resample("1min").agg(agg).dropna()
        
        # 3. Compute Technicals (RAW, exactly like Live Agent)
        # RSI 14
        delta = df_m1["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df_m1["rsi"] = 100 - (100 / (1 + rs))

        # Volatility 20 (Log Returns)
        df_m1["log_ret"] = np.log(df_m1["close"] / df_m1["close"].shift(1))
        df_m1["volatility"] = df_m1["log_ret"].rolling(20).std()

        # EMA 60
        ema = df_m1["close"].ewm(span=60).mean()
        df_m1["dist_ema60"] = (df_m1["close"] - ema) / ema
        
        df_m1.dropna(inplace=True)
        return df_m1

    def run_inference(self, df):
        logger.info("Running Inference (Target: model_EUR_USD.json)...")
        feat_cols = [
            "stability", "entropy", "coherence", "lambda_hazard",
            "rsi", "volatility", "dist_ema60"
        ]
        
        # Prepare X (Raw)
        X = df[feat_cols].values
        
        # Predict
        probs = self.model.predict_proba(X)[:, 1]
        df["prob"] = probs
        
        trades = df[df["prob"] > 0.55]
        
        logger.info("="*40)
        logger.info(f"FORENSIC RESULTS: EUR_USD (Last {LOOKBACK_DAYS} Days)")
        logger.info(f"Total Candles (M1): {len(df)}")
        logger.info(f"Total TRADES Triggered: {len(trades)}")
        logger.info(f"Max Prob: {df['prob'].max():.4f}")
        logger.info(f"Avg Coherence: {df['coherence'].mean():.4f}")
        logger.info(f"Avg Hazard: {df['lambda_hazard'].mean():.4f}")
        logger.info("="*40)
        
        if not trades.empty:
             print("TRADES FOUND:")
             print(trades[["close", "prob", "coherence", "lambda_hazard", "volatility"]])
        else:
             print("NO TRADES FOUND (Confirmed Live Behavior)")

if __name__ == "__main__":
    fb = ForensicBacktest()
    df_s5 = fb.fetch_data("EUR_USD")
    signals = fb.run_manifold(df_s5)
    df_m1 = fb.resample_and_compute_features(df_s5, signals)
    fb.run_inference(df_m1)
