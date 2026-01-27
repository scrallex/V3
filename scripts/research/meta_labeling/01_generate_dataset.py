
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
sys.path.append("/app") 
from scripts.trading.oanda import OandaConnector

# CONFIG
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
LOOKBACK_DAYS = 7
MODEL_DIR = "/app/models"
BIN_PATH = "/app/bin/manifold_generator"
SPREAD_PIPS = 2.0
OUTPUT_FILE = "/tmp/meta_training_data.csv" # Ephemeral /tmp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("MetaMiner")

class MetaDataMiner:
    def __init__(self):
        self.connector = OandaConnector(read_only=True)
        self.models = {}
        for p in PAIRS:
            self.models[p] = self.load_model(p)

    def load_model(self, instrument):
        model = xgb.XGBClassifier()
        try:
            model.load_model(f"{MODEL_DIR}/model_{instrument}.json")
            return model
        except:
            return None

    def fetch_data(self, instrument):
        logger.info(f"[{instrument}] Fetching S5 data ({LOOKBACK_DAYS} Days)...")
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
            if not candles: break
            
            for c in candles:
                dt = datetime.fromisoformat(c["time"].replace("Z", "+00:00"))
                mid = c["mid"]
                all_candles.append({
                    "timestamp_ns": int(dt.timestamp() * 1_000_000_000),
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "volume": float(c.get("volume", 0))
                })
            
            last_dt = datetime.fromisoformat(candles[-1]["time"].replace("Z", "+00:00"))
            if last_dt >= to_time or len(candles) < 10: break
            current_from = last_dt + timedelta(seconds=5)
            
        return pd.DataFrame(all_candles)

    def run_manifold(self, df):
        input_data = []
        for _, row in df.iterrows():
            input_data.append({
                "timestamp": int(row["timestamp_ns"] // 1_000_000_000),
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
                "volume": float(row["volume"])
            })
            
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_in:
            json.dump(input_data, f_in)
            input_path = f_in.name
        output_path = input_path + ".out.json"
        
        try:
            subprocess.run([BIN_PATH, "--input", input_path, "--output", output_path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            with open(output_path, "r") as f_out:
                return json.load(f_out).get("signals", [])
        except:
            return []
        finally:
            if os.path.exists(input_path): os.unlink(input_path)
            if os.path.exists(output_path): os.unlink(output_path)

    def compute_features(self, df_prices, sfi_signals):
        recs = []
        for s in sfi_signals:
            m = s.get("metrics", {})
            c = s.get("coeffs", {})
            ts = int(s["timestamp_ns"])
            if ts < 10_000_000_000: ts *= 1_000_000_000
            elif ts < 10_000_000_000_000_000: ts *= 1_000
            
            recs.append({
                "timestamp_ns": ts,
                "coherence": m.get("coherence", 0),
                "stability": m.get("stability", 0),
                "entropy": m.get("entropy", 0),
                "lambda_hazard": c.get("lambda_hazard", 0),
            })
            
        df_sfi = pd.DataFrame(recs)
        df = pd.merge_asof(
            df_prices.sort_values("timestamp_ns"),
            df_sfi.sort_values("timestamp_ns"),
            on="timestamp_ns",
            direction="backward",
        )
        
        df["dt"] = pd.to_datetime(df["timestamp_ns"], unit="ns")
        df.set_index("dt", inplace=True)
        
        agg = {
            "close": "last",          
            "coherence": "last",
            "entropy": "last",
            "stability": "last",
            "lambda_hazard": "last",
            "timestamp_ns": "last"
        }
        df = df.resample("1min").agg(agg).dropna()
        
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_ret"].rolling(20).std()
        
        ema = df["close"].ewm(span=60).mean()
        df["dist_ema60"] = (df["close"] - ema) / ema
        
        return df.dropna()

    def generate_events(self, df, instrument):
        multiplier = 100.0 if "JPY" in instrument else 10000.0
        
        prob = df["prob"].values
        price = df["close"].values
        
        # Features for Context
        vol = df["volatility"].values
        haz = df["lambda_hazard"].values
        rsi = df["rsi"].values
        ent = df["entropy"].values
        
        events = [] # List of dicts
        
        # Loose thresholds to catch ALL potential signals
        entry_th = 0.55 
        exit_th = 0.45 
        
        position = 0
        entry_idx = 0
        
        for i in range(len(df)):
            p = prob[i]
            pr = price[i]
            
            if position == 0:
                if p > entry_th:
                    position = 1
                    entry_idx = i
                elif p < (1 - entry_th):
                    position = -1
                    entry_idx = i
            
            elif position == 1: # Long
                if p < exit_th:
                    diff = (pr - price[entry_idx]) * multiplier
                    pnl = diff - SPREAD_PIPS
                    
                    # Record Event
                    events.append({
                        "pair": instrument,
                        "prob": prob[entry_idx],
                        "volatility": vol[entry_idx],
                        "hazard": haz[entry_idx],
                        "rsi": rsi[entry_idx],
                        "entropy": ent[entry_idx],
                        "pnl": pnl,
                        "label": 1 if pnl > 0 else 0
                    })
                    position = 0
            
            elif position == -1: # Short
                if p > (1 - exit_th):
                    diff = (price[entry_idx] - pr) * multiplier
                    pnl = diff - SPREAD_PIPS
                    
                    events.append({
                        "pair": instrument,
                        "prob": 1.0 - prob[entry_idx], # Normalize to 'Confidence in Signal'
                        "volatility": vol[entry_idx],
                        "hazard": haz[entry_idx],
                        "rsi": rsi[entry_idx],
                        "entropy": ent[entry_idx],
                        "pnl": pnl,
                        "label": 1 if pnl > 0 else 0
                    })
                    position = 0
                    
        return events

    def run(self):
        all_events = []
        for pair in PAIRS:
            print(f"Mining {pair}...")
            model = self.models[pair]
            if not model: continue
            
            df_s5 = self.fetch_data(pair)
            if df_s5.empty: continue
            
            sigs = self.run_manifold(df_s5)
            df = self.compute_features(df_s5, sigs)
            
            feat_cols = ["stability", "entropy", "coherence", "lambda_hazard", "rsi", "volatility", "dist_ema60"]
            df["prob"] = model.predict_proba(df[feat_cols].values)[:, 1]
            
            events = self.generate_events(df, pair)
            print(f" -> Found {len(events)} events.")
            all_events.extend(events)
            
        df_out = pd.DataFrame(all_events)
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"Exported {len(df_out)} events to {OUTPUT_FILE}")

if __name__ == "__main__":
    miner = MetaDataMiner()
    miner.run()
