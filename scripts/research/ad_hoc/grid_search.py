
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

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"] # Focus on Top 3
LOOKBACK_DAYS = 2 
MODEL_DIR = "/app/models"
BIN_PATH = "/app/bin/manifold_generator"
ENTRY_COST_PIPS = 2.0

# Grid
ENTRY_CANDIDATES = [0.60, 0.65, 0.70, 0.75]
EXIT_CANDIDATES = [0.45, 0.50, 0.55] 

logging.basicConfig(level=logging.ERROR)

class GridSearchBot:
    def __init__(self):
        self.connector = OandaConnector(read_only=True)
    
    def load_model(self, instrument):
        model = xgb.XGBClassifier()
        try:
            model.load_model(f"{MODEL_DIR}/model_{instrument}.json")
            return model
        except Exception as e:
            print(f"Model Load Error: {e}")
            return None

    def fetch_data(self, instrument):
        print(f"[{instrument}] Fetching S5 data...", end="", flush=True)
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
                    "timestamp": int(dt.timestamp()), 
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0))
                })
            
            last_dt = datetime.fromisoformat(candles[-1]["time"].replace("Z", "+00:00"))
            if last_dt >= to_time or len(candles) < 10: break
            current_from = last_dt + timedelta(seconds=5)
            
        print(f" Done ({len(all_candles)})")
        return pd.DataFrame(all_candles)

    def run_manifold(self, df):
        input_data = []
        for _, row in df.iterrows():
            input_data.append({
                "timestamp": int(row["timestamp"]),
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"])
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
        except Exception as e:
            print(f"Manifold Error: {e}")
            return []
        finally:
            if os.path.exists(input_path): os.unlink(input_path)
            if os.path.exists(output_path): os.unlink(output_path)

    def compute_features(self, df_s5, signals):
        sig_data = []
        for s in signals:
            mets = s.get("metrics", {})
            coeffs = s.get("coeffs", {})
            ts = int(s.get("timestamp_ns", 0))
            if ts < 10_000_000_000: ts *= 1_000_000_000
            elif ts < 10_000_000_000_000_000: ts *= 1_000
            
            sig_data.append({
                "timestamp_ns": ts,
                "coherence": float(mets.get("coherence", 0)),
                "entropy": float(mets.get("entropy", 0)),
                "stability": float(mets.get("stability", 0)),
                "lambda_hazard": float(coeffs.get("lambda_hazard", 0))
            })
            
        if not sig_data: return pd.DataFrame()
        df_sig = pd.DataFrame(sig_data)
        
        df_merged = pd.merge_asof(
            df_s5.sort_values("timestamp_ns"), df_sig.sort_values("timestamp_ns"),
            on="timestamp_ns", direction="nearest", tolerance=5_000_000_000
        )
        
        df = df_merged.copy()
        
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

    def simulate(self, df, entry_th, exit_th, instrument):
        # CORRECT MULTIPLIER LOGIC
        # JPY pairs (e.g. 150.00) -> 0.01 is a pip. Multiplier 100.
        # USD pairs (e.g. 1.0800) -> 0.0001 is a pip. Multiplier 10000.
        multiplier = 100.0 if "JPY" in instrument else 10000.0
        
        position = 0 
        entry_price = 0
        pnl = 0
        trades = 0
        
        closes = df["close"].values
        probs = df["prob"].values
        
        for i in range(len(df)):
            price = closes[i]
            prob = probs[i]
            
            if position == 0:
                if prob > entry_th:
                    position = 1 
                    entry_price = price
                    trades += 1
                elif prob < (1 - entry_th):
                    position = -1 
                    entry_price = price
                    trades += 1
            
            elif position == 1: # Long
                if prob < exit_th:
                    diff = (price - entry_price) * multiplier
                    pnl += (diff - ENTRY_COST_PIPS) # NET PNL
                    position = 0
            
            elif position == -1: # Short
                if prob > (1 - exit_th):
                    diff = (entry_price - price) * multiplier
                    pnl += (diff - ENTRY_COST_PIPS) # NET PNL
                    position = 0
                    
        return pnl, trades

if __name__ == "__main__":
    bot = GridSearchBot()
    results = {} 
    
    for pair in PAIRS:
        print(f"\nProcessing {pair}...")
        model = bot.load_model(pair)
        if not model: continue
        
        df = bot.fetch_data(pair)
        if df.empty: continue
        
        sigs = bot.run_manifold(df)
        df = bot.compute_features(df, sigs)
        if df.empty: continue
        
        feat_cols = ["stability", "entropy", "coherence", "lambda_hazard", "rsi", "volatility", "dist_ema60"]
        df["prob"] = model.predict_proba(df[feat_cols].values)[:, 1]
        
        print(f"{'Entry':<6} | {'Exit':<6} | {'PnL (Net Pips)':<16} | {'Trades':<6}")
        print("-" * 50)
        
        for entry in ENTRY_CANDIDATES:
            for exit in EXIT_CANDIDATES:
                if exit >= entry: continue 
                
                pnl, trades = bot.simulate(df, entry, exit, pair)
                
                key = (entry, exit)
                current = results.get(key, 0)
                results[key] = current + pnl
                
                print(f"{entry:<6} | {exit:<6} | {pnl:<16.1f} | {trades:<6}")

    print("\n" + "="*50)
    print("AGGREGATE RESULTS (ALL PAIRS)")
    print("="*50)
    
    best_pnl = -99999
    best_cfg = None
    
    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for (entry, exit), pnl in sorted_res:
        print(f"{entry:<6} | {exit:<6} | {pnl:<10.1f}")
        if pnl > best_pnl:
            best_pnl = pnl
            best_cfg = (entry, exit)
            
    if best_cfg:
        print(f"\nWINNER: Entry={best_cfg[0]}, Exit={best_cfg[1]} (Net Pnl: {best_pnl:.1f})")
