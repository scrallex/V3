
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("UnifiedBT")

class UnifiedBacktester:
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
        # Fetch S5 Data (Exact same source as Live Agent)
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
        # Exact logic from regime_agent.py
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

    def compute_features_live_parity(self, df_prices, sfi_signals):
        # ---------------------------------------------------------
        # EXACT LOGIC COPY FROM `regime_agent.py`
        # ---------------------------------------------------------
        
        # 1. Align Signals
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
        
        # Merge
        df = pd.merge_asof(
            df_prices.sort_values("timestamp_ns"),
            df_sfi.sort_values("timestamp_ns"),
            on="timestamp_ns",
            direction="backward",
        )
        
        # 2. Resample to M1 (CRITICAL PARITY STEP)
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
        
        # 3. Features
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

    def strategy_logic_adaptive(self, df, instrument):
        multiplier = 100.0 if "JPY" in instrument else 10000.0
        
        prob = df["prob"].values
        price = df["close"].values
        # haz = df["lambda_hazard"].values
        vol = df["volatility"].values
        
        # Fixed Entry/Exit (Optimized for JPY, let's see if filters help others)
        entry_th = 0.65 # Lowering entry to allow filters to work
        exit_th = 0.45
        
        # Sweep Volatility Filters (High Volatility Regime Focus)
        # Hypothesis: Price Action models fail in low vol (noise). 
        # We want to trade only when the market is moving.
        
        # Volatility is usually around 1e-4 to 5e-4 on S5/M1
        results = {}
        
        # Min Volatility Thresholds
        vol_thresholds = [0.0, 0.00005, 0.00010, 0.00015, 0.00020, 0.00030] 
        
        for min_vol in vol_thresholds:
            position = 0
            entry_price = 0
            total_pnl = 0
            trades = 0
            
            for i in range(len(df)):
                # FILTER: Only trade if Volatility is High Enough
                if vol[i] < min_vol:
                    if position == 0: continue
                    # If in position, we allow hold/exit logic to proceed (don't force close)
                
                p = prob[i]
                pr = price[i]
                
                if position == 0:
                    if p > entry_th:
                        position = 1
                        entry_price = pr
                        trades += 1
                    elif p < (1 - entry_th):
                        position = -1
                        entry_price = pr
                        trades += 1
                
                elif position == 1:
                    if p < exit_th:
                        diff = (pr - entry_price) * multiplier
                        total_pnl += (diff - SPREAD_PIPS)
                        position = 0
                
                elif position == -1:
                    if p > (1 - exit_th):
                        diff = (entry_price - pr) * multiplier
                        total_pnl += (diff - SPREAD_PIPS)
                        position = 0
            
            results[min_vol] = (total_pnl, trades)
            
        return results

    def run(self):
        print(f"{'Pair':<8} | {'MinVol':<8} | {'Net PnL':<10} | {'Trades':<6}")
        print("-" * 42)
        
        for pair in PAIRS:
            if not self.models[pair]: continue
            
            df_s5 = self.fetch_data(pair)
            if df_s5.empty: continue
            
            sigs = self.run_manifold(df_s5)
            df = self.compute_features_live_parity(df_s5, sigs)
            
            feat_cols = ["stability", "entropy", "coherence", "lambda_hazard", "rsi", "volatility", "dist_ema60"]
            df["prob"] = self.models[pair].predict_proba(df[feat_cols].values)[:, 1]
            
            results = self.strategy_logic_adaptive(df, pair)
            
            for vol, (pnl, tr) in results.items():
                print(f"{pair:<8} | {vol:<8} | {pnl:<10.1f} | {tr:<6}")


if __name__ == "__main__":
    bt = UnifiedBacktester()
    bt.run()
