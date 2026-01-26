
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
sys.path.append("/app") # /app in container
from scripts.trading.oanda import OandaConnector

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]
LOOKBACK_DAYS = 1 # Focus on immediate context
MODEL_DIR = "/app/models" # Container path
BIN_PATH = "/app/bin/manifold_generator" # Container path
OUTPUT_DIR = "/app/logs"

logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s :: %(message)s") # Reduce noise
logger = logging.getLogger("forensic")

class ForensicBacktest:
    def __init__(self):
        self.connector = OandaConnector(read_only=True)
        self.model = xgb.XGBClassifier()

    def load_model_for_pair(self, instrument):
        model = xgb.XGBClassifier()
        model_path = f"{MODEL_DIR}/model_{instrument}.json"
        try:
            model.load_model(model_path)
            return model
        except Exception as e:
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
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0))
                })
            
            last_dt = datetime.fromisoformat(candles[-1]["time"].replace("Z", "+00:00"))
            if last_dt >= to_time or len(candles) < 10:
                break
            current_from = last_dt + timedelta(seconds=5)
            
        print(f" Done ({len(all_candles)} candles)")
        return pd.DataFrame(all_candles)

    def run_manifold(self, df):
        # logger.info("Running Manifold...")
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
        except Exception as e:
            return []
        finally:
            if os.path.exists(input_path): os.unlink(input_path)
            if os.path.exists(output_path): os.unlink(output_path)

    def resample_and_compute_features(self, df_s5, signals):
        # 1. Merge S5 Signals
        sig_data = []
        for s in signals:
            mets = s.get("metrics", {})
            coeffs = s.get("coeffs", {})
            ts = s.get("timestamp_ns", 0)
            if ts < 10_000_000_000: ts *= 1_000_000_000
            elif ts < 10_000_000_000_000_000: ts *= 1_000
            
            sig_data.append({
                "timestamp_ns": int(ts),
                "coherence": float(mets.get("coherence", 0)),
                "entropy": float(mets.get("entropy", 0)),
                "stability": float(mets.get("stability", 0)),
                "lambda_hazard": float(coeffs.get("lambda_hazard", 0))
            })
            
        df_sig = pd.DataFrame(sig_data)
        if df_sig.empty: return pd.DataFrame()

        df_merged = pd.merge_asof(
            df_s5.sort_values("timestamp_ns"),
            df_sig.sort_values("timestamp_ns"),
            on="timestamp_ns",
            direction="nearest",
            tolerance=5000000000
        )
        
        df_features = df_merged.copy()
        
        # 3. Compute Technicals (RAW S5)
        delta = df_features["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df_features["rsi"] = 100 - (100 / (1 + rs))

        df_features["log_ret"] = np.log(df_features["close"] / df_features["close"].shift(1))
        df_features["volatility"] = df_features["log_ret"].rolling(20).std()

        ema = df_features["close"].ewm(span=60).mean()
        df_features["dist_ema60"] = (df_features["close"] - ema) / ema
        
        df_features.dropna(inplace=True)
        return df_features

    def run_inference(self, df, model):
        feat_cols = [
            "stability", "entropy", "coherence", "lambda_hazard",
            "rsi", "volatility", "dist_ema60"
        ]
        
        X = df[feat_cols].values
        probs = model.predict_proba(X)[:, 1]
        df["prob"] = probs
        
        trades = df[df["prob"] > 0.65] # Match Live Threshold
        return trades, df

if __name__ == "__main__":
    fb = ForensicBacktest()
    
    print("\n" + "="*80)
    print(f"FORENSIC AUDIT: LAST {LOOKBACK_DAYS} DAYS | {len(PAIRS)} INSTRUMENTS")
    print("="*80)
    print(f"{'Instrument':<10} | {'Candles':<8} | {'Trades':<6} | {'MaxProb':<8} | {'AvgHaz':<8} | {'AvgVol':<10}")
    print("-" * 80)
    
    for pair in PAIRS:
        model = fb.load_model_for_pair(pair)
        if not model:
            print(f"{pair:<10} | ERROR LOAD | -      | -        | -        | -")
            continue
            
        df_s5 = fb.fetch_data(pair)
        if df_s5.empty:
            print(f"{pair:<10} | NO DATA    | -      | -        | -        | -")
            continue

        signals = fb.run_manifold(df_s5)
        if not signals:
            print(f"{pair:<10} | MANIFOLD X | -      | -        | -        | -")
            continue

        df_feat = fb.resample_and_compute_features(df_s5, signals)
        if df_feat.empty:
            print(f"{pair:<10} | FEAT FAIL  | -      | -        | -        | -")
            continue

        trades, df_res = fb.run_inference(df_feat, model)
        
        max_prob = df_res['prob'].max()
        print(f"{pair:<10} | {len(df_s5):<8} | {len(trades):<6} | {max_prob:<8.4f} | {df_res['lambda_hazard'].mean():<8.4f} | {df_res['volatility'].mean():<10.2e}")
        
        if not df_res.empty:
            # Find row with max prob
            idx = df_res['prob'].idxmax()
            r = df_res.loc[idx]
            print(f"   -> PEAK: Prob={r['prob']:.4f} | Haz={r['lambda_hazard']:.4f} | Vol={r['volatility']:.2e} | RSI={r['rsi']:.1f} | Dist={r['dist_ema60']:.2e}")
            
    print("="*80)
