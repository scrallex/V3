
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

# Reuse OANDA connector
sys.path.append("/app") 
from scripts.trading.oanda import OandaConnector

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]
LOOKBACK_DAYS = 3 # 2 days train, 1 day test
BIN_PATH = "/app/bin/manifold_generator"
MODEL_DIR = "/app/models"

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("calibration")

class Calibrator:
    def __init__(self):
        self.connector = OandaConnector(read_only=True)

    def fetch_data(self, instrument):
        print(f"[{instrument}] Fetching...", end="", flush=True)
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
                dt = datetime.fromisoformat(c["time"].replace("Z", "+00:00"))
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
            
        print(f" {len(all_candles)}")
        return pd.DataFrame(all_candles)

    def run_manifold_and_features(self, df):
        # 1. Manifold Input
        input_data = []
        for _, row in df.iterrows():
            input_data.append({
                "timestamp": int(row["timestamp"]),
                "open": row["open"], "high": row["high"], 
                "low": row["low"], "close": row["close"]
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
            signals = res.get("signals", [])
        except:
            signals = []
        finally:
            if os.path.exists(input_path): os.unlink(input_path)
            if os.path.exists(output_path): os.unlink(output_path)

        if not signals: return pd.DataFrame()

        # 2. Resample / Featurize
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
            df.sort_values("timestamp_ns"),
            df_sig.sort_values("timestamp_ns"),
            on="timestamp_ns",
            direction="nearest",
            tolerance=5000000000
        )
        
        df = df_merged.copy()
        
        # Technicals
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_ret"].rolling(20).std()
        
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        ema = df["close"].ewm(span=60).mean()
        df["dist_ema60"] = (df["close"] - ema) / ema

        # Target Creation (15m forward return)
        df["target_return"] = df["close"].shift(-180) / df["close"] - 1.0
        df["target"] = (df["target_return"] > 0.00015).astype(int) # Beat very small spread
        
        df.dropna(inplace=True)
        return df

    def calibrate(self, df):
        # Split train/test (70/30)
        split = int(len(df) * 0.7)
        train = df.iloc[:split]
        test = df.iloc[split:]
        
        feat_cols = ["stability", "entropy", "coherence", "lambda_hazard", "rsi", "volatility", "dist_ema60"]
        
        X_train = train[feat_cols].values
        y_train = train["target"].values
        X_test = test[feat_cols].values
        y_test = test["target"].values
        
        pos_ratio = sum(y_train) / len(y_train)
        if pos_ratio < 0.001:
            print("  -> SKIP (No trend targets)")
            return None, None
            
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, eval_metric="logloss")
        model.fit(X_train, y_train)
        
        # Optimize Threshold on Train
        probs_train = model.predict_proba(X_train)[:, 1]
        best_thresh = 0.5
        best_acc = 0.0
        
        for t in np.arange(0.5, 0.7, 0.01):
            preds = (probs_train > t).astype(int)
            if sum(preds) == 0: continue
            prec = np.mean(y_train[preds==1])
            if prec > best_acc and sum(preds) > 10:
                best_acc = prec
                best_thresh = t

        # Evaluate on Test
        probs_test = model.predict_proba(X_test)[:, 1]
        preds_test = (probs_test > best_thresh).astype(int)
        n_trades = sum(preds_test)
        
        if n_trades > 0:
            win_rate = np.mean(y_test[preds_test==1])
            avg_return = np.mean(test.loc[preds_test==1, "target_return"])
            print(f"  -> FOUND: Trades={n_trades} | WR={win_rate:.2%} | Return={avg_return:.5f}")
            return model, best_thresh
        else:
            print(f"  -> NO TRADES")
            return None, None


if __name__ == "__main__":
    calib = Calibrator()
    print("DEPLOYING CALIBRATED MODELS...")
    
    for pair in PAIRS:
        try:
            df = calib.fetch_data(pair)
            if len(df) < 200: continue
            
            df_feat = calib.run_manifold_and_features(df)
            if df_feat.empty: continue
            
            model, thresh = calib.calibrate(df_feat)
            if model:
                # SAVE MODEL to TMP (Volume is RO)
                save_path = f"/tmp/model_{pair}.json"
                print(f"  -> SAVING to {save_path}")
                # Use booster save for robustness against sklearn wrapper issues
                try:
                    model.get_booster().save_model(save_path)
                    print(f"     [OK] Saved via Booster")
                except Exception as e:
                    print(f"     [FAIL] Booster Save: {e}")
                    # Fallback
                    try:
                        model.save_model(save_path)
                        print(f"     [OK] Saved via Wrapper")
                    except Exception as e2:
                        print(f"     [FAIL] Wrapper Save: {e2}")
        except Exception as e:
            print(f"Error on {pair}: {e}")
