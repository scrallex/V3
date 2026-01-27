
import os
import sys
import json
import logging
import subprocess
import tempfile
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta, timezone

# Add path for OandaConnector
sys.path.append("/sep")
sys.path.append("/app")
from scripts.trading.oanda import OandaConnector

# Configuration
PAIRS = ["EUR_USD", "GBP_USD", "AUD_USD"]  # Top liquid pairs
LOOKBACK_DAYS = 5  # Expanded window for robustness
MODEL_DIR = "/app/models"
BIN_PATH = "/app/bin/manifold_generator"

logging.basicConfig(level=logging.ERROR) # Only show errors to keep output clean for stats
logger = logging.getLogger("sweeper")

class ParameterSweep:
    def __init__(self):
        self.connector = OandaConnector(read_only=True)
        self.models = {}
        self.meta = {}
        self.output_dir = "/app/logs"
        
        # SEARCH GRID
        self.GRID_ENTRY = [0.55, 0.60, 0.65, 0.70, 0.75]
        self.GRID_EXIT_DELTA = [0.05, 0.10, 0.15] # Exit = Entry - Delta
        self.GRID_REGIME = ["ALL", "HighVol", "LowVol"] 

    def load_artifacts(self):
        try:
            with open(f"{MODEL_DIR}/feature_builder_meta.json", "r") as f:
                self.meta = json.load(f)
            
            self.model_low = xgb.XGBClassifier()
            self.model_low.load_model(f"{MODEL_DIR}/model_low_vol.json")
            
            self.model_high = xgb.XGBClassifier()
            self.model_high.load_model(f"{MODEL_DIR}/model_high_vol.json")
            return True
        except Exception as e:
            print(f"Artifact Load Error: {e}")
            return False

    def fetch_data(self, instrument):
        print(f"Fetching {LOOKBACK_DAYS} days for {instrument}...", end="", flush=True)
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
                ts_ns = int(dt.timestamp() * 1_000_000_000)
                mid = c["mid"]
                
                all_candles.append({
                    "timestamp_ns": ts_ns,
                    "timestamp": int(dt.timestamp()), 
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": float(c["volume"])
                })
            
            last_dt = datetime.fromisoformat(candles[-1]["time"].replace("Z", "+00:00"))
            if last_dt >= to_time or len(candles) < 10: break
            current_from = last_dt + timedelta(seconds=5)
            
        df = pd.DataFrame(all_candles)
        print(f" Done. {len(df)} candles.")
        return df

    def run_manifold(self, df):
        # Prepare Input JSON
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
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(output_path, "r") as f_out:
                return json.load(f_out).get("signals", [])
        except:
            return []
        finally:
            if os.path.exists(input_path): os.unlink(input_path)
            if os.path.exists(output_path): os.unlink(output_path)

    def resample_and_features(self, df_s5, signals):
        # 1. Resample Signals to M1
        sig_data = []
        for s in signals:
            try:
                # Parsing timestamp robustly
                ts_val = s.get("timestamp_ns", 0)
                if ts_val == 0: ts_val = s.get("timestamp", 0) * 1_000_000_000
                elif ts_val < 10_000_000_000: ts_val *= 1_000_000_000
                elif ts_val < 10_000_000_000_000_000: ts_val *= 1_000
                
                mets = s.get("metrics", {})
                coeffs = s.get("coeffs", {})
                
                # FULL MANIFOLD METRIC SET
                sig_data.append({
                    "timestamp_ns": int(ts_val),
                    "structure_coherence": float(mets.get("coherence", 0)),
                    "structure_stability": float(mets.get("stability", 0)),
                    "structure_entropy": float(mets.get("entropy", 0)),
                    "structure_hazard": float(coeffs.get("lambda_hazard", 0)),
                    "structure_rupture": float(mets.get("rupture", 0)),
                    "structure_reynolds_ratio": float(mets.get("reynolds_ratio", 0)),
                    "structure_pinned_alignment": float(mets.get("pinned_alignment", 1)),
                    "structure_domain_wall_ratio": float(mets.get("domain_wall_ratio", 0)),
                })
            except: continue
            
        if not sig_data: return pd.DataFrame()
        
        df_sig = pd.DataFrame(sig_data)
        df_sig["timestamp_ns"] = df_sig["timestamp_ns"].astype("int64")
        df_s5["timestamp_ns"] = df_s5["timestamp_ns"].astype("int64")
        
        df_merged = pd.merge_asof(
            df_s5.sort_values("timestamp_ns"),
            df_sig.sort_values("timestamp_ns"),
            on="timestamp_ns", 
            direction="nearest",
            tolerance=5000000000
        )
        
        df_merged["dt"] = pd.to_datetime(df_merged["timestamp_ns"], unit="ns")
        df_merged.set_index("dt", inplace=True)
        
        # STRICT M1 RESAMPLING (Aggregating all structure columns)
        agg_cols = {
            "close": "last",
            "timestamp_ns": "last"
        }
        for col in df_sig.columns:
            if col != "timestamp_ns":
                agg_cols[col] = "last"
                
        df_m1 = df_merged.resample("1min").agg(agg_cols).dropna()
        df = df_m1.copy()
        
        # -------------------------------------------------------------
        # FULL FEATURE ENGINEERING (Targeting 27 Features)
        # -------------------------------------------------------------
        
        # 1. Base Technicals
        df["close"] = df["close"].astype(float)
        
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_ret"].rolling(20).std()

        ema = df["close"].ewm(span=60).mean()
        df["dist_ema60"] = (df["close"] - ema) / ema

        # 2. Advanced Structure Features (Matches validate_v3.py)
        # Coherence Delta
        df["coherence_delta_10m"] = df["structure_coherence"] - df["structure_coherence"].shift(10)
        
        # Stability vs 30m
        stab_mean_30 = df["structure_stability"].rolling(window=30, min_periods=1).mean()
        df["stability_vs_30m_avg"] = df["structure_stability"] - stab_mean_30
        
        # Hazard ZScore 60m
        haz_mean_60 = df["structure_hazard"].rolling(window=60, min_periods=1).mean()
        haz_std_60 = df["structure_hazard"].rolling(window=60, min_periods=1).std().replace(0, 1)
        df["hazard_zscore_60m"] = (df["structure_hazard"] - haz_mean_60) / haz_std_60
        
        # Regime Relative (Adaptive)
        for col in ["structure_coherence", "structure_stability", "structure_hazard"]:
             base_col = col.replace("structure_", "")
             target = f"{base_col}_minus_regime_mean"
             rolling_mean = df[col].rolling(window=60, min_periods=1).mean()
             df[target] = df[col] - rolling_mean

        # Interactions
        df["interaction_high_rupture_pos_domain"] = 0.0
        df["interaction_low_hazard_neg_reynolds"] = 0.0
        
        # 3. Z-Scoring ALL Numeric Features (Meta based)
        # Using the exact list from validation script logic (all numeric cols)
        # Or simpler: Just ensure we iterate all available columns that might be features.
        # But we need to use the exact names expected by the model.
        # The model uses the features in self.meta["numeric_features"]
        
        if not self.meta:
             # Fallback if meta missing (shouldn't happen)
             pass
        else:
             feats = self.meta.get("numeric_features", [])
             for f in feats:
                 if f in df.columns:
                     roll_mean = df[f].rolling(60).mean()
                     roll_std = df[f].rolling(60).std().replace(0, 1)
                     df[f] = (df[f] - roll_mean) / roll_std
                 elif f not in df.columns:
                     df[f] = 0.0 # Missing feature fill

        df["instrument_code"] = 0.0
        df["regime_code"] = 0.0
        
        return df.dropna()

    def inference(self, df):
        if df.empty: return df
        
        # Use dynamic features from meta (matches Validate V3)
        cols = self.meta.get("numeric_features", []) + ["instrument_code", "regime_code"]
        # Ensure all cols exist
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
                
        # Reorder to match model expectation
        X = df[cols].values
        
        # Hazard Split
        h_col = "structure_hazard" # New naming
        if h_col not in df.columns: h_col = "lambda_hazard" # Fallback
        
        # Norm Hazard (Approx) used for splitting in validate_v3
        # In validate_v3 we calculated norm_hazard explicitly. 
        # Here we already z-scored everything including hazard.
        # But 'structure_hazard' column in DF is Z-Scored now? 
        # Yes, loop above z-scored it.
        # So mask_high = df[h_col] >= 0.0 is correct.
        
        mask_high = df[h_col] >= 0.0
        mask_low = ~mask_high
        
        df["prob"] = 0.0
        df["regime_label"] = "LowVol"
        df.loc[mask_high, "regime_label"] = "HighVol"
        
        if mask_high.any():
            df.loc[mask_high, "prob"] = self.model_high.predict_proba(X[mask_high])[:, 1]
        
        if mask_low.any():
            df.loc[mask_low, "prob"] = self.model_low.predict_proba(X[mask_low])[:, 1]
            
        return df

    def run_sweep(self):
        if not self.load_artifacts(): return

        results = []

        for pair in PAIRS:
            print(f"\n--- Processing {pair} ---")
            df_s5 = self.fetch_data(pair)
            if df_s5.empty or len(df_s5) < 1000: continue
            
            s5_signals = self.run_manifold(df_s5)
            if not s5_signals: continue
            
            df_inf = self.resample_and_features(df_s5, s5_signals)
            if df_inf.empty: continue
            
            df_inf = self.inference(df_inf)
            
            # THE SWEEP LOOP
            for entry_thresh in self.GRID_ENTRY:
                for exit_delta in self.GRID_EXIT_DELTA:
                    exit_thresh = entry_thresh - exit_delta
                    
                    for regime_filter in self.GRID_REGIME:
                        
                        # SIMULATE
                        trades = []
                        position = 0 # 1 Long, -1 Short
                        entry_price = 0.0
                        
                        df_sim = df_inf.copy()
                        
                        for i, row in df_sim.iterrows():
                            # Filter Check
                            if regime_filter != "ALL" and row["regime_label"] != regime_filter:
                                # Start of loop, forcing neutral isn't right if holding.
                                # If regime changes while holding, do we close? 
                                # Let's assume Filter applies to ENTRY only.
                                can_enter = False
                            else:
                                can_enter = True
                                
                            prob = row["prob"]
                            price = row["close"]
                            dist = row["dist_ema60"] # Normalised dist!
                            
                            # Signal Logic (Trend: Price > EMA -> Long)
                            # Since dist is z-scored, > 0 means above mean dist. 
                            # But wait, original code used raw dist. 
                            # Z-Scored dist > 0 still implies > Average. 
                            # Let's use the Raw Sign if possible? 
                            # We Z-Scored everything. 
                            # If dist > 0, strictly it means "More above EMA than average".
                            # It's a decent proxy for Trend.
                            
                            signal = position
                            
                            # Entry
                            if can_enter and prob > entry_thresh:
                                if dist > 0: signal = 1
                                else: signal = -1
                            
                            # Exit
                            elif prob < exit_thresh:
                                signal = 0
                                
                            # Execution
                            if position != signal:
                                # Trade occurred
                                if position != 0:
                                    # Closing
                                    pnl = (price - entry_price) * position if position == 1 else (entry_price - price)
                                    trades.append(pnl)
                                
                                if signal != 0:
                                    # Opening
                                    entry_price = price
                                position = signal

                        # METRICS
                        n_trades = len(trades)
                        if n_trades == 0: continue
                        
                        total_pnl = sum(trades) * 10000 # Pips
                        win_rate = sum([1 for x in trades if x > 0]) / n_trades
                        avg_pnl = total_pnl / n_trades
                        
                        res = {
                            "Pair": pair,
                            "Entry": entry_thresh,
                            "Exit": exit_thresh,
                            "Regime": regime_filter,
                            "Trades": n_trades,
                            "WinRate": win_rate,
                            "TotalPnL": total_pnl,
                            "AvgPnL": avg_pnl
                        }
                        results.append(res)

        # FINAL REPORT
        if not results:
            print("No results generated.")
            return

        df_res = pd.DataFrame(results)
        df_res.sort_values("TotalPnL", ascending=False, inplace=True)
        
        print("\n\n====== OPTIMIZATION RESULTS (Top 20) ======")
        print(df_res.head(20).to_string(index=False))
        
        # Aggregated Best Config
        print("\n=== Best Overall Config (Avg PnL across Pairs) ===")
        # Group by params
        df_agg = df_res.groupby(["Entry", "Exit", "Regime"]).agg({
            "TotalPnL": "sum",
            "Trades": "sum",
            "WinRate": "mean" 
        }).reset_index()
        df_agg.sort_values("TotalPnL", ascending=False, inplace=True)
        print(df_agg.head(5).to_string(index=False))

if __name__ == "__main__":
    ParameterSweep().run_sweep()
