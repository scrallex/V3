
import os
import sys
import json
import time
import logging
import asyncio
import tempfile
import subprocess
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta, timezone

# Add path for OandaConnector
sys.path.append("/sep")
sys.path.append("/app")
from scripts.trading.oanda import OandaConnector

# Configuration
PAIRS = ["EUR_USD"]  # Focus on EUR_USD first
LOOKBACK_DAYS = 14
# MODEL_DIR = "/sep/models" # mapped locally?
# BIN_PATH = "/sep/bin/manifold_generator"
MODEL_DIR = "/app/models" # Container path
BIN_PATH = "/app/bin/manifold_generator"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
logger = logging.getLogger("validator")

class StrategyValidator:
    def __init__(self):
        self.connector = OandaConnector(read_only=True)
        self.models = {}
        self.meta = {}
        self.output_dir = "/app/logs" # Mapped to /sep/logs/backend
        os.makedirs(self.output_dir, exist_ok=True)

    def load_artifacts(self):
        logger.info("Loading Models & Metadata...")
        try:
            with open(f"{MODEL_DIR}/feature_builder_meta.json", "r") as f:
                self.meta = json.load(f)
            
            self.model_low = xgb.XGBClassifier()
            self.model_low.load_model(f"{MODEL_DIR}/model_low_vol.json")
            
            self.model_high = xgb.XGBClassifier()
            self.model_high.load_model(f"{MODEL_DIR}/model_high_vol.json")
            
            logger.info("Models loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            sys.exit(1)

    def fetch_data(self, instrument):
        logger.info(f"Fetching {LOOKBACK_DAYS} days of S5 data for {instrument}...")
        
        # OANDA V20 Limit
        to_time = datetime.now(timezone.utc)
        from_time = to_time - timedelta(days=LOOKBACK_DAYS)
        
        all_candles = []
        current_from = from_time
        
        while current_from < to_time:
            # logger.info(f"Fetching chunk from {current_from}...")
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
                    "timestamp": int(dt.timestamp()), # For binary
                    "datetime": dt,
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": c["volume"]
                })
            
            # Next chunk
            last_dt = datetime.fromisoformat(candles[-1]["time"].replace("Z", "+00:00"))
            if last_dt >= to_time or len(candles) < 10:
                break
            current_from = last_dt + timedelta(seconds=5)
            # time.sleep(0.1) # Respect rate limits
            
        logger.info(f"Fetched {len(all_candles)} S5 candles.")
        return pd.DataFrame(all_candles)

    def resample_to_m1(self, df_s5):
        logger.info("Resampling S5 to M1...")
        df_s5.set_index("datetime", inplace=True)
        
        df_m1 = df_s5.resample("1min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "timestamp": "last", # Use close timestamp
            "timestamp_ns": "last"
        }).dropna()
        
        df_m1.reset_index(inplace=True)
        # Ensure int types
        df_m1["timestamp"] = df_m1["timestamp"].astype(int)
        df_m1["timestamp_ns"] = df_m1["timestamp_ns"].astype(int)
        
        logger.info(f"Resampled to {len(df_m1)} M1 candles.")
        return df_m1

    def run_manifold(self, df):
        logger.info("Running Manifold Generator...")
        
        # Prepare Input JSON
        # Proven Format (V3.1 from diagnosis): Timestamp (seconds) + OHLC
        input_data = []
        for _, row in df.iterrows():
            input_data.append({
                "timestamp": int(row["timestamp"]), # Seconds
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
            cmd = [BIN_PATH, "--input", input_path, "--output", output_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            with open(output_path, "r") as f_out:
                res = json.load(f_out)
                
            signals = res.get("signals", [])
            logger.info(f"Manifold generated {len(signals)} signals.")
            if signals:
                logger.info(f"Sample Signal [0]: {json.dumps(signals[0])}")
                logger.info(f"Sample Signal [1]: {json.dumps(signals[1])}")
                logger.info(f"Sample Signal [2]: {json.dumps(signals[2])}")
            return signals
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Manifold failed: {e.stderr.decode()}")
            return []
        except Exception as e:
            logger.error(f"Manifold exception: {e}")
            return []
        finally:
            if os.path.exists(input_path): os.unlink(input_path)
            if os.path.exists(output_path): os.unlink(output_path)

    def compute_features(self, df_price, sfi_signals):
        logger.info("Computing Features...")
        
        # Convert signals to DF
        sfi_data = []
        for s in sfi_signals:
            m = s.get("metrics", {})
            coeffs = s.get("coeffs", {})
            # Use timestamp from signal (seconds) - Binary outputs 'timestamp_ns' only if input had it?
            # Diagnosis showed 'timestamp_ns' in output even with V3.1 input?
            # Diagnosis output: "timestamp_ns": 1769000005000000
            # Wait, binary output format MIGHT change based on input?
            # Diagnosis output showed: "timestamp_ns": 1769000005000000 (Microseconds? 16 digits)
            # Base was 1769000000 * 1e9 = 1769000000000000000 (19 digits).
            # The diagnosis output 1769000005000000 is 1.769e15 -> Microseconds.
            # My validate code expects nanoseconds (19 digits) int(s.get("timestamp_ns", 0))
            
            # Let's robustly handle timestamp.
            ts_val = s.get("timestamp_ns", 0)
            if ts_val == 0:
                ts_val = s.get("timestamp", 0) * 1_000_000_000 # If seconds
            elif ts_val < 10_000_000_000: # Seconds
                ts_val = ts_val * 1_000_000_000
            elif ts_val < 10_000_000_000_000_000: # Microseconds (16 digits)
                ts_val = ts_val * 1_000
                
            ts_ns = int(ts_val)
            
            row = {
                "timestamp_ns": ts_ns,
                "structure_coherence": m.get("coherence", 0),
                "structure_stability": m.get("stability", 0),
                "structure_entropy": m.get("entropy", 0),
                "structure_hazard": coeffs.get("lambda_hazard", 0),
                "structure_rupture": m.get("rupture", 0),
                "structure_coherence_tau_1": m.get("coherence_tau_1", 0),
                "structure_coherence_tau_4": m.get("coherence_tau_4", 0),
                "structure_coherence_tau_slope": m.get("coherence_tau_slope", 0),
                "structure_domain_wall_ratio": m.get("domain_wall_ratio", 0),
                "structure_domain_wall_slope": m.get("domain_wall_slope", 0),
                "structure_spectral_lowf_share": m.get("spectral_lowf_share", 0),
                "structure_reynolds_ratio": m.get("reynolds_ratio", 0),
                "structure_temporal_half_life": m.get("temporal_half_life", 0),
                "structure_spatial_corr_length": m.get("spatial_corr_length", 0),
                "structure_pinned_alignment": m.get("pinned_alignment", 1),
            }
            sfi_data.append(row)
            
        df_sfi = pd.DataFrame(sfi_data)
        
        # Merge
        df = pd.merge_asof(
            df_price.sort_values("timestamp_ns"),
            df_sfi.sort_values("timestamp_ns"),
            on="timestamp_ns",
            direction="backward"
        )
        
        # Feature Engineering (Matching regime_agent.py V3.1 Adaptive)
        # 1. Coherence Delta 10m
        df["coherence_delta_10m"] = df["structure_coherence"] - df["structure_coherence"].shift(10)
        
        # 2. Stability vs 30m Avg
        stab_mean_30 = df["structure_stability"].rolling(window=30, min_periods=1).mean()
        df["stability_vs_30m_avg"] = df["structure_stability"] - stab_mean_30
        
        # 3. Hazard ZScore 60m
        haz_mean_60 = df["structure_hazard"].rolling(window=60, min_periods=1).mean()
        haz_std_60 = df["structure_hazard"].rolling(window=60, min_periods=1).std().replace(0, 1)
        df["hazard_zscore_60m"] = (df["structure_hazard"] - haz_mean_60) / haz_std_60

        # 4. Regime Relative - Using Dynamic Rolling Mean (Adaptive)
        # Instead of static stats, we subtract the rolling mean (proxy for current regime mean)
        for col in ["structure_coherence", "structure_stability", "structure_hazard"]:
             base_col = col.replace("structure_", "")
             target = f"{base_col}_minus_regime_mean"
             # Use 60m rolling mean as regime proxy
             rolling_mean = df[col].rolling(window=60, min_periods=1).mean()
             df[target] = df[col] - rolling_mean

        df["interaction_high_rupture_pos_domain"] = 0.0
        df["interaction_low_hazard_neg_reynolds"] = 0.0
        
        return df.dropna()

    def run_inference(self, df, instrument):
        logger.info("Running Inference (Adaptive)...")
        
        # Prepare Feature Vector
        features = self.meta["numeric_features"]
        
        # Adaptive Normalization: Rolling Window=60
        # We calculate z-score for ALL numeric features based on trailing 60 candles
        
        X = pd.DataFrame(index=df.index)
        
        for f in features:
            if f == "semantic_tag_count":
                X[f] = 0.0
                continue
            
            if f not in df.columns:
                X[f] = 0.0
                continue
                
            # Dynamic Z-Score
            roll_mean = df[f].rolling(window=60, min_periods=1).mean()
            roll_std = df[f].rolling(window=60, min_periods=1).std()
            
            # Avoid /0 or NaN
            roll_std = roll_std.replace(0, 1.0).fillna(1.0)
            
            X[f] = (df[f] - roll_mean) / roll_std
            
        # Categoricals
        # inst_map = self.meta["category_mappings"].get("instrument", {})
        # X["instrument_code"] = int(inst_map.get(instrument, 0))
        # X["regime_code"] = 0 
        
        # User update used 0.0 for codes
        X["instrument_code"] = 0.0
        X["regime_code"] = 0.0
        
        # Reorder columns matches training
        cols = features + ["instrument_code", "regime_code"]
        cols = [c for c in cols if c in X.columns]
        X = X[cols]
        
        # Predictions
        # Branch on Normalized Hazard
        # We need the Normalized Hazard specifically
        
        # Recalculate 'structure_hazard' normalized 
        h_col = "structure_hazard"
        h_mean = df[h_col].rolling(window=60, min_periods=1).mean()
        h_std = df[h_col].rolling(window=60, min_periods=1).std().replace(0, 1.0).fillna(1.0)
        norm_hazard = (df[h_col] - h_mean) / h_std
        
        mask_high = norm_hazard >= 0.0
        mask_low = ~mask_high
        
        # Initialize
        df["prob"] = 0.0
        df["regime"] = "LowVol"
        df.loc[mask_high, "regime"] = "HighVol"
        
        # Predict High Vol
        if mask_high.any():
            p_high = self.model_high.predict_proba(X.loc[mask_high])[:, 1]
            df.loc[mask_high, "prob"] = p_high
            
        # Predict Low Vol
        if mask_low.any():
            p_low = self.model_low.predict_proba(X.loc[mask_low])[:, 1]
            df.loc[mask_low, "prob"] = p_low
            
        return df

    def simulate(self, df):
        logger.info("Simulating Trades...")
        
        # Logic: 
        # Long if Prob > 0.60
        # Short if Prob < 0.40
        # Neutral otherwise.
        
        # Flatten positions? Or hold?
        # "Trading every minute based on Regime-specific model confidence"
        # Implies we take a position for that minute.
        
        df["signal"] = 0
        df.loc[df["prob"] > 0.60, "signal"] = 1
        df.loc[df["prob"] < 0.40, "signal"] = -1
        
        # Returns
        # Return = Signal * (NextOpen - Open) / Open ? 
        # Or Signal * (NextClose - Close) / Close ? 
        # User backtest: "Trading every minute".
        # Assume entry at Close (or Open of next?), exit at next Close.
        # Vectorized standard: Strategy returns = Shifted Signal * Market Returns.
        # If we signal at row i, we take position for candle i+1.
        
        df["market_return"] = df["close"].pct_change().shift(-1) # Return of next period
        df["strategy_return"] = df["signal"] * df["market_return"]
        
        # Costs (1.5bps per side = 3bps round trip if we flip)
        # Cost is incurred when position changes.
        df["pos_change"] = df["signal"].diff().abs()
        # 1.5bps = 0.00015
        # If pos change 1 (0->1), cost 1.5bps.
        # If pos change 2 (-1->1), cost 3.0bps (close short, open long).
        cost_per_turn = 0.00015
        df["costs"] = df["pos_change"] * cost_per_turn
        
        df["costs"] = df["pos_change"] * cost_per_turn
        
        df["net_return"] = df["strategy_return"] - df["costs"]
        df["cum_return"] = df["net_return"].cumsum()
        
        return df
        # 1440 mins/day. 
        # Total PnL (Log units/Percent)
        total_pnl = df_sim["net_return"].sum()
        
        logger.info("="*40)
        logger.info(f"RESULTS: {pair}")
        logger.info(f"Period: {LOOKBACK_DAYS} days")
        logger.info(f"Candles (M1): {len(df_sim)}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Net PnL (%): {total_pnl*100:.2f}%")
        logger.info("="*40)
        
        # Save CSV
        csv_path = f"{self.output_dir}/{pair}_validation.csv"
        df_sim.to_csv(csv_path)
        logger.info(f"Detailed results saved to {csv_path}")

    def resample_signals_to_m1(self, df_s5, signals):
        """
        Merge S5 candles with Signals, then resample to M1 (taking last).
        """
        # Convert signals to DataFrame
        import pandas as pd
        sig_data = []
        for s in signals:
            try:
                ts_val = s.get("timestamp_ns", 0)
                if ts_val == 0:
                     ts_val = s.get("timestamp", 0) * 1_000_000_000
                elif ts_val < 10_000_000_000:
                     ts_val = ts_val * 1_000_000_000
                elif ts_val < 10_000_000_000_000_000:
                     ts_val = ts_val * 1_000
                     
                mets = s.get("metrics", {})
                coeffs = s.get("coeffs", {})
                
                row = {
                    "timestamp_ns": int(ts_val),
                    # Key metrics
                    "coherence": float(mets.get("coherence", 0)),
                    "entropy": float(mets.get("entropy", 0)),
                    "field_strength": float(mets.get("field_strength", 0) or 0), # Some binaries use different keys
                    "stability": float(mets.get("stability", 0)),
                    "lambda_hazard": float(coeffs.get("lambda_hazard", 0)),
                }
                sig_data.append(row)
            except Exception:
                continue
                
        if not sig_data:
            return pd.DataFrame(), [] # Return empty list for signals
            
        df_sig = pd.DataFrame(sig_data)
        # Merge on timestamp_ns (S5)
        # Assuming df_s5 has 'timestamp_ns'
        
        # Ensure Types
        df_sig["timestamp_ns"] = df_sig["timestamp_ns"].astype("int64")
        df_s5["timestamp_ns"] = df_s5["timestamp_ns"].astype("int64")
        
        # Merge (left on candle)
        df_merged = pd.merge_asof(
            df_s5.sort_values("timestamp_ns"),
            df_sig.sort_values("timestamp_ns"),
            on="timestamp_ns",
            direction="nearest",
            tolerance=5000000000 # 5s tolerance
        )
        
        # Set Index to Datetime
        df_merged["dt"] = pd.to_datetime(df_merged["timestamp_ns"], unit="ns")
        df_merged.set_index("dt", inplace=True)
        
        # Resample to 1 Min
        # rules: OHLC aggregation, metrics take LAST
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "timestamp_ns": "last",
            "timestamp": "last",
            # Metrics
            "coherence": "last",
            "entropy": "last",
            "stability": "last",
            "lambda_hazard": "last"
        }
        
        df_m1 = df_merged.resample("1min").agg(agg_dict).dropna()
        df_m1["timestamp_ns"] = df_m1["timestamp_ns"].astype(int)
        df_m1["timestamp"] = df_m1["timestamp"].astype(int)
        
        # Reconstruct 'signals' list for feature_builder (which expects dicts)
        # Or better: modify compute_features to take df_metrics?
        # Existing compute_features takes (df, signals_list).
        # We can reconstruct the signals list from the M1 df.
        
        m1_signals = []
        for _, row in df_m1.iterrows():
            m1_signals.append({
                "timestamp_ns": int(row["timestamp_ns"]),
                "metrics": {
                    "coherence": row["coherence"],
                    "entropy": row["entropy"],
                    "stability": row["stability"]
                },
                "coeffs": {
                    "lambda_hazard": row["lambda_hazard"]
                }
            })
            
        return df_m1.reset_index(drop=True), m1_signals

    def run_pipeline_for_pair(self, pair):
        try:
            # 1. Fetch S5
            # Use fetch_data (existing name)
            df_s5 = self.fetch_data(pair) 
            if df_s5.empty:
                logger.error(f"No data for {pair}")
                return
                
            # 2. Run Manifold on S5 (High Fidelity)
            logger.info("Running Manifold on S5 Data (High Fidelity)...")
            s5_signals = self.run_manifold(df_s5)
            if not s5_signals:
                logger.error("No manifold signals")
                return

            # 3. Resample to M1 (for Model Compatibility)
            logger.info("Resampling S5 Metrics to M1...")
            df_m1, m1_signals = self.resample_signals_to_m1(df_s5, s5_signals)
            if df_m1.empty:
                logger.error(f"No M1 data after resampling for {pair}")
                return
            logger.info(f"Resampled to {len(df_m1)} M1 candles.")
            
            # 4. Features (on M1)
            logger.info("Computing Features (M1)...")
            df_feats = self.compute_features(df_m1, m1_signals)
            
            # 5. Inference
            df_res = self.run_inference(df_feats, pair)
            
            # 6. Simulate
            df_sim = self.simulate(df_res)
            
            # 7. Stats
            if df_sim is None or df_sim.empty:
                logger.error("Simulation returned empty results")
                return

            total_trades = (df_sim["signal"].diff().abs() > 0).sum()
            total_pnl = df_sim["net_return"].sum()
            avg_return = df_sim["net_return"].mean()
            std_return = df_sim["net_return"].std()
            sharpe = (avg_return / std_return) * np.sqrt(1440*365) if std_return > 0 else 0
            
            logger.info("="*40)
            logger.info(f"RESULTS: {pair}")
            logger.info(f"Period: {LOOKBACK_DAYS} days")
            logger.info(f"Candles (M1): {len(df_sim)}")
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"Sharpe Ratio: {sharpe:.2f}")
            logger.info(f"Net PnL (%): {total_pnl*100:.2f}%")
            logger.info("="*40)
            
            # Save CSV
            csv_path = f"{self.output_dir}/{pair}_validation.csv"
            df_sim.to_csv(csv_path)
            logger.info(f"Detailed results saved to {csv_path}")

        except Exception as e:
            logger.exception(f"Pipeline failed for {pair}")

    def run(self):
        print("DEBUG: Starting run()")
        if not self.load_artifacts():
            print("DEBUG: Artifacts failed to load")
            return
            
        print(f"DEBUG: PAIRS: {PAIRS}")
        for pair in PAIRS:
            print(f"DEBUG: Processing {pair}")
            self.run_pipeline_for_pair(pair)

if __name__ == "__main__":
    validator = StrategyValidator()
    validator.run()
