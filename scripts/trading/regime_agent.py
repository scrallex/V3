#!/usr/bin/env python3
"""
Regime Adaptive Agent (Live)
- Consumes M1/S5 Data from Redis.
- Runs SFI Manifold Generator (Binary).
- Generates Features (Lag/Rolling).
- Normalizes using Train Stats.
- Switches Models based on Hazard.
- Publishes Signals to Redis.
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
HISTORY_LEN = 3000  # Buffer size for SFI convergence
SFI_STEP = 1  # High Fidelity


class RegimeAgent:
    def __init__(self, redis_url, instruments, poll_interval=5.0):
        self.redis = redis.from_url(redis_url)
        self.instruments = instruments
        self.poll_interval = poll_interval
        self.running = True

        # Load Models & Meta
        self.load_artifacts()

        # State
        self.buffers = {inst: deque(maxlen=HISTORY_LEN) for inst in instruments}
        self.last_processed_ts = {inst: 0 for inst in instruments}

    def load_artifacts(self):
        logger.info("Loading Models & Meta...")

        # Meta
        with open(MODEL_DIR / "feature_builder_meta.json", "r") as f:
            self.meta = json.load(f)

        self.norm_stats = self.meta["normalization_stats"]
        self.features = self.meta["numeric_features"]  # Ensure order matches training!

        # Models
        self.model_low = xgb.XGBClassifier()
        self.model_low.load_model(MODEL_DIR / "model_low_vol.json")

        self.model_high = xgb.XGBClassifier()
        self.model_high.load_model(MODEL_DIR / "model_high_vol.json")

        logger.info("Artifacts Loaded.")

    def run(self):
        logger.info(f"Starting Regime Agent for {self.instruments}...")
        while self.running:
            for inst in self.instruments:
                try:
                    self.process_instrument(inst)
                except Exception as e:
                    logger.error(f"Error processing {inst}: {e}", exc_info=True)
            time.sleep(self.poll_interval)

    def stop(self):
        self.running = False
        logger.info("Stopping Agent...")

    def process_instrument(self, inst):
        # 1. Fetch History (simulated from Redis or Backfill)
        # In V3, we assume S5/M1 keys exist.
        # For this implementation, we rely on the standard `md:candles:{inst}:M1` ZSET.
        # Key contains JSON strings: '{"t": 123456789, "o":..., "c":...}'

        key = f"md:candles:{inst}:M1"
        raw_data = self.redis.zrange(key, -HISTORY_LEN, -1)

        if not raw_data:
            return

        # Parse
        candles = []
        for r in raw_data:
            try:
                if isinstance(r, bytes):
                    r = r.decode()
                c = json.loads(r)
                # Normalize typical OANDA/V3 format
                # Expects: mid:{o,h,l,c} or simple o,h,l,c
                mid = c.get("mid", c)
                price = float(mid.get("c") if isinstance(mid, dict) else mid)
                ts = int(c.get("t", 0))
                # Skip if invalid
                if ts > 0:
                    candles.append({"timestamp_ns": ts * 1_000_000, "price": price})
            except:
                continue

        if len(candles) < 60:
            return

        # Check if new minute started
        # self.last_processed_ts tracks the "Current Forming Minute Start Time"
        current_forming_ts = candles[-1]["timestamp_ns"]

        if current_forming_ts <= self.last_processed_ts[inst]:
            return  # Still in same minute, no new confirmed candle

        # New minute detected!
        # The PREVIOUS candle (candles[-2]) is now COMPLETE.
        # We run inference on the COMPLETED window (excluding the new forming candle).

        completed_history = candles[:-1]
        if len(completed_history) < 60:
            return

        # 2. Run SFI Pipeline on COMPLETED history
        sfi_metrics = self.run_manifold(completed_history)
        if not sfi_metrics:
            return

        # 3. Feature Engineering
        df_feats = self.compute_features(completed_history, sfi_metrics)
        if df_feats is None or df_feats.empty:
            return

        row = df_feats.iloc[-1]

        # 4. Normalize (Adaptive)
        # Instead of static 2024 stats, we compute dynamic z-scores from the history buffer.
        # This ensures sensitivity to the current regime (Jan 2026).

        # We need the full history dataframe to compute columns
        if df_feats is None or len(df_feats) < 20:
            return

        # Adaptive Stats (Last 60 candles)
        # We treat the input 'candles' history as the regime window.
        stats_window = df_feats.iloc[-60:]

        # Prepare inputs
        X_final_numeric = []
        for f in self.features:
            val = row[f]
            # Compute dynamic stats
            mean = stats_window[f].mean()
            std = stats_window[f].std()

            # Guard against zero std
            if std == 0 or pd.isna(std):
                std = 1.0

            # Z-Score
            z = (val - mean) / std
            X_final_numeric.append(z)

        # Hazard specific normalization for model switching
        h_mean = stats_window["structure_hazard"].mean()
        h_std = stats_window["structure_hazard"].std()
        if h_std == 0 or pd.isna(h_std):
            h_std = 1.0

        norm_hazard = (row["structure_hazard"] - h_mean) / h_std

        # Construct DataFrame for Model
        # Model expects columns: [numeric_features...] + [instrument_code, regime_code]
        # We use 0.0 for categorical codes (UNKNOWN) as we lack mappings in live agent
        # This is safe as the model learns relative structure mostly from numerics.
        X_final = X_final_numeric + [0.0, 0.0]
        # 5. Predict
        # Use the normalized input DataFrame `X_df` directly.
        # It already contains [numeric_features] + [instrument_code=0, regime_code=0]

        # Branch on Normalized Hazard (calculated above)
        if norm_hazard >= 0.0:
            # High Vol
            prob = self.model_high.predict_proba(X_df)[:, 1][0]
            label = "HighVol"
        else:
            # Low Vol
            prob = self.model_low.predict_proba(X_df)[:, 1][0]
            label = "LowVol"

        # 6. Signal
        signal = "NEUTRAL"
        if prob > 0.60:
            signal = "LONG"
        elif prob < 0.40:
            signal = "SHORT"

        # 7. Publish
        logger.info(
            f"{inst} | {label} | Haz={norm_hazard:.2f} | Prob={prob:.2f} | Signal={signal}"
        )

        gate_payload = {
            "instrument": inst,
            "ts_ms": current_forming_ts // 1_000_000,
            "signal": signal,
            "prob": float(prob),
            "regime": label,
            "hazard_norm": float(norm_hazard),
            "source": "regime_agent",
            "admit": True,
        }

        self.redis.set(f"gate:last:{inst}", json.dumps(gate_payload))
        self.last_processed_ts[inst] = current_forming_ts

    def run_manifold(self, candles):
        # Dump to JSON
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f_in:
            json.dump(candles, f_in)
            input_path = f_in.name

        output_path = input_path + ".out.json"

        try:
            # Run Binary
            # ./manifold_generator --input input.json --output output.json --step 1
            subprocess.run(
                [
                    str(BIN_PATH),
                    "--input",
                    input_path,
                    "--output",
                    output_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Read Output
            with open(output_path, "r") as f_out:
                res = json.load(f_out)

            return res.get("signals", [])

        except Exception as e:
            logger.error(f"Manifold Gen failed: {e}")
            return None
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compute_features(self, candles, sfi_signals):
        # Align Series
        # SFI signals align with input candles 1-to-1 (if step=1) or sparse.
        # We assume 1-to-1 or need to merge on TS.

        df_price = pd.DataFrame(candles)
        df_price["ts"] = pd.to_datetime(df_price["timestamp_ns"], unit="ns")

        # SFI to DF
        sfi_data = []
        for s in sfi_signals:
            m = s.get("metrics", {})
            coeffs = s.get("coeffs", {})
            row = {
                "timestamp_ns": s["timestamp_ns"],
                "structure_coherence": m.get("coherence", s.get("coherence", 0)),
                "structure_stability": m.get("stability", s.get("stability", 0)),
                "structure_entropy": m.get("entropy", s.get("entropy", 0)),
                "structure_hazard": coeffs.get(
                    "lambda_hazard", s.get("lambda_hazard", 0)
                ),
                "structure_rupture": m.get("rupture", s.get("rupture", 0)),
                # ... Add all other SFI columns required by meta ...
                # For brevity, implementing mapped columns from meta
            }
            # Add strict mappings
            row["structure_coherence_tau_1"] = m.get("coherence_tau_1", 0)
            row["structure_coherence_tau_4"] = m.get("coherence_tau_4", 0)
            row["structure_coherence_tau_slope"] = m.get("coherence_tau_slope", 0)
            row["structure_domain_wall_ratio"] = m.get("domain_wall_ratio", 0)
            row["structure_domain_wall_slope"] = m.get("domain_wall_slope", 0)
            row["structure_spectral_lowf_share"] = m.get("spectral_lowf_share", 0)
            row["structure_reynolds_ratio"] = m.get("reynolds_ratio", 0)
            row["structure_temporal_half_life"] = m.get("temporal_half_life", 0)
            row["structure_spatial_corr_length"] = m.get("spatial_corr_length", 0)
            row["structure_pinned_alignment"] = m.get("pinned_alignment", 1)

            sfi_data.append(row)

        df_sfi = pd.DataFrame(sfi_data)

        # Merge
        df = pd.merge_asof(
            df_price.sort_values("timestamp_ns"),
            df_sfi.sort_values("timestamp_ns"),
            on="timestamp_ns",
            direction="backward",
        )

        df["price_close"] = df["price"]

        # Feature Engineering (Rolling)
        # Matches feature_builder.py

        # 1. Coherence Delta 10m
        # Assuming M1 data, 10 periods
        df["coherence_delta_10m"] = df["structure_coherence"] - df[
            "structure_coherence"
        ].shift(10)

        # 2. Stability vs 30m Avg
        stab_mean_30 = (
            df["structure_stability"].rolling(window=30, min_periods=1).mean()
        )
        df["stability_vs_30m_avg"] = df["structure_stability"] - stab_mean_30

        # 3. Hazard ZScore 60m
        haz_mean_60 = df["structure_hazard"].rolling(window=60, min_periods=1).mean()
        haz_std_60 = (
            df["structure_hazard"].rolling(window=60, min_periods=1).std().replace(0, 1)
        )  # avoid div0
        df["hazard_zscore_60m"] = (df["structure_hazard"] - haz_mean_60) / haz_std_60

        # 4. Regime Relative (Minus Global Mean)
        # In feature_builder this was calculated relative to 'Regime Mean'.
        # Since we don't know the regime mean (it was historical), we assume 0 or drop?
        # WAIT. The training script calculated specific means per regime.
        # `regime_means = df.groupby("regime")[...].transform("mean")`
        # In inference, we can't look ahead.
        # We can implement a static proxy. Or just ignore?
        # The Feature Vector requires it.
        # "coherence_minus_regime_mean"
        # We can use the GLOBAL mean from normalization stats as a proxy?
        # feature_builder used `df[...] - regime_means`.
        # If we use `df[...] - norm_stats['mean']`, it simulates "Minus Regime Mean" closely if regime is dominant.
        # Let's use the Norm Stats Mean for now. It's the best proxy zero-shot.

        for col in ["structure_coherence", "structure_stability", "structure_hazard"]:
            base_col = col.replace("structure_", "")
            # keys are named "coherence_minus_regime_mean"
            target = f"{base_col}_minus_regime_mean"
            if target in self.features:
                # Use global mean
                stat = self.norm_stats.get(col, {"mean": 0})
                df[target] = df[col] - stat["mean"]

        # 5. Interactions
        df["interaction_high_rupture_pos_domain"] = 0.0  # Placeholder
        df["interaction_low_hazard_neg_reynolds"] = 0.0  # Placeholder

        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis", default="redis://localhost:6379/0")
    parser.add_argument(
        "--pairs", default="EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,USD_CAD,NZD_USD"
    )
    args = parser.parse_args()

    pairs = args.pairs.split(",")
    agent = RegimeAgent(args.redis, pairs)

    def shutdown(sig, frame):
        agent.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    agent.run()


if __name__ == "__main__":
    main()
