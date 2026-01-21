#!/usr/bin/env python3
"""
Simulate Stream Disconnect (HFD Analysis)
Quantifies the difference between trading on "Live S5 Updates" vs "Confirmed M1 Closes".
"""
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Config
BIN_PATH = "/sep/V3/bin/manifold_generator"
MODEL_DIR = "/sep/V3/models"
INPUT_S5 = "/sep/data/hfd_analysis/EUR_USD_S5.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("HFD")


class HFDAnalyzer:
    def __init__(self):
        self.load_artifacts()

    def load_artifacts(self):
        with open(Path(MODEL_DIR) / "feature_builder_meta.json", "r") as f:
            self.meta = json.load(f)
        self.norm_stats = self.meta["normalization_stats"]
        self.features = self.meta["numeric_features"]

        self.model_low = xgb.XGBClassifier()
        self.model_low.load_model(Path(MODEL_DIR) / "model_low_vol.json")
        self.model_high = xgb.XGBClassifier()
        self.model_high.load_model(Path(MODEL_DIR) / "model_high_vol.json")

    def run_sfi(self, candles):
        # candles: list of dicts {"timestamp_ns":..., "price":...}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f_in:
            json.dump(candles, f_in)
            src_path = f_in.name
        dst_path = src_path + ".out"

        try:
            subprocess.run(
                [BIN_PATH, "--input", src_path, "--output", dst_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            with open(dst_path, "r") as f_out:
                data = json.load(f_out)
            return data.get("signals", [])
        except subprocess.CalledProcessError:
            with open(src_path, "r") as f_dbg:
                logger.error(f"Binary Crashed! Input Snippet: {f_dbg.read()[:500]}...")
            raise
        finally:
            Path(src_path).unlink(missing_ok=True)
            Path(dst_path).unlink(missing_ok=True)

    def normalize_and_predict(self, df_row):
        X = []
        for f in self.features:
            if f == "semantic_tag_count":
                X.append(0.0)
                continue
            val = df_row.get(f, 0.0)
            stats = self.norm_stats.get(f)
            if stats:
                val = (val - stats["mean"]) / stats["std"]
            X.append(val)

        # Add Dummies
        cols = self.features + ["instrument_code", "regime_code"]
        row_vals = X + [0, 0]  # Default codes
        df_X = pd.DataFrame([row_vals], columns=cols)

        # Hazard
        norm_haz = (
            df_row["structure_hazard"] - self.norm_stats["structure_hazard"]["mean"]
        ) / self.norm_stats["structure_hazard"]["std"]

        if norm_haz >= 0.0:
            prob = self.model_high.predict_proba(df_X)[:, 1][0]
        else:
            prob = self.model_low.predict_proba(df_X)[:, 1][0]

        if prob > 0.6:
            return "LONG"
        if prob < 0.4:
            return "SHORT"
        return "NEUTRAL"

    def analyze(self):
        logger.info("Loading S5 Data...")
        df_s5 = pd.read_csv(INPUT_S5)
        df_s5["time"] = pd.to_datetime(df_s5["time"])
        df_s5 = df_s5.sort_values("time")

        logger.info(f"Loaded {len(df_s5)} S5 candles.")

        # State
        m1_history = []  # Confirmed M1 candles
        current_m1 = None  # Forming

        results = []

        # Warmup: We need 60 M1 candles before we can predict.
        # We will fast-forward M1 construction.

        # Iterate S5
        for idx, row in df_s5.iterrows():
            ts = row["time"]
            price = float(row["close"])  # S5 Close
            m1_ts_start = ts.floor("1min")

            # New Minute?
            if current_m1 and current_m1["ts"] != m1_ts_start:
                # Close previous
                m1_history.append(current_m1)
                if len(m1_history) > 64:
                    m1_history.pop(0)
                current_m1 = None

            if current_m1 is None:
                current_m1 = {
                    "ts": m1_ts_start,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": price,
                    "count": 1,
                }
            else:
                # Update Streaming
                current_m1["high"] = max(current_m1["high"], float(row["high"]))
                current_m1["low"] = min(current_m1["low"], float(row["low"]))
                current_m1["close"] = price
                current_m1["count"] += 1

            if len(m1_history) < 60:
                continue

            # Construct Window for Inference
            # Window = History[-59:] + Current_Forming
            window = m1_history[-60:]  # Actually use History if confirmed?
            # Ideally: History[-59:] + [Current_Forming]
            inference_window = m1_history[-min(len(m1_history), 59) :] + [current_m1]

            # Prepare for SFI
            # SFI expects {"timestamp_ns", "price"} (using closes)
            inputs = []
            for c in inference_window:
                inputs.append({"timestamp_ns": int(c["ts"].value), "price": c["close"]})

            # Run SFI
            sfi_out = self.run_sfi(inputs)
            if not sfi_out:
                continue

            # Get Last Metrics
            last_sfi = sfi_out[-1]
            metrics = last_sfi.get("metrics", {})
            coeffs = last_sfi.get("coeffs", {})

            # Simple Feature Construction (Zero Lag for simplicity of check)
            # We ignore rolling features for this quick check, or approximate?
            # Creating rolling features requires history.
            # We will approximate by just checking the RAW SFI stability/hazard.

            # To do this right, we'd need the full feature builder.
            # BUT, the disconnect is primarily about INPUT DATA STABILITY.
            # Does the SFI output fluctuate wildly on forming candles?

            res = {
                "ts_s5": ts,
                "ts_m1": m1_ts_start,
                "s5_price": price,
                "hazard": coeffs.get("lambda_hazard", 0),
                "stability": metrics.get("stability", 0),
                "is_new_minute": (current_m1["count"] == 1),
            }
            results.append(res)

            if len(results) >= 2000:
                break

            if len(results) % 100 == 0:
                logger.info(f"Processed {len(results)} S5 steps...")

        # Analysis
        df_res = pd.DataFrame(results)

        # Calculate Volatility of Metric within the Minute
        # Group by M1 TS
        grouped = df_res.groupby("ts_m1")

        metric_vol = grouped["stability"].std().mean()
        metric_range_avg = (
            grouped["stability"].max() - grouped["stability"].min()
        ).mean()

        logger.info("------------------------------------------------")
        logger.info("HFD ANALYSIS RESULTS (EUR_USD 2 Weeks)")
        logger.info("------------------------------------------------")
        logger.info(f"Avg Intra-Minute Stability StdDev: {metric_vol:.5f}")
        logger.info(f"Avg Intra-Minute Stability Range:  {metric_range_avg:.5f}")

        # How often does it cross the regime threshold (0.25)?
        # Count flips within minute
        flips = 0
        total_mins = 0
        for name, group in grouped:
            total_mins += 1
            # Check if crossing 0.25
            above = group["stability"] > 0.25
            if above.nunique() > 1:  # Both True and False exist
                flips += 1

        logger.info(
            f"Regime Flickers (Stability > 0.25): {flips} / {total_mins} minutes ({flips/total_mins*100:.2f}%)"
        )
        logger.info("------------------------------------------------")

        # Recommendations
        if flips / total_mins > 0.05:
            logger.info(
                "VERDICT: SIGNIFICANT DISCONNECT. Streaming forming candles causes regime flicker."
            )
            logger.info("RECOMMENDATION: Enforce M1-Close Execution.")
        else:
            logger.info("VERDICT: LOW NOISE. Streaming execution is safe.")


if __name__ == "__main__":
    HFDAnalyzer().analyze()
