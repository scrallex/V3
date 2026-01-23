#!/usr/bin/env python3
"""
01_data_prep.py
Purpose: Fetch S5 Data, Run Manifold Generator, Save Rich Dataset.
Phase 4: Multi-Instrument Support (Remaining 4 Majors).
"""
import json
import logging
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("DataPrep")

# Config
DATA_DIR = Path("/sep/research_lab/data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

BIN_PATH = "/sep/bin/manifold_generator"
FETCH_SCRIPT = "/sep/scripts/research/fetch_oanda_history.py"


def fetch_data(instrument, output_path):
    if output_path.exists():
        logger.info(f"Using existing raw data: {output_path}")
        return

    logger.info(f"Fetching S5 Data for {instrument}...")

    if not Path(FETCH_SCRIPT).exists():
        logger.error(f"Fetch script not found at {FETCH_SCRIPT}")
        raise FileNotFoundError(FETCH_SCRIPT)

    # Calculate Last 14 Days
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=14)
    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    cmd = [
        "python3",
        FETCH_SCRIPT,
        "--instrument",
        instrument,
        "--granularity",
        "S5",
        "--start",
        start_str,
        "--end",
        end_str,
        "--output",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def run_manifold(df):
    """Run the C++ binary on the DataFrame."""
    logger.info(f"Running Manifold Generator on {len(df)} candles...")

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "timestamp": int(row["timestamp_ns"]) // 1_000_000,  # MS
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_in:
        json.dump(records, f_in)
        src_path = f_in.name
    dst_path = src_path + ".out"

    try:
        result = subprocess.run(
            [BIN_PATH, "--input", src_path, "--output", dst_path],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"Binary Failed: {result.stderr}")
            raise Exception("Binary execution failed")

        with open(dst_path, "r") as f_out:
            data = json.load(f_out)

        signals = data.get("signals", [])
        if not signals:
            logger.error(f"Binary returned ZERO signals. Stderr: {result.stderr}")

        return signals

    finally:
        Path(src_path).unlink(missing_ok=True)
        Path(dst_path).unlink(missing_ok=True)


def merge_and_save(df, sfi_data, output_path):
    """Align SFI metrics with Candle Data."""
    logger.info("Merging Metrics...")

    sfi_records = []
    for item in sfi_data:
        rec = {}
        ts = item.get("timestamp_ns", item.get("timestamp"))
        rec["timestamp_ns"] = int(ts)

        rec.update(item.get("metrics", {}))
        rec.update(item.get("coeffs", {}))
        sfi_records.append(rec)

    df_sfi = pd.DataFrame(sfi_records)

    df["timestamp_ns"] = df["timestamp_ns"].astype("int64")
    df_sfi["timestamp_ns"] = df_sfi["timestamp_ns"].astype("int64")

    merged = pd.merge(df, df_sfi, on="timestamp_ns", how="inner")

    logger.info(f"Saving Rich Dataset to {output_path} ({len(merged)} rows)...")
    merged.to_parquet(output_path)
    logger.info("Done.")


def process_instrument(instrument):
    logger.info(f"=== Processing {instrument} ===")

    RAW_CSV = DATA_DIR / f"{instrument}_S5_raw.csv"
    OUTPUT_PARQUET = DATA_DIR / f"{instrument}_rich.parquet"

    # 1. Fetch
    try:
        fetch_data(instrument, RAW_CSV)
    except subprocess.CalledProcessError as e:
        logger.error(f"Fetch failed for {instrument}: {e}")
        return

    # 2. Load
    df = pd.read_csv(RAW_CSV)

    if "timestamp_ns" not in df.columns:
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df["timestamp_ns"] = df["time"].astype("int64")
        else:
            logger.error("No 'time' column found!")
            return

    # 3. Manifold
    signals = run_manifold(df)
    if not signals:
        logger.error(f"Manifold failed for {instrument}")
        return

    # 4. Merge
    merge_and_save(df, signals, OUTPUT_PARQUET)


def main():
    # The remaining 4 majors
    instruments = ["USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]

    for inst in instruments:
        try:
            process_instrument(inst)
        except Exception as e:
            logger.error(f"Failed to process {inst}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
