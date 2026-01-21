import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone

# Add parent dir to path to find lib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.manifold import ManifoldRunner

# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
logger = logging.getLogger("DataPrep")

# Path to OANDA Connector (reusing backend)
sys.path.append("/app/scripts")
from trading.oanda import OandaConnector

def fetch_s5_data(instrument="EUR_USD", days=7):
    logger.info(f"Fetching {days} days of S5 data for {instrument}...")
    try:
        connector = OandaConnector()
    except Exception as e:
        logger.error("Could not init OandaConnector. Are credentials set?")
        raise e

    to_time = datetime.now(timezone.utc)
    from_time = to_time - timedelta(days=days)
    
    all_candles = []
    current_from = from_time
    
    # Simple chunking loop
    while current_from < to_time:
        candles = connector.get_candles(
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
            ts_sec = int(dt.timestamp())
            ts_ns = int(ts_sec * 1_000_000_000)
            
            mid = c["mid"]
            all_candles.append({
                "timestamp": ts_sec,
                "timestamp_ns": ts_ns,
                "datetime": dt,
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c["volume"])
            })
            
        last_dt = datetime.fromisoformat(candles[-1]["time"].replace("Z", "+00:00"))
        if last_dt >= to_time or len(candles) < 10:
            break
        current_from = last_dt + timedelta(seconds=5)

    df = pd.DataFrame(all_candles)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    logger.info(f"Fetched {len(df)} candles.")
    return df

def merge_manifold_metrics(df_s5, signals):
    """
    Parses signal list and merges metrics back to DF.
    """
    logger.info("Merging metrics...")
    sig_data = []
    for s in signals:
        # Robust Timestamp Parsing (Handle Secs, MS, US, NS)
        ts_val = s.get("timestamp", 0)
        if ts_val == 0:
            ts_raw = s.get("timestamp_ns", 0)
            if ts_raw > 1e18: # Nanoseconds (19 digits)
                ts_val = int(ts_raw / 1_000_000_000)
            elif ts_raw > 1e15: # Microseconds (16 digits)
                ts_val = int(ts_raw / 1_000_000)
            elif ts_raw > 1e12: # Milliseconds (13 digits)
                ts_val = int(ts_raw / 1_000)
            else:
                ts_val = int(ts_raw)
        
        ts_val = int(ts_val)

        mets = s.get("metrics", {})
        coeffs = s.get("coeffs", {})
        
        sig_data.append({
            "timestamp": ts_val,
            "coherence": float(mets.get("coherence", 0)),
            "entropy": float(mets.get("entropy", 0)),
            "stability": float(mets.get("stability", 0)),
            "lambda_hazard": float(coeffs.get("lambda_hazard", 0))
        })
        
    df_sig = pd.DataFrame(sig_data)
    
    # Merge
    # Ensure join key types match
    df_s5["timestamp"] = df_s5["timestamp"].astype(int)
    
    df_rich = pd.merge(df_s5, df_sig, on="timestamp", how="left")
    # Forward fill metrics (S5 hold)
    df_rich.ffill(inplace=True)
    return df_rich

if __name__ == "__main__":
    PAIR = "EUR_USD"
    DAYS = 14 # Production Default
    
    # Path relative to script
    # SCRIPT_DIR is RO. Use logs.
    OUTPUT_FILE = f"/app/logs/s5_rich_data_{PAIR}.csv"
    
    # 1. Fetch
    df = fetch_s5_data(PAIR, DAYS)
    
    # 2. Manifold
    runner = ManifoldRunner()
    signals = runner.run(df)
    
    # 3. Merge
    df_rich = merge_manifold_metrics(df, signals)
    
    # 4. Save
    df_rich.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Success! Saved rich dataset to {OUTPUT_FILE}")
