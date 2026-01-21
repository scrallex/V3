#!/usr/bin/env python3
"""
Candle Streamer Tool
Polls OANDA for recent candles and pushes them to Redis for the Regime Manifold Service.
"""

import argparse
import json
import logging
import os
import sys
import time
import redis
from pathlib import Path

# Ensure root is on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.trading.oanda import OandaConnector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stream-candles")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis", default=os.getenv("VALKEY_URL", "redis://localhost:6379/0"))
    parser.add_argument("--instruments", default="EUR_USD,GBP_USD,USD_JPY")
    parser.add_argument("--granularity", default="M5")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--recent-count", type=int, default=5)
    parser.add_argument("--max-entries", type=int, default=5000)
    args = parser.parse_args()

    # Connect to Redis
    try:
        r = redis.from_url(args.redis)
        r.ping()
        logger.info(f"Connected to Redis at {args.redis}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        sys.exit(1)

    # Connect to OANDA
    connector = OandaConnector(read_only=True)
    instruments = [i.strip() for i in args.instruments.split(",") if i.strip()]
    logger.info(f"Streaming {args.granularity} for {instruments}")

    while True:
        try:
            for inst in instruments:
                # Fetch recent candles with pagination logic manually
                # OANDA Limit is typically 5000 per request.
                total_needed = args.recent_count
                all_candles = []
                
                # We need to fetch in chunks, starting from "now" backwards? 
                # Oanda get_candles usually takes "count" or "from/to".
                # If we use "count", it returns last N.
                # If we need 50,000, we can't just ask for 50,000.
                # We have to iterate. Ideally, we just ask for 5000, get the time of the oldest, 
                # then ask for 5000 before that time.
                
                # Initial fetch (latest)
                chunk_size = 5000 # OANDA limit
                remaining = total_needed
                to_time = None # Latest
                
                while remaining > 0:
                    fetch_count = min(remaining, chunk_size)
                    try:
                        # Need to support 'to' param in connector? 
                        # Standard OandaConnector might not expose it in get_candles if it's simple.
                        # Let's check if we can pass kwargs or if we need to implement "to".
                        # Assuming connector.get_candles(..., to=X) works if it passes **kwargs to api.
                        # If not, we might be stuck.
                        # Let's try passing 'to' if to_time is set.
                        
                        # Note: Oanda 'to' is exclusive usually.
                        batch = connector.get_candles(inst, granularity=args.granularity, count=fetch_count, to_time=to_time)
                    except TypeError:
                         # Fallback if 'to' not supported by wrapper: just fetch max and break
                         batch = connector.get_candles(inst, granularity=args.granularity, count=fetch_count)
                         all_candles = batch + all_candles # Newest at end
                         break
                    except Exception as e:
                        logger.error(f"Fetch error for {inst}: {e}")
                        break
                        
                    if not batch:
                        break
                        
                    # Prepend to list (we are fetching backwards)
                    # batch is usually [Oldest ... Newest]
                    all_candles = batch + all_candles
                    
                    remaining -= len(batch)
                    
                    # Update to_time for next batch (oldest of this batch)
                    # OANDA time format is RFC3339.
                    first_candle_time = batch[0]["time"]
                    to_time = first_candle_time
                    
                    # Safety break if we aren't making progress
                    if len(batch) < fetch_count:
                         break
                         
                    time.sleep(0.5) # Rate limit protection

                candles = all_candles
                
                if not candles:
                    continue

                pipe = r.pipeline()
                key = f"md:candles:{inst.upper()}:{args.granularity}"
                
                count_new = 0
                for c in candles:
                    # Normalize timestamp to ms
                    # OANDA returns RFC3339 string usually
                    ts_str = c.get("time")
                    try:
                        dt = time.strptime(ts_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                        ts_ms = int(time.mktime(dt) * 1000)
                    except:
                        continue
                    
                    # Ensure minimal payload matches candle_utils expectations
                    payload = {
                        "t": ts_ms,
                        "mid": c.get("mid"),
                        "v": c.get("volume"),
                        "complete": c.get("complete")
                    }
                    
                    # Add to ZSET
                    blob = json.dumps(payload)
                    pipe.zadd(key, {blob: ts_ms})
                    count_new += 1
                
                # Trim
                pipe.zremrangebyrank(key, 0, -args.max_entries - 1)
                pipe.execute()
                # logger.debug(f"Pushed {count_new} candles for {inst}")

        except Exception as e:
            logger.error(f"Stream error: {e}")
        
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
