#!/usr/bin/env python3
"""
Backfill S5 candles to Redis for Regime Service.
Requires ~43,000 candles (60 hours) for H1 aggregation.
We will fetch 4 days (approx 70k S5 candles) to be safe.
"""

import os
import sys
import json
import logging
import asyncio
import redis.asyncio as redis
from datetime import datetime, timedelta, timezone

# Add path to scripts to import OandaConnector
sys.path.append("/sep")
sys.path.append("/app")
from scripts.trading.oanda import OandaConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backfill")

PAIRS = os.getenv("HOTBAND_PAIRS", "EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CHF,USD_CAD,NZD_USD").split(",")
REDIS_URL = os.getenv("VALKEY_URL", "redis://valkey:6379/0")

async def backfill_pair(connector, r, instrument):
    logger.info(f"Backfilling {instrument}...")
    
    # Target: 4 days ago to now
    to_time = datetime.now(timezone.utc)
    from_time = to_time - timedelta(days=4)
    
    # OANDA limit is 5000 per request.
    # We need to chunk it.
    
    current_from = from_time
    total_count = 0
    
    while current_from < to_time:
        # Request usually takes "from" only and gives 5000 candles forward, 
        # but OandaConnector.get_candles wrapper uses count if from/to not specified?
        # Actually OandaConnector.get_candles supports from/to.
        
        # We need to be careful with "count" vs "to".
        # Let's verify OandaConnector logic.
        # It passes `from` and `to` and `count` to params.
        
        # We will use strict 5000 count chunks logic manually or just rely on `to_time`?
        # OANDA V3 candles endpoint: if `to` is specified, `count` is ignored? No.
        # "The number of candles to return... default 500, max 5000."
        # If `from` is specified, it returns candles FROM that time.
        
        # Let's loop by fetching 5000 candles starting from `current_from`
        # We can't easily know the EXACT time of the 5000th candle without fetching.
        # So we fetch, take the last candle time, and use that as next `from`.
        
        candles = connector.get_candles(
            instrument,
            granularity="S5",
            count=5000,
            from_time=current_from.isoformat().replace("+00:00", "Z"),
            price="M"
        )
        
        if not candles:
            logger.warning(f"No candles returned for {instrument} at {current_from}")
            break
            
        # Push to Redis
        pipe = r.pipeline()
        key = f"md:candles:{instrument}:S5"
        
        added = 0
        last_time_str = ""
        
        for c in candles:
            # Format: Same as streamer
            # {"inst": "...", "gran": "S5", "t": ms, "mid": {"o":..., "c":...}}
            # Actually regime service expects: {"mid": ..., "t": ...}
            # Streamer format: 
            # item = {"t": t_ms, "mid": {"o": o, "h": h, "l": l, "c": c}, "vol": vol}
            
            t_str = c["time"] # "2024-01-01T00:00:00.000000000Z"
            # Parse to ms
            dt = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
            t_ms = int(dt.timestamp() * 1000)
            
            mid = c["mid"]
            val = {
                "t": t_ms,
                "mid": mid,
                "vol": c["volume"]
            }
            
            pipe.zadd(key, {json.dumps(val): t_ms})
            added += 1
            last_time_str = t_str
            
        await pipe.execute()
        total_count += added
        logger.info(f"  Pushed {added} candles. Last: {last_time_str}")
        
        # Update current_from
        last_dt = datetime.fromisoformat(last_time_str.replace("Z", "+00:00"))
        if last_dt >= to_time or added < 10:
            break
        
        current_from = last_dt + timedelta(seconds=5)

    logger.info(f"Finished {instrument}: {total_count} total candles.")

async def main():
    connector = OandaConnector(read_only=True)
    r = redis.from_url(REDIS_URL, decode_responses=True)
    
    tasks = [backfill_pair(connector, r, pair) for pair in PAIRS]
    await asyncio.gather(*tasks)
    await r.aclose()

if __name__ == "__main__":
    asyncio.run(main())
