#!/usr/bin/env python3
"""
Inspect Redis keys to confirm data flow mismatch.
"""
import os
import redis
import json

def main():
    redis_url = os.getenv("VALKEY_URL", "redis://localhost:6379/0")
    print(f"Connecting to {redis_url}...")
    try:
        r = redis.from_url(redis_url)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # Check for streamer keys
    streamer_keys = r.keys("md:candles:*")
    print(f"\nStreamer Keys (md:candles:*): {len(streamer_keys)} found")
    for k in streamer_keys[:5]:
        k_str = k.decode('utf-8')
        t = r.type(k_str).decode('utf-8')
        print(f"  {k_str} [{t}]")
        if t == 'zset':
             count = r.zcard(k_str)
             print(f"    Count: {count}")

    # Check for regime expected keys
    regime_keys = r.keys("pricing:history:*")
    print(f"\nRegime Expected Keys (pricing:history:*): {len(regime_keys)} found")
    for k in regime_keys[:5]:
        print(f"  {k.decode('utf-8')}")

if __name__ == "__main__":
    main()
