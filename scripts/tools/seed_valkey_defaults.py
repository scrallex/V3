#!/usr/bin/env python3
"""
Seed Valkey (Redis) with default values for operation.
Sets kill switch to ENABLED (1) by default for safety.
"""

import os
import sys
import redis
import time

def main():
    redis_url = os.getenv("VALKEY_URL", "redis://valkey:6379/0")
    print(f"Connecting to Valkey at {redis_url}...")
    
    try:
        r = redis.from_url(redis_url)
        r.ping()
    except Exception as e:
        print(f"Failed to connect to Valkey: {e}")
        sys.exit(1)

    # Seed Kill Switch (Safe Default: True/1)
    # The trading service checks "ops:kill_switch".
    # If "1", trading is DISABLED.
    kill_switch_key = os.getenv("KILL_SWITCH_KEY", "ops:kill_switch")
    current = r.get(kill_switch_key)
    
    if current is None:
        print(f"Seeding {kill_switch_key} = 1 (ENABLED)")
        r.set(kill_switch_key, "1")
    else:
        print(f"{kill_switch_key} already set to {current.decode('utf-8')}")

    # Seed Risk Limits?
    # Currently RiskLimits are hardcoded in code, but we can reserve the key.
    # r.set("risk:config", "{}") 

    print("Valkey seeding complete.")

if __name__ == "__main__":
    main()
