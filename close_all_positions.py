#!/usr/bin/env python3
"""
Close All Positions Script
"""
import sys
import os
import time

# Ensure path includes root
sys.path.insert(0, os.getcwd())

try:
    from scripts.trading.oanda import OandaConnector
except ImportError:
    # Try alternate path if running from root
    from V3.scripts.trading.oanda import OandaConnector

def main():
    print("Connecting to OANDA...")
    client = OandaConnector(read_only=False)
    
    if not client.account_id:
        print("Error: OANDA_ACCOUNT_ID not found in environment.")
        return

    print(f"Using Account: {client.account_id}")
    positions = client.positions()
    
    if not positions:
        print("No open positions found.")
        return

    print(f"Found {len(positions)} positions. Closing now...")

    for pos in positions:
        instrument = pos["instrument"]
        long_units = int(pos.get("long", {}).get("units", 0))
        short_units = int(pos.get("short", {}).get("units", 0))
        
        if long_units == 0 and short_units == 0:
            continue
            
        print(f"Closing {instrument} (Long: {long_units}, Short: {short_units})...")
        
        # Close Long
        if long_units > 0:
            resp = client.close_position(instrument, long_units)
            print(f"  > Closed Long: {resp}")
            
        # Close Short
        if short_units < 0:
            resp = client.close_position(instrument, short_units)
            print(f"  > Closed Short: {resp}")
        
        # OANDA rate limit politeness
        time.sleep(0.5)

    print("All positions processed.")
    
    # Verify
    time.sleep(2)
    final_pos = client.positions()
    open_count = sum(1 for p in final_pos if int(p.get("long", {}).get("units", 0)) != 0 or int(p.get("short", {}).get("units", 0)) != 0)
    
    if open_count == 0:
        print("SUCCESS: No open positions remaining.")
    else:
        print(f"WARNING: {open_count} positions still open. Check logs.")

if __name__ == "__main__":
    main()
