#!/usr/bin/env python3
"""
Execution Agent (Live)
- Connects to OANDA (REST API).
- Connects to Redis (Signals).
- Fetches NAV and Open Positions.
- Sizes Trades based on % NAV (Dynamic).
- Executes Orders (Market).
- Manages Overlap (Max Exposure per Pair).
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import redis
import requests

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("execution-agent")

# OANDA Config
OANDA_URL = os.getenv("OANDA_API_URL", "https://api-fxtrade.oanda.com")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
API_KEY = os.getenv("OANDA_API_KEY")

if not API_KEY or not ACCOUNT_ID:
    logger.error("Missing OANDA_API_KEY or OANDA_ACCOUNT_ID env vars.")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept-Datetime-Format": "RFC3339",
}

# Risk Config
TARGET_ALLOC_PER_PAIR = 12.0  # 1200% NAV per Pair (12x Leverage) -> Allows ~4 concurrent trades
MAX_LEVERAGE_GLOBAL = 45.0  # Safety Cap (Below OANDA 50x Limit)


class ExecutionAgent:
    def __init__(self, redis_url, instruments, poll_interval=10.0):
        self.redis = redis.from_url(redis_url)
        self.instruments = instruments
        self.poll_interval = poll_interval
        self.running = True
        self.state = {}  # Local state cache

    def run(self):
        print(f"DEBUG: Starting Execution Agent for {len(self.instruments)} pairs...")
        print(f"DEBUG: Target Allocation: {TARGET_ALLOC_PER_PAIR*100}% NAV per pair.")
        logger.info(f"Starting Execution Agent for {len(self.instruments)} pairs...")
        logger.info(f"Target Allocation: {TARGET_ALLOC_PER_PAIR*100}% NAV per pair.")

        while self.running:
            try:
                self.tick()
            except Exception as e:
                logger.error(f"Tick Loop Error: {e}", exc_info=True)
            time.sleep(self.poll_interval)

    def stop(self):
        self.running = False
        logger.info("Stopping Execution Agent...")

    def tick(self):
        # 1. Fetch Account Info (NAV)
        nav, margin_avail = self.get_account_summary()
        if not nav:
            return

        # 2. Fetch Open Positions
        positions = self.get_open_positions()

        # 3. Process Each Instrument
        for inst in self.instruments:
            self.process_instrument(inst, nav, positions.get(inst))

    def process_instrument(self, inst, nav, current_pos):
        # Fetch Signal from Redis
        key = f"gate:last:{inst}"
        raw = self.redis.get(key)
        if not raw:
            # print(f"DEBUG: No key for {inst}")
            return
        
        print(f"DEBUG: Found key {key} -> {raw[:50]}...")

        gate = json.loads(raw)
        signal_dir = gate.get("signal", "NEUTRAL")  # LONG, SHORT, NEUTRAL

        # Current State
        pos_units = (
            int(current_pos["long"]["units"]) - int(current_pos["short"]["units"])
            if current_pos
            else 0
        )

        # Logic
        # If Signal LONG and Not Long -> Buy
        # If Signal SHORT and Not Short -> Sell
        # If Signal NEUTRAL -> Close (or Hold? Strategy said hold 60m?)
        # For simplicity/safety on live: Signal NEUTRAL = Close.
        # This matches the backtest "Position = Signal" logic.

        target_units = 0

        if signal_dir == "LONG":
            target_units = self.calc_size(inst, nav, 1)
        elif signal_dir == "SHORT":
            target_units = -self.calc_size(inst, nav, -1)
        else:
            target_units = 0

        # Delta
        delta = target_units - pos_units

        # Threshold to trade (avoid noise)
        # e.g. min change of 100 units? Or simply if sign flips or zero.
        # We want to sticky execution.

        if delta == 0:
            return

        # Execution
        # CASE 1: Close (Target 0, Current != 0)
        if target_units == 0 and pos_units != 0:
            logger.info(f"{inst}: Signal {signal_dir}. Closing {pos_units}.")
            self.close_position(inst, current_pos)

        # CASE 2: Flip (Long -> Short or Short -> Long)
        elif (target_units > 0 and pos_units < 0) or (
            target_units < 0 and pos_units > 0
        ):
            logger.info(
                f"{inst}: Signal {signal_dir}. Flipping {pos_units} -> {target_units}."
            )
            self.close_position(inst, current_pos)  # Close first
            self.market_order(inst, target_units)

        # CASE 3: Entry (Zero -> Pos)
        elif pos_units == 0 and target_units != 0:
            logger.info(f"{inst}: Signal {signal_dir}. Entry {target_units}.")
            self.market_order(inst, target_units)

        # CASE 4: Re-Size (Increase/Decrease due to NAV Change?)
        # User said: "As account size grows, sizings should grow".
        # If we re-calculate target units every tick, we might drift trade constantly.
        # BETTER STRATEGY: Only re-size if deviation is > 10%?
        # Or just keep it simple: Sticky to signal. If signal is same, don't churn unless NAV doubled.
        # For this version: Stick to Signal Direction. If Direction matches, do nothing.

        elif (target_units > 0 and pos_units > 0) or (
            target_units < 0 and pos_units < 0
        ):
            # Same direction. Check size efficiency?
            # If current size is way off target (e.g. > 20% diff), adjust?
            # This handles "Growth".
            deviation = abs(target_units - pos_units) / abs(pos_units)
            if deviation > 0.20:
                logger.info(
                    f"{inst}: Re-Sizing {pos_units} -> {target_units} (NAV Drift)."
                )
                self.market_order(inst, delta)  # Adjust diff

    def calc_size(self, inst, nav, direction):
        # Target: 10% of NAV in Notional Exposure?
        # No, usually "Risk %" or "Notional".
        # User said "tied to NAV sizing as a percentage".
        # Assuming "Percentage of NAV allocated to Margined Notional".
        # If Allocation = 10% NAV.
        # Size = (NAV * 0.10) * Leverage?
        # Standard Conservative: Size = NAV * 1.0 (1x Leverage) / 7 pairs ~ 14% each.
        # Let's target Notional Value = NAV * TARGET_ALLOC_PER_PAIR?
        
        # FIX: Actually use the constant for sizing
        alloc_usd = nav * TARGET_ALLOC_PER_PAIR

        # Get Price
        price = self.get_price(inst)
        if not price:
            return 0

        # Cross conversion if needed (e.g. USD_JPY -> JPY terms).
        # Valid approximation: Units = Alloc_USD / Price  (if Base is Quote? No).
        # EUR_USD: Price ~ 1.10. Units = 1000. Notional = 1100 USD.
        # USD_JPY: Price ~ 150. Units = 1000. Notional = 1000 USD.

        # Simple approx for "Majors":
        # If USD is Quote (EUR_USD, GBP_USD, AUD_USD, NZD_USD):
        #   Notional = Units * Price
        #   Units = Alloc_USD / Price

        # If USD is Base (USD_JPY, USD_CHF, USD_CAD):
        #   Notional = Units * 1
        #   Units = Alloc_USD

        if inst.endswith("_USD"):
            units = alloc_usd / price
        else:
            units = alloc_usd
            
        return int(units)

    # --- OANDA API Helpers ---
    def get_account_summary(self):
        url = f"{OANDA_URL}/v3/accounts/{ACCOUNT_ID}/summary"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("account", {})
                return float(data.get("NAV", 0)), float(data.get("marginAvailable", 0))
            else:
                logger.error(f"Account fetch failed: {resp.text}")
                return None, None
        except Exception as e:
            logger.error(f"Account network error: {e}")
            return None, None

    def get_open_positions(self):
        url = f"{OANDA_URL}/v3/accounts/{ACCOUNT_ID}/positions"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                pos_list = resp.json().get("positions", [])
                return {p["instrument"]: p for p in pos_list}
            return {}
        except:
            return {}

    def get_price(self, inst):
        url = (
            f"{OANDA_URL}/v3/instruments/{inst}/candles?count=1&price=M&granularity=M1"
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                c = resp.json()["candles"][0]["mid"]
                return float(c["c"])
            return None
        except:
            return None

    def close_position(self, inst, current_pos=None):
        # Close specific side if known, else try ALL (which fails if one side empty)
        body = {}
        
        if current_pos:
            longs = int(current_pos.get("long", {}).get("units", 0))
            shorts = int(current_pos.get("short", {}).get("units", 0))
            if longs != 0:
                body["longUnits"] = "ALL"
            if shorts != 0:
                body["shortUnits"] = "ALL"
        else:
            # Fallback (legacy/risky)
            body = {"longUnits": "ALL", "shortUnits": "ALL"}

        if not body:
            logger.warning(f"Close requested for {inst} but no open units found in state.")
            return

        url = f"{OANDA_URL}/v3/accounts/{ACCOUNT_ID}/positions/{inst}/close"
        try:
            resp = requests.put(url, headers=HEADERS, json=body, timeout=5)
            if resp.status_code != 200:
                logger.error(f"Close failed {inst}: {resp.status_code} {resp.text}")
            else:
                logger.info(f"Closed {inst} successfully.")
        except Exception as e:
            logger.error(f"Close network error {inst}: {e}")

    def market_order(self, inst, units):
        url = f"{OANDA_URL}/v3/accounts/{ACCOUNT_ID}/orders"
        body = {
            "order": {
                "units": str(units),
                "instrument": inst,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT",
            }
        }
        try:
            resp = requests.post(url, headers=HEADERS, json=body, timeout=5)
            if resp.status_code != 201:
                logger.error(f"Order failed {inst} {units}: {resp.text}")
            else:
                logger.info(f"Filled {inst} {units}")
        except Exception as e:
            logger.error(f"Order network error {inst}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis", default="redis://localhost:6379/0")
    parser.add_argument(
        "--pairs", default="EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,USD_CAD,NZD_USD"
    )
    args = parser.parse_args()

    pairs = args.pairs.split(",")
    agent = ExecutionAgent(args.redis, pairs)

    def shutdown(sig, frame):
        agent.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    agent.run()


if __name__ == "__main__":
    main()
