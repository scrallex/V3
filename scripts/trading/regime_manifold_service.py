#!/usr/bin/env python3
"""
V3 Regime Manifold Service
Uses Native C++ Bindings for Zero-Lag Manifold Calculation.
Logic: High Stability (0.25+) -> Mean Reversion.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import redis
from prometheus_client import Gauge, start_http_server

# Ensure repository root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Add src to sys.path for sep_text_manifold
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


_configure_logging("INFO")
logger = logging.getLogger("regime-manifold-service")

# Import Native Bindings
# Import Native Bindings or Real Python Fallback
try:
    from sep_text_manifold.encode import encode_window
    ENCODE_AVAILABLE = True
except ImportError as e:
    logger.critical("Failed to import sep_text_manifold.encode: %s", e)
    ENCODE_AVAILABLE = False
    sys.exit(1) # Fail hard if we can't even load the Python logic

try:
    from sep_text_manifold.native import analyze_bits, score_bytes
    HAVE_NATIVE = True
except ImportError as e:
    logger.warning("Failed to import native bindings: %s. Using Python fallback.", e)
    HAVE_NATIVE = False

# V3 Configuration (Hardcoded for Robustness based on Validation)
CONFIG = {
    "H": 0.25,  # Stability Threshold
    "T": 8,  # Hold Hours (Not used here, but part of logic)
    "WINDOW": 60,  # 60 Candles (H1 equivalent if aggregated, but we use what we have)
}


def _now_ms() -> int:
    return int(time.time() * 1000)


class V3RegimeService:
    def __init__(
        self,
        redis_url: str,
        instruments: List[str],
        prom_port: int,
        granularity: str = "H1",
    ):
        self.redis = redis.from_url(redis_url)
        self.instruments = instruments
        self.prom_port = prom_port
        self.granularity = granularity
        self._stop = False
        self.metric_hazard = Gauge("regime_hazard", "Hazard Metric", ["instrument"])
        self.metric_stability = Gauge(
            "regime_stability", "Stability Metric", ["instrument"]
        )

    if not HAVE_NATIVE:
        logger.warning(
            "Native Manifold bindings not found! Running in Pure Python fallback mode. Performance may be degraded."
        )

    def run(self):
        logger.info(
            "V3 Regime Service Starting for %s (Native Mode: %s)",
            self.instruments,
            HAVE_NATIVE,
        )
        start_http_server(self.prom_port)

        while not self._stop:
            for inst in self.instruments:
                try:
                    self._process_instrument(inst)
                except Exception:
                    logger.exception("Error processing %s", inst)

            time.sleep(2.0)  # Loop every 2s

    def stop(self):
        self._stop = True

    def _process_instrument(self, instrument: str):
        # 1. Fetch recent candles (S5 or H1?)
        # For V3, we want H1 candles.
        # If Streamer provides M1/M5, we might need to query OANDA or use what's in Redis.
        # Assuming existing Redis keys: "md:candles:{INST}:H1" (if streamer covers it)
        # OR we fetch M1 and Aggregate?
        # Simpler: Use "pricing:history:H1:{INST}" key from TradingService cache?
        # Or Oanda direct?
        # Let's assume Streamer is writing candles to `md:candles:{INST}:{GRANULARITY}`
        # We will use "H1" if available, or fall back to M1 and aggregated.
        # Actually, `fetch_oanda_data.py` showed S5 -> H1.
        # Let's try to read "H1" from Redis. The Streamer usually supports it.

        # 1. Fetch recent candles from Streamer ZSET
        # Key format: md:candles:{INST}:{GRANULARITY}
        # 1. Fetch recent S5 candles from Streamer ZSET to aggregate
        # Key format: md:candles:{INST}:S5
        # We need 60 hours of history. 1 hour = 720 S5 candles. 60 * 720 = 43200 candles.
        # This is heavy for a ZRANGE. Let's assume we fetch what fits (Streamer max-entries is 50k now).
        key_s5 = f"md:candles:{instrument.upper()}:S5"
        
        # Lookback: 50,000 should cover ~69 hours
        raw_rows = self.redis.zrange(key_s5, -50000, -1)
        
        if not raw_rows:
            return

        # Parse S5 Points
        s5_points = []
        for row in raw_rows:
            try:
                if isinstance(row, bytes):
                    row = row.decode("utf-8")
                c = json.loads(row)
                mid = c.get("mid")
                # Normalize mid
                if isinstance(mid, dict):
                     close = float(mid.get("c", 0.0))
                     open_p = float(mid.get("o", 0.0))
                else:
                     close = float(mid)
                     open_p = close
                
                s5_points.append({
                    "time": int(c.get("t")), # ms
                    "open": open_p,
                    "close": close
                })
            except Exception:
                continue

        # Aggregate to H1
        # Bucket by Hour (3600*1000 ms)
        h1_bars = {}
        for p in s5_points:
            ts = p["time"]
            # Floor to hour
            hour_ts = (ts // 3600000) * 3600000
            
            if hour_ts not in h1_bars:
                h1_bars[hour_ts] = {
                    "time": hour_ts,
                    "open": p["open"],
                    "close": p["close"], # Will update
                    "count": 0
                }
            
            # Update Close (keep updating until last point in bucket)
            h1_bars[hour_ts]["close"] = p["close"]
            h1_bars[hour_ts]["count"] += 1

        # Sort by time
        sorted_ts = sorted(h1_bars.keys())
        points = [h1_bars[ts] for ts in sorted_ts]

        if len(points) < 60:
            logger.debug("Insufficient H1 history for %s: %d/60 (from %d S5 points)", instrument, len(points), len(s5_points))
            return



        if len(points) < 60:
            logger.debug("Insufficient history for %s: %d/60", instrument, len(points))
            return

        # 2. Encode using Native Manifold (Price String)
        # Take last 60 closes
        window_points = points[-60:]
        closes = [str(p["close"]) for p in window_points]  # Use close price
        # Format: "1.0500 1.0510 ..."
        price_string = " ".join(closes)

        encoded_payload = price_string.encode("utf-8")

        try:
            metrics = encode_window(encoded_payload)
        except Exception as e:
            logger.error("Encoding failed for %s: %s", instrument, e)
            return

        stability = metrics.get("stability", 0.0)
        hazard = metrics.get("lambda_hazard", 0.0)

        self.metric_stability.labels(instrument=instrument).set(stability)
        self.metric_hazard.labels(instrument=instrument).set(hazard)

        # DEBUG: Log every 10th update or if stability > 0 to avoid noise, but for now log everything to prove it works
        logger.info("UPDATE %s: Pts=%d Stability=%.4f Hazard=%.4f", instrument, len(points), stability, hazard)

        # 3. Strategy Logic (Mean Reversion)
        # If Stability > 0.25 (High Stability Regime)
        # Direction = Revert last candle

        signal = "NEUTRAL"
        if stability > CONFIG["H"]:
            last_close = float(window_points[-1]["close"])
            last_open = float(
                window_points[-1].get("open", last_close)
            )  # Pricing history might not have open?
            # TradingService.price_history only stores "close" usually?
            # Check price_history method in trading_service.py:
            # "series.append({'time': time_str, 'close': price_val})"
            # It ONLY stores Close.
            # So we can't determine Open vs Close of last bar easily.
            # Workaround: Compare Close[-1] vs Close[-2].
            prev_close = float(window_points[-2]["close"])

            if last_close > prev_close:
                signal = "SHORT"  # Revert UP move
            elif last_close < prev_close:
                signal = "LONG"  # Revert DOWN move

        # 4. Write Gate
        gate_payload = {
            "instrument": instrument,
            "ts_ms": _now_ms(),
            "v3_signal": signal,
            "stability": stability,
            "hazard": hazard,
            "regime": "STABLE" if stability > 0.25 else "UNSTABLE",
            "admit": True,  # Always admit to let PM decide execution
            "hold_hours": CONFIG["T"],
        }

        # Write to Gate keys
        key_last = f"gate:last:{instrument.upper()}"
        self.redis.set(key_last, json.dumps(gate_payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--redis", default=os.getenv("VALKEY_URL", "redis://localhost:6379/0")
    )
    parser.add_argument("--prom-port", type=int, default=9105)
    parser.add_argument(
        "--instruments",
        default=os.getenv("HOTBAND_PAIRS", "EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CHF"),
    )
    args = parser.parse_args()

    inst_list = [x.strip() for x in args.instruments.split(",")]

    service = V3RegimeService(args.redis, inst_list, args.prom_port)

    def _shutdown(sig, frame):
        logger.info("Shutdown signal received")
        service.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    service.run()


if __name__ == "__main__":
    main()
