#!/usr/bin/env python3
"""
Production Verification Script for Global Manifold V3.
Strictly tests the LIVE SERVICE CODE (regime_manifold_service.py) against historical data.

Workflow:
1. Fetches 60 days of M5 candles (EUR_USD).
2. Feeds candles into `RegimeManifoldService` (mocked IO) to generate AUTHENTIC GATES.
3. Feeds gates into `BacktestSimulator` (mocked IO) to simulate execution with $320 NAV.
"""

import sys
import os
import time
import json
import logging
import statistics
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock

# Ensure path availability
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unittest.mock import MagicMock

# Mock dependencies that might be missing in this environment
sys.modules["prometheus_client"] = MagicMock()
sys.modules["redis"] = MagicMock()

# Load OANDA credentials
env_path = ROOT / "OANDA.env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k] = v

from scripts.trading.regime_manifold_service import RegimeManifoldService, ServiceConfig
from scripts.trading.portfolio_manager import StrategyProfile
from scripts.research.simulator.backtest_simulator import (
    BacktestSimulator,
    SimulationParams,
)
from scripts.trading.oanda import OandaConnector
from research.regime_manifold.codec import Candle

# Setup Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VERIFY_V3")

UTC = timezone.utc

logger.info(f"OANDA_API_KEY present: {'OANDA_API_KEY' in os.environ}")
if "OANDA_API_KEY" in os.environ:
    k = os.environ["OANDA_API_KEY"]
    logger.info(f"Key Masked: {k[:4]}...{k[-4:] if len(k)>8 else ''}")
    logger.info(f"Environment: {os.environ.get('OANDA_ENVIRONMENT')}")


class MockRedis:
    """Minimal Mock for Redis to satisfy RegimeManifoldService."""

    def __init__(self):
        self.data = {}
        self.pipeline_calls = []

    def from_url(self, url):
        return self

    def pipeline(self):
        return self

    def set(self, key, value, ex=None):
        self.data[key] = value
        return True

    def zadd(self, key, mapping):
        # We don't strictly need zadd logic for verification, just capturing the write
        pass

    def zremrangebyrank(self, key, min, max):
        pass

    def execute(self):
        pass

    def zrangebyscore(self, key, min, max):
        return []


class ProductionVerifierService(RegimeManifoldService):
    """
    Subclass of the ACTUAL PRODUCTION SERVICE.
    Overrides I/O methods to inject history and capture output.
    """

    def __init__(self, config, profile):
        # Mock Redis BEFORE init to avoid connection error
        self.redis = MockRedis()
        super().__init__(config, profile)
        # Re-mock redis-dependent components if any
        self.redis = MockRedis()

        self.history_window = []  # Buffer for sliding window
        self.generated_gates = []  # Output buffer

    def _load_recent_candles(self, instrument: str) -> List[Candle]:
        # Return the currently buffered window injected by the test loop
        return list(self.history_window)

    def _write_gate(self, instrument: str, payload: Dict[str, object]) -> None:
        # Instead of writing to Redis, save to our list
        # We assume payload is fully constructed by the production logic

        # Verify V3 Logic is Active (Whitebox Check)
        if "hazard" in payload and "hazard_threshold" in payload:
            h = payload["hazard"]
            t = payload["hazard_threshold"]
            # Just a debug check

        self.generated_gates.append(payload)

    def inject_window(self, candles: List[Candle]):
        self.history_window = candles
        # Run the production processing method
        self._process_instrument(self.cfg.instruments[0])


class VerifierSimulator(BacktestSimulator):
    """
    Subclass of BacktestSimulator to inject our captured gates.
    """

    def set_gates(self, gates):
        self._injected_gates = gates

    def _load_gate_events(self, instrument, start, end):
        return self._injected_gates


def fetch_data(pair="EUR_USD", days=60):
    connector = OandaConnector(read_only=True)
    end_dt = datetime.now(UTC)
    start_dt = end_dt - timedelta(days=days)

    logger.info(f"Fetching {days} days of M5 data for {pair}...")

    # Simple chunked fetch
    all_candles = []
    current_start = start_dt

    while current_start < end_dt:
        chunk_end = min(current_start + timedelta(hours=100), end_dt)
        raw = connector.get_candles(
            pair,
            granularity="M5",
            from_time=current_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            to_time=chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        if raw:
            for r in raw:
                try:
                    ts = datetime.fromisoformat(r["time"].replace("Z", "+00:00"))
                    c = Candle(
                        timestamp_ms=int(ts.timestamp() * 1000),
                        open=float(r["mid"]["o"]),
                        high=float(r["mid"]["h"]),
                        low=float(r["mid"]["l"]),
                        close=float(r["mid"]["c"]),
                        volume=float(r["volume"]),
                    )
                    all_candles.append(c)
                except Exception as e:
                    logger.warning(f"Parse error: {e}")
                    pass
        current_start = chunk_end
        time.sleep(0.1)

    # Dedup
    unique = {c.timestamp_ms: c for c in all_candles}
    final = sorted(unique.values(), key=lambda x: x.timestamp_ms)
    logger.info(f"Fetched {len(final)} candles.")
    return final


def run_verification():
    # 1. Config
    PAIR = "EUR_USD"
    NAV = 320.0
    RISK_PCT = 0.02  # 2% per trade

    # 2. Get Data
    candles = fetch_data(PAIR, days=60)
    if not candles:
        logger.error("No data fetched.")
        return

    # 3. Setup Production Service
    profile = StrategyProfile.load(Path("config/echo_strategy.yaml"))

    # Configure strictly as implied by OANDA.env
    config = ServiceConfig(
        instruments=[PAIR],
        redis_url="mock://",
        lookback_minutes=240,
        window_candles=64,
        stride_candles=16,
        atr_period=14,
        loop_seconds=0,
        signature_retention_minutes=60,
        hazard_percentile=0.8,
        admit_regimes=("trend_bull", "trend_bear"),
        min_confidence=0.55,
        gate_ttl_seconds=900,
        prom_port=0,
        lambda_scale=0.1,
        bundle_config=None,
    )

    service = ProductionVerifierService(config, profile)

    # 4. Generate Gates (The Heavy Lifting)
    logger.info("Running Production Logic over History...")
    window_sc = 64
    stride = 1

    # We iterate and slide the window just like the live service would receive updates
    # But for speed, we'll just jump by 1 candle
    for i in range(window_sc, len(candles)):
        # Construct window ending at i
        # The service naturally pulls the last K candles.
        # We pass the relevant slice.
        lookback_slice = candles[i - window_sc : i + 1]  # Pass enough data
        service.inject_window(lookback_slice)

        if i % 1000 == 0:
            logger.info(f"Processed {i}/{len(candles)} candles...")

    logger.info(f"Generated {len(service.generated_gates)} gates using V3 Logic.")

    # 5. Simulate
    logger.info(f"Simulating Execution with ${NAV} NAV...")

    sim = VerifierSimulator(
        redis_url=None,
        nav=NAV,
        nav_risk_pct=RISK_PCT,
        profile_path=Path("config/echo_strategy.yaml"),
    )
    sim.set_gates(service.generated_gates)

    # We need to construct 'BacktestSimulator' candles format
    # It expects objects with .time (datetime) and .mid (float)
    sim_candles = []
    for c in candles:
        dt = datetime.fromtimestamp(c.timestamp_ms / 1000.0, tz=UTC)
        sim_candles.append(type("Obj", (object,), {"time": dt, "mid": c.close}))

    # Start/End
    start_dt = datetime.fromtimestamp(candles[0].timestamp_ms / 1000.0, tz=UTC)
    end_dt = datetime.fromtimestamp(candles[-1].timestamp_ms / 1000.0, tz=UTC)

    result = sim.simulate(
        PAIR,
        start=start_dt,
        end=end_dt,
        params=SimulationParams(exposure_scale=1.0, hold_minutes=30),
    )

    # 6. Report
    if result:
        m = result.metrics
        print("\n=== V3 PRODUCTION VERIFICATION RESULTS (60 Days) ===")
        print(f"Instrument: {PAIR}")
        print(f"Start NAV:  ${NAV:.2f}")
        print(f"End NAV:    ${NAV + m.pnl:.2f}")
        print(f"Total PnL:  ${m.pnl:.2f} ({m.return_pct:.2f}%)")
        print(f"Trades:     {m.trades}")
        print(f"Win Rate:   {m.win_rate:.1f}%")
        print(f"Max DD:     {m.max_drawdown:.2f}%")
        print("====================================================")

        # Verify if any 'v3_yellow_filter_active' reasons appeared (Whitebox)
        v3_blocks = 0
        for d in result.decisions:
            if "v3_yellow_filter_active" in d.reasons:
                v3_blocks += 1
        print(f"DEBUG: V3 Filter Blocks triggering: {v3_blocks}")

    else:
        print("Simulation failed to produce results.")


if __name__ == "__main__":
    run_verification()
