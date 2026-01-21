#!/usr/bin/env python3
"""
Verify Dashboard API Logic
==========================
Simulates the backend API response for Dashboard endpoints.
This runs LOCALLY to prove the code logic is correct before deployment.
"""

import sys
import logging
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

# Ensure path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.trading_service import TradingService
from scripts.trading.portfolio_manager import PortfolioManager


# Mock OANDA to avoid creds/network issues during logical check
def mock_oanda_connector(*args, **kwargs):
    m = MagicMock()
    m.get_candles.return_value = [
        {"time": "2025-01-01T12:00:00Z", "mid": {"c": 1.0500}, "volume": 100},
        {"time": "2025-01-01T12:05:00Z", "mid": {"c": 1.0510}, "volume": 150},
    ]
    m.account_info.return_value = {"balance": 10000.0, "nav": 10000.0}
    m.positions.return_value = []
    return m


def main():
    print(">>> Initializing TradingService (Mocked Mode)...")
    # Patch OANDA
    import scripts.trading.oanda as oanda_module

    oanda_module.OandaConnector = mock_oanda_connector

    service = TradingService(read_only=True)

    # Inject fake gate data to test dashboard visualization logic
    print(">>> Injecting Fake V3 Gate Data...")
    service.portfolio_manager.latest_gate_payloads = MagicMock(
        return_value={
            "EUR_USD": {
                "ts": time.time(),
                "regime": "YELLOW",
                "lambda": 0.51,  # V3 Threshold area
                "component_metrics": {"entropy": 0.8, "trend": 0.5},
                "hazard_threshold": 0.6,
                "repetitions": 2,
                "bundle_hits": ["MB003"],
                "raw": {"regime": {"realized_vol": 0.0006}},  # Vol > 0.0005 -> Admit
            }
        }
    )

    print(">>> Testing /api/metrics/nav...")
    nav_data = service.nav_metrics()
    print(f"NAV Status: {json.dumps(nav_data, indent=2)}")

    print("\n>>> Testing /api/metrics/gates...")
    gate_data = service.gate_metrics()
    print(f"Gates Payload: {json.dumps(gate_data, indent=2)}")

    gates = gate_data.get("gates", [])
    if not gates:
        print("ERROR: No gates returned!")
        sys.exit(1)

    eur_gate = gates[0]
    print(f"\n[Check] EUR_USD Regime: {eur_gate['regime']}")
    print(f"[Check] EUR_USD Hazard: {eur_gate['hazard']}")

    if eur_gate["regime"] == "YELLOW":
        print("SUCCESS: Dashboard API logic correctly serializes V3 Regime data.")
    else:
        print("FAILURE: Dashboard API logic did not return expected data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
