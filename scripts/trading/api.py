#!/usr/bin/env python3
"""Minimal HTTP API surface for the lean SEP trading service."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


class TradingAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler exposing a handful of JSON endpoints."""

    server_version = "SEPTrading/2.0"

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def svc(self):  # type: ignore[override]
        return getattr(self.server, "trading_service", None)

    # ------------------------------------------------------------------
    # HTTP verbs
    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # pragma: no cover - exercised in integration flow
        parsed = urlparse(self.path)
        if parsed.path in {"/health", "/api/health", "/api/status"}:
            self._send_json(self._health_payload(parsed.path))
            return
        if parsed.path == "/api/pricing":
            params = parse_qs(parsed.query or "")
            instruments = params.get("instrument") or params.get("instruments") or []
            if not instruments:
                instruments = list(getattr(self.svc, "enabled_pairs", []) or [])
            payload = self.svc.get_pricing(instruments) if self.svc else {"prices": {}}
            self._send_json(payload)
            return
        if parsed.path == "/api/backtests/status":
            payload = self.svc.backtest_status() if self.svc else {"error": "service_unavailable"}
            self._send_json(payload if isinstance(payload, dict) else {"error": "invalid_payload"})
            return
        if parsed.path == "/api/backtests/latest":
            payload = self.svc.latest_backtests() if self.svc else {"error": "service_unavailable"}
            self._send_json(payload if isinstance(payload, dict) else {"error": "invalid_payload"})
            return
        if parsed.path == "/api/metrics/nav":
            payload = self.svc.nav_metrics() if self.svc else {}
            self._send_json({"nav": payload})
            return
        if parsed.path == "/api/metrics/gates":
            payload = self.svc.gate_metrics() if self.svc else {}
            self._send_json(payload)
            return
        if parsed.path == "/api/evidence/signal-outcomes":
            payload = self.svc.signal_outcomes() if self.svc else {"error": "service_unavailable"}
            self._send_json(payload if isinstance(payload, dict) else {"error": "invalid_payload"})
            return
        if parsed.path == "/api/evidence/bundle-outcomes":
            payload = self.svc.bundle_outcomes() if self.svc else {"error": "service_unavailable"}
            self._send_json(payload if isinstance(payload, dict) else {"error": "invalid_payload"})
            return
        if parsed.path == "/api/evidence/roc-summary":
            payload = self.svc.regime_roc_summary() if self.svc else {"error": "service_unavailable"}
            self._send_json(payload if isinstance(payload, dict) else {"error": "invalid_payload"})
            return
        if parsed.path == "/api/pricing/history":
            params = parse_qs(parsed.query or "")
            instrument = (params.get("instrument") or [""])[0]
            granularity = (params.get("granularity") or ["M5"])[0]
            count_raw = (params.get("count") or ["120"])[0]
            try:
                count = int(count_raw)
            except Exception:
                count = 120
            payload = self.svc.price_history(instrument, granularity=granularity, count=count) if self.svc else {"points": []}
            self._send_json(payload)
            return
        if parsed.path == "/api/account":
            payload = self.svc.get_oanda_account_info() if self.svc else {}
            # OANDA response already wraps in "account", so send direct
            self._send_json(payload)
            return
        if parsed.path == "/api/positions":
            # OANDA positions returns a list, wrap in dict "positions" if not already?
            # get_oanda_positions calls connector.positions() -> returns list.
            # So {"positions": payload} IS correct for positions.
            # But let's check oanda.py positions().
            # oanda.py: return payload.get("positions", [])
            # So payload is LIST.
            # api.py: self._send_json({"positions": payload}) -> {"positions": [list]}. Correct.
            # Wait. frontend JS: posData.positions.
            # So positions was correct?
            # User said "Unrealized PnL $NaN". That's account data.
            # "Active Positions" table empty or just not mentioned as broken. "everything... broken".
            # I will only change account.
            payload = self.svc.get_oanda_positions() if self.svc else []
            self._send_json({"positions": payload})
            return
        self._send_json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:  # pragma: no cover - exercised in integration flow
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            payload = {}
        if self.path == "/api/kill-switch":
            flag = bool(payload.get("kill_switch", False))
            if self.svc:
                try:
                    self.svc.set_kill_switch(flag)
                except AttributeError:
                    self.svc.kill_switch_enabled = flag
            self._send_json({"kill_switch": flag})
            return
        if self.path == "/api/order":
            result = self.svc.place_order(  # type: ignore[assignment]
                payload.get("instrument", "EUR_USD"),
                int(payload.get("units", 0) or 0),
                payload.get("stop_loss"),
                payload.get("take_profit"),
            ) if self.svc else {"error": "service_unavailable"}
            self._send_json(result or {"ok": False})
            return
        if self.path == "/api/trade/close":
            result = self.svc.close_position(  # type: ignore[assignment]
                payload.get("instrument", "EUR_USD"),
                payload.get("units"),
            ) if self.svc else {"error": "service_unavailable"}
            self._send_json(result or {"ok": False})
            return
        if self.path == "/api/candles/fetch":
            instrument = payload.get("instrument")
            granularity = payload.get("granularity", "M5")
            count = int(payload.get("count", 200) or 200)
            ok = False
            if self.svc:
                if instrument:
                    ok = self.svc.fetch_and_store_candles(instrument, granularity, count)
                else:
                    self.svc.fetch_candles_for_enabled_pairs(granularity, count)
                    ok = True
            self._send_json({"success": ok})
            return
        if self.path == "/api/backtests/run":
            start = payload.get("start") if isinstance(payload, dict) else None
            end = payload.get("end") if isinstance(payload, dict) else None
            instruments = payload.get("instruments") if isinstance(payload, dict) else None
            success = False
            status_payload: Dict[str, Any]
            if self.svc:
                success, status_payload = self.svc.trigger_backtest(
                    start=start,
                    end=end,
                    instruments=instruments,
                )
            else:
                status_payload = {"error": "service_unavailable"}
            self._send_json(
                status_payload,
                status=202 if success else (409 if status_payload.get("state") == "running" else 400),
            )
            return
        self._send_json({"error": "not_found"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - noisy in tests
        logger.info("%s - %s", self.address_string(), format % args)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _health_payload(self, path: str) -> Dict[str, Any]:
        svc = self.svc
        base = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kill_switch": bool(getattr(svc, "kill_switch_enabled", True)),
            "trading_active": bool(getattr(svc, "trading_active", False)),
            "enabled_pairs": list(getattr(svc, "enabled_pairs", []) or []),
        }
        if path == "/health":
            return {"status": "ok", **base}
        if path == "/api/health":
            return {"service": "SEP Trading", **base}
        return {"service": "SEP Trading", "status": "running" if getattr(svc, "running", False) else "stopped", **base}


def start_http_server(trading_service: Any, host: str = "0.0.0.0", port: int = 8000) -> HTTPServer:
    """Start the HTTP server in a background thread and return the server."""

    class _Server(HTTPServer):  # pragma: no cover - thin wrapper
        def __init__(self, address):
            super().__init__(address, TradingAPIHandler)
            self.trading_service = trading_service

    server = _Server((host, port))

    def _serve() -> None:
        logger.info("HTTP server listening on %s:%s", host, port)
        server.serve_forever()

    thread = threading.Thread(target=_serve, name="TradingAPI", daemon=True)
    thread.start()
    return server


__all__ = ["start_http_server", "TradingAPIHandler"]
