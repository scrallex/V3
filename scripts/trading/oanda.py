"""Minimal OANDA client wrapper used by the lean trading stack."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

API_BASE = os.getenv("OANDA_BASE_URL", "https://api-fxtrade.oanda.com")
PRACTICE_BASE = os.getenv("OANDA_PRACTICE_URL", "https://api-fxpractice.oanda.com")


def _api_base() -> str:
    return (
        API_BASE
        if str(os.getenv("OANDA_ENVIRONMENT", "live")).lower() == "live"
        else PRACTICE_BASE
    )


class OandaConnector:
    """Tiny wrapper around the handful of REST calls we rely on."""

    def __init__(self, *, read_only: bool | None = None) -> None:
        self.api_key = os.getenv("OANDA_API_KEY")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.read_only = (
            bool(read_only)
            if read_only is not None
            else str(os.getenv("READ_ONLY", "0")).lower() in ("1", "true")
        )
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept-Datetime-Format": "RFC3339",
                }
            )
        else:
            logger.warning(
                "OANDA_API_KEY not configured; connector will operate in no-op mode"
            )
        if not self.account_id:
            logger.warning("OANDA_ACCOUNT_ID not configured")

    # ------------------------------------------------------------------
    # Core request helper
    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.api_key or not self.account_id:
            return None
        if self.read_only and method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
            logger.debug("READ_ONLY guard active: %s %s skipped", method, path)
            return None
        url = f"{_api_base()}{path}"
        try:
            resp = self.session.request(
                method, url, params=params, json=json_body, timeout=15
            )
            if 200 <= resp.status_code < 300:
                return resp.json() if resp.content else {}
            logger.warning(
                "OANDA request failed: %s %s -> %s", method, path, resp.status_code
            )
        except Exception as exc:  # pragma: no cover - network failure path
            logger.error("OANDA request error: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def pricing(self, instruments: Iterable[str]) -> Dict[str, Dict[str, float]]:
        if not instruments:
            return {}
        payload = (
            self._request(
                "GET",
                f"/v3/accounts/{self.account_id}/pricing",
                params={"instruments": ",".join(sorted(instruments))},
            )
            or {}
        )
        out: Dict[str, Dict[str, float]] = {}
        for item in payload.get("prices", []) or []:
            try:
                bid = float(item["bids"][0]["price"]) if item.get("bids") else None
                ask = float(item["asks"][0]["price"]) if item.get("asks") else None
            except Exception:
                bid = ask = None
            mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
            out[item.get("instrument", "").upper()] = {
                "bid": bid,
                "ask": ask,
                "mid": mid,
            }
        return out

    def place_market_order(
        self,
        instrument: str,
        units: int,
        *,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if not units:
            return None
        body: Dict[str, Any] = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        if stop_loss is not None:
            body["order"]["stopLossOnFill"] = {"price": f"{float(stop_loss):.5f}"}
        if take_profit is not None:
            body["order"]["takeProfitOnFill"] = {"price": f"{float(take_profit):.5f}"}
        return self._request(
            "POST", f"/v3/accounts/{self.account_id}/orders", json_body=body
        )

    def positions(self) -> List[Dict[str, Any]]:
        payload = (
            self._request("GET", f"/v3/accounts/{self.account_id}/positions") or {}
        )
        return payload.get("positions", []) or []

    def account(self) -> Dict[str, Any]:
        payload = self._request("GET", f"/v3/accounts/{self.account_id}") or {}
        return payload

    def close_position(
        self, instrument: str, units: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        body: Dict[str, Any] = {}
        if units is not None:
            field = "longUnits" if units > 0 else "shortUnits"
            body[field] = str(abs(units))
        else:
            body = {"longUnits": "ALL", "shortUnits": "ALL"}
        return self._request(
            "PUT",
            f"/v3/accounts/{self.account_id}/positions/{instrument}/close",
            json_body=body,
        )

    def get_candles(
        self,
        instrument: str,
        *,
        granularity: str = "M5",
        count: int = 200,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        price: str = "M",
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"granularity": granularity}
        if from_time:
            params["from"] = from_time
        if to_time:
            params["to"] = to_time
        if price:
            params["price"] = price
        if not from_time and not to_time:
            params["count"] = min(max(count, 1), 5000)
        payload = (
            self._request(
                "GET",
                f"/v3/instruments/{instrument}/candles",
                params=params,
            )
            or {}
        )
        candles: List[Dict[str, Any]] = []
        for item in payload.get("candles", []) or []:
            mid = item.get("mid") or {}
            candles.append(
                {
                    "time": item.get("time"),
                    "bid": item.get("bid"),
                    "ask": item.get("ask"),
                    "mid": {k: float(v) for k, v in mid.items()} if mid else {},
                    "volume": int(item.get("volume", 0)),
                    "complete": item.get("complete", False),
                }
            )
        return candles


# =============================================================================
# Convenience functions used by the service / API layer
# =============================================================================


def pricing(trading_service: Any, instruments: List[str]) -> Dict[str, Any]:
    if not getattr(trading_service, "oanda", None):
        return {"prices": {}}
    return {"prices": trading_service.oanda.pricing(instruments)}


def current_price(
    trading_service: Any, instrument: str
) -> Optional[Tuple[float, float]]:
    payload = pricing(trading_service, [instrument])
    price = (
        payload["prices"].get(instrument)
        if isinstance(payload.get("prices"), dict)
        else None
    )
    if not price:
        return None
    return price.get("bid"), price.get("ask")


def submit_market_order(
    trading_service: Any,
    instrument: str,
    units: int,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    connector = getattr(trading_service, "oanda", None)
    if not connector:
        return None
    return connector.place_market_order(
        instrument, units, stop_loss=stop_loss, take_profit=take_profit
    )


def close_position(
    trading_service: Any, instrument: str, units: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    connector = getattr(trading_service, "oanda", None)
    if not connector:
        return None
    try:
        parsed_units = int(units) if units is not None else None
    except Exception:
        parsed_units = None
    return connector.close_position(instrument, parsed_units)


def fetch_and_store_candles(
    trading_service: Any, instrument: str, granularity: str = "M5", count: int = 200
) -> bool:
    connector = getattr(trading_service, "oanda", None)
    if not connector:
        return False
    candles = connector.get_candles(instrument, granularity=granularity, count=count)
    if not candles:
        return False
    store = getattr(trading_service, "candle_sink", None)
    if callable(store):
        store(instrument, candles)
    return True


def fetch_candles_for_enabled_pairs(
    trading_service: Any, granularity: str = "M5", count: int = 200
) -> None:
    for inst in getattr(trading_service, "enabled_pairs", []) or []:
        fetch_and_store_candles(
            trading_service, inst, granularity=granularity, count=count
        )


def get_stored_candles(
    trading_service: Any, instrument: str, granularity: str = "M5", limit: int = 200
) -> List[Dict[str, Any]]:
    store = getattr(trading_service, "candle_source", None)
    if callable(store):
        return store(instrument, granularity, limit)
    return []


def positions(trading_service: Any) -> List[Dict[str, Any]]:
    connector = getattr(trading_service, "oanda", None)
    return connector.positions() if connector else []


def account_info(trading_service: Any) -> Dict[str, Any]:
    connector = getattr(trading_service, "oanda", None)
    payload = connector.account() if connector else {}
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    return payload


__all__ = [
    "OandaConnector",
    "account_info",
    "close_position",
    "current_price",
    "fetch_and_store_candles",
    "fetch_candles_for_enabled_pairs",
    "get_stored_candles",
    "positions",
    "pricing",
    "submit_market_order",
]
