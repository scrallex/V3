"""Shared helpers for parsing Valkey candle payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from research.regime_manifold.codec import Candle


def _coerce_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except Exception:
        return 0.0


def _parse_timestamp(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def candle_from_payload(payload: Mapping[str, Any]) -> Candle:
    """Convert a generic JSON payload from md:candles into a Candle dataclass."""

    timestamp = _parse_timestamp(
        payload.get("t") or payload.get("timestamp") or payload.get("time")
    )

    def pick_price(keys: tuple[str, ...]) -> float:
        for key in keys:
            if key in payload:
                value = _coerce_float(payload[key])
                if value:
                    return value
        mid = payload.get("mid")
        if isinstance(mid, Mapping):
            for key in keys:
                if key in mid:
                    value = _coerce_float(mid[key])
                    if value:
                        return value
        return 0.0

    open_ = pick_price(("o", "open"))
    high = pick_price(("h", "high"))
    low = pick_price(("l", "low"))
    close = pick_price(("c", "close"))
    volume = _coerce_float(payload.get("v") or payload.get("volume"))
    spread_value: Optional[float] = None
    if "spread" in payload:
        spread_value = _coerce_float(payload.get("spread"))
    elif "bid" in payload and "ask" in payload:
        bid = payload.get("bid")
        ask = payload.get("ask")
        if isinstance(bid, Mapping) and isinstance(ask, Mapping):
            spread_value = abs(_coerce_float(ask.get("c")) - _coerce_float(bid.get("c")))
    spread = spread_value if spread_value and spread_value > 0 else max(high - low, 0.0)
    return Candle(timestamp, open_, high, low, close, volume, spread)
