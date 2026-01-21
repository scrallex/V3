"""Synthetic gate derivation for backtest fallback flows."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.trading.oanda import OandaConnector
from scripts.trading.portfolio_manager import StrategyInstrument, StrategyProfile

UTC = timezone.utc


@dataclass
class _Candle:
    time: datetime
    mid: float


def _parse_time(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _normalize_candles(candles: Sequence[Any]) -> List[_Candle]:
    out: List[_Candle] = []
    for row in candles:
        if hasattr(row, "time") and hasattr(row, "mid"):
            ts = getattr(row, "time")
            price = getattr(row, "mid")
        elif isinstance(row, dict):
            ts = row.get("time")
            price = row.get("mid")
            if isinstance(price, dict):
                price = price.get("c") or price.get("close") or price.get("C")
        else:
            continue
        if ts is None or price is None:
            continue
        if not isinstance(ts, datetime):
            try:
                ts = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(UTC)
            except Exception:
                continue
        try:
            mid = float(price)
        except Exception:
            continue
        out.append(_Candle(time=ts, mid=mid))
    out.sort(key=lambda candle: candle.time)
    return out


def _load_candles(
    instrument: str,
    *,
    start: datetime,
    end: datetime,
    granularity: str = "M1",
) -> List[_Candle]:
    connector = OandaConnector(read_only=True)
    payload = connector.get_candles(
        instrument,
        granularity=granularity,
        from_time=start.isoformat().replace("+00:00", "Z"),
        to_time=end.isoformat().replace("+00:00", "Z"),
    )
    if not payload:
        approx = int(max(1, min(5000, ((end - start).total_seconds() / 60) + 10)))
        payload = connector.get_candles(
            instrument,
            granularity=granularity,
            count=approx,
        )
    return _normalize_candles(payload or [])


def _entropy(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    total = len(labels)
    pos = sum(1 for value in labels if value > 0)
    neg = total - pos
    probs = [x / total for x in (pos, neg) if x > 0]
    return -sum(p * math.log(p, 2) for p in probs)


def _compute_metrics(
    returns_window: Sequence[float],
    streak_len: int,
) -> Tuple[float, float, float, float]:
    if not returns_window:
        return 0.0, 0.0, 0.0, 0.0
    magnitude = abs(sum(returns_window)) / max(1, len(returns_window))
    coherence = min(0.99, streak_len / max(3, len(returns_window)))
    stability = 0.0
    try:
        variance = statistics.pvariance(returns_window)
        stability = min(0.99, 1.0 / (1.0 + variance * 10))
    except statistics.StatisticsError:
        stability = 0.5
    entropy = _entropy([1 if val > 0 else -1 for val in returns_window])
    hazard = min(0.99, 1.0 - math.exp(-streak_len * max(0.5, magnitude * 400)))
    return coherence, stability, entropy, hazard


def derive_signals(
    instrument: str,
    start: datetime | str,
    end: datetime | str,
    *,
    candles: Optional[Sequence[Any]] = None,
    profile: Optional[StrategyInstrument] = None,
    granularity: str = "M1",
) -> List[Dict[str, Any]]:
    """Generate synthetic gate-like events when Valkey data is missing."""

    start_dt = _parse_time(start)
    end_dt = _parse_time(end)
    if candles is None:
        candles = _load_candles(instrument, start=start_dt, end=end_dt, granularity=granularity)
    else:
        candles = _normalize_candles(candles)

    if not candles or len(candles) < 3:
        return []

    profile_obj = profile
    if profile_obj is None:
        strategy = StrategyProfile.load(Path("config/echo_strategy.yaml"))
        profile_obj = strategy.get(instrument)

    min_reps = max(1, profile_obj.min_repetitions)
    guards = profile_obj.guards or {}
    min_coherence = float(guards.get("min_coherence") or 0.0)
    min_stability = float(guards.get("min_stability") or 0.0)
    max_entropy = float(guards.get("max_entropy") or 2.5)
    hazard_cap = float(profile_obj.hazard_max or 0.3)

    window_size = max(10, min(60, min_reps * 3))
    events: List[Dict[str, Any]] = []

    returns: List[float] = []
    streak_dir = 0
    streak_len = 0

    for prev, curr in zip(candles, candles[1:]):
        if prev.mid <= 0 or curr.mid <= 0:
            continue
        change = (curr.mid - prev.mid) / prev.mid
        returns.append(change)
        if len(returns) > window_size:
            returns.pop(0)

        direction = 1 if change >= 0 else -1
        if direction == streak_dir:
            streak_len += 1
        else:
            streak_dir = direction
            streak_len = 1

        if streak_len < min_reps:
            continue

        coherence, stability, entropy, hazard = _compute_metrics(returns, streak_len)
        if coherence < min_coherence or stability < min_stability or entropy > max_entropy:
            continue
        hazard = min(hazard_cap, hazard)

        direction_label = "BUY" if streak_dir > 0 else "SELL"
        reason_tags = ["synthetic:thresholds"]
        if streak_len == min_reps:
            reason_tags.append("synthetic:min_reps_met")

        ts_ms = int(curr.time.timestamp() * 1000)
        events.append(
            {
                "instrument": instrument.upper(),
                "admit": 1,
                "direction": direction_label,
                "lambda": round(hazard, 6),
                "components": {
                    "coherence": round(coherence, 6),
                    "stability": round(stability, 6),
                    "entropy": round(entropy, 6),
                },
                "repetitions": streak_len,
                "repetition_count": streak_len,
                "repetitions_lookback_minutes": window_size,
                "signal_key": f"synthetic:{instrument.upper()}:{ts_ms}",
                "first_seen_ms": ts_ms,
                "ts_ms": ts_ms,
                "schema_version": 1,
                "source": "synthetic",
                "reasons": reason_tags,
            }
        )

    return events


__all__ = ["derive_signals"]
