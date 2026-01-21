"""Backtest simulator that mirrors the live trading pipeline.

This module reuses the live stack's gate evaluation, trade planner, and risk
sizer to replay historical gate events (or synthetic fallbacks) against candle
series. The goal is to keep research outcomes aligned with production
behaviour.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import statistics
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None

try:  # pragma: no cover - optional dependency
    from scripts.research.fetch_history import chunked_fetch, GRANULARITY_SECONDS
except Exception:  # pragma: no cover - minimal environments may miss this module
    chunked_fetch = None  # type: ignore
    GRANULARITY_SECONDS: Dict[str, int] = {}

from scripts.trading.oanda import OandaConnector
from scripts.trading.portfolio_manager import (
    GateReader,
    SessionPolicy,
    StrategyInstrument,
    StrategyProfile,
    gate_evaluation,
)
from scripts.trading.risk_planner import (
    RiskLimits,
    RiskManager,
    RiskSizer,
    TradePlanner,
    TradeStateStore,
)
from scripts.research.semantic_tagger import generate_semantic_tags

try:
    from .signal_deriver import derive_signals
except ImportError:  # pragma: no cover - CLI execution path
    from scripts.research.simulator.signal_deriver import derive_signals

UTC = timezone.utc


@dataclass(frozen=True)
class SimulationParams:
    """Parameter overrides for a simulation run."""

    hazard_multiplier: Optional[float] = 1.0
    hazard_override: Optional[float] = None
    min_repetitions: int = 1
    hold_minutes: int = 30
    exposure_scale: float = 0.02


@dataclass
class GateDecision:
    """Snapshot of the gate + session outcome for a minute."""

    timestamp: datetime
    instrument: str
    admit: bool
    hard_blocked: bool
    reasons: List[str]
    source: str
    metrics: Dict[str, Any]
    direction: Optional[str]
    semantic_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "instrument": self.instrument,
            "admit": bool(self.admit),
            "hard_blocked": bool(self.hard_blocked),
            "reasons": list(self.reasons),
            "source": self.source,
            "direction": self.direction,
            "metrics": dict(self.metrics),
            "semantic_tags": list(self.semantic_tags),
        }


@dataclass
class TradeRecord:
    instrument: str
    entry_time: datetime
    exit_time: datetime
    direction: str
    units: int
    entry_price: float
    exit_price: float
    pnl: float
    commission: float
    mae: float = 0.0
    mfe: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument": self.instrument,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "direction": self.direction,
            "units": int(self.units),
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "pnl": float(self.pnl),
            "commission": float(self.commission),
            "mae": float(self.mae),
            "mfe": float(self.mfe),
        }


@dataclass
class SimulationMetrics:
    pnl: float
    return_pct: float
    sharpe: float
    max_drawdown: float
    trades: int
    win_rate: float
    avg_mae: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pnl": float(self.pnl),
            "return_pct": float(self.return_pct),
            "sharpe": float(self.sharpe),
            "max_dd": float(self.max_drawdown),
            "trades": int(self.trades),
            "win_rate": float(self.win_rate),
            "avg_mae": float(self.avg_mae),
        }


@dataclass
class SimulationResult:
    instrument: str
    params: SimulationParams
    metrics: SimulationMetrics
    trades: List[TradeRecord]
    equity_curve: List[Tuple[datetime, float]]
    decisions: List[GateDecision]
    gate_coverage: Dict[str, Any]
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument": self.instrument,
            "params": dataclasses.asdict(self.params),
            "metrics": self.metrics.to_dict(),
            "trades": [trade.to_dict() for trade in self.trades],
            "equity_curve": [
                {"time": ts.isoformat(), "equity": float(value)}
                for ts, value in self.equity_curve
            ],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "gate_coverage": dict(self.gate_coverage),
            "source": self.source,
        }


@dataclass
class Candle:
    time: datetime
    mid: float


class PositionState:
    __slots__ = ("units", "avg_price", "entry_time", "entry_cost_usd", "max_adverse_usd", "max_favorable_usd")

    def __init__(self) -> None:
        self.units: int = 0
        self.avg_price: float = 0.0
        self.entry_time: Optional[datetime] = None
        self.entry_cost_usd: float = 0.0
        self.max_adverse_usd: float = 0.0
        self.max_favorable_usd: float = 0.0


class PositionTracker:
    """Track open exposure, realized PnL, and executed trades."""

    def __init__(self, *, cost_bps: float) -> None:
        self.realized: float = 0.0
        self.trade_log: List[TradeRecord] = []
        self._states: Dict[str, PositionState] = {}
        self.cost_bps = max(0.0, float(cost_bps))

    def state(self, instrument: str) -> PositionState:
        inst = instrument.upper()
        if inst not in self._states:
            self._states[inst] = PositionState()
        return self._states[inst]

    def has_position(self, instrument: str) -> bool:
        return bool(self.state(instrument.upper()).units)

    def net_units(self, instrument: str) -> int:
        return self.state(instrument.upper()).units

    # ------------------------------------------------------------------
    # PnL handling
    # ------------------------------------------------------------------
    def execute(
        self,
        instrument: str,
        delta_units: int,
        price: float,
        timestamp: datetime,
    ) -> None:
        if not delta_units:
            return
        state = self.state(instrument)
        remaining = int(delta_units)
        inst = instrument.upper()

        while remaining != 0:
            if state.units == 0:
                entry_units = remaining
                units_abs = abs(entry_units)
                cost = self._commission(inst, price, units_abs)
                self.realized -= cost
                state.units = entry_units
                state.avg_price = float(price)
                state.entry_time = timestamp
                state.entry_cost_usd = cost
                state.max_adverse_usd = 0.0
                state.max_favorable_usd = 0.0
                remaining = 0
                continue

            if state.units > 0:
                if remaining > 0:
                    # Scaling long
                    cost = self._commission(inst, price, abs(remaining))
                    self.realized -= cost
                    total_units = state.units + remaining
                    state.avg_price = (
                        state.avg_price * state.units + float(price) * remaining
                    ) / max(1, total_units)
                    state.units = total_units
                    state.entry_cost_usd += cost
                    remaining = 0
                else:
                    close_units = min(abs(remaining), state.units)
                    units_before = max(1, state.units)
                    exit_cost = self._commission(inst, price, close_units)
                    self.realized -= exit_cost
                    gross = (float(price) - state.avg_price) * close_units
                    gross_usd = self._convert_to_usd(inst, gross, price)
                    self.realized += gross_usd

                    entry_cost_share = (
                        state.entry_cost_usd * (close_units / units_before)
                    )
                    state.entry_cost_usd -= entry_cost_share
                    trade_commission = entry_cost_share + exit_cost
                    self.trade_log.append(
                        TradeRecord(
                            instrument=inst,
                            entry_time=state.entry_time or timestamp,
                            exit_time=timestamp,
                            direction="LONG",
                            units=close_units,
                            entry_price=state.avg_price,
                            exit_price=float(price),
                            pnl=gross_usd - trade_commission,
                            commission=trade_commission,
                            mae=abs(state.max_adverse_usd) * (close_units / units_before),
                            mfe=max(0.0, state.max_favorable_usd) * (close_units / units_before),
                        )
                    )
                    state.units -= close_units
                    remaining += close_units  # remaining is negative
                    self._rescale_excursions(state, units_before)
                    if state.units == 0:
                        state.avg_price = 0.0
                        state.entry_time = None
                        state.entry_cost_usd = 0.0
                        state.max_adverse_usd = 0.0
                        state.max_favorable_usd = 0.0

            else:  # state.units < 0 -> short
                if remaining < 0:
                    # Scaling short (selling more)
                    cost = self._commission(inst, price, abs(remaining))
                    self.realized -= cost
                    total_units = state.units + remaining  # both negative
                    weighted = (
                        abs(state.units) * state.avg_price + abs(remaining) * float(price)
                    ) / max(1, abs(total_units))
                    state.avg_price = float(weighted)
                    state.units = total_units
                    state.entry_cost_usd += cost
                    remaining = 0
                else:
                    close_units = min(remaining, abs(state.units))
                    units_before = max(1, abs(state.units))
                    exit_cost = self._commission(inst, price, close_units)
                    self.realized -= exit_cost
                    gross = (state.avg_price - float(price)) * close_units
                    gross_usd = self._convert_to_usd(inst, gross, price)
                    self.realized += gross_usd

                    entry_cost_share = (
                        state.entry_cost_usd * (close_units / units_before)
                    )
                    state.entry_cost_usd -= entry_cost_share
                    trade_commission = entry_cost_share + exit_cost
                    self.trade_log.append(
                        TradeRecord(
                            instrument=inst,
                            entry_time=state.entry_time or timestamp,
                            exit_time=timestamp,
                            direction="SHORT",
                            units=close_units,
                            entry_price=state.avg_price,
                            exit_price=float(price),
                            pnl=gross_usd - trade_commission,
                            commission=trade_commission,
                            mae=abs(state.max_adverse_usd) * (close_units / units_before),
                            mfe=max(0.0, state.max_favorable_usd) * (close_units / units_before),
                        )
                    )
                    state.units += close_units  # moving towards zero
                    remaining -= close_units
                    self._rescale_excursions(state, units_before)
                    if state.units == 0:
                        state.avg_price = 0.0
                        state.entry_time = None
                        state.entry_cost_usd = 0.0
                        state.max_adverse_usd = 0.0
                        state.max_favorable_usd = 0.0

    def mark(self, instrument: str, price: Optional[float]) -> None:
        if price is None:
            return
        state = self.state(instrument)
        if state.units == 0:
            state.max_adverse_usd = 0.0
            state.max_favorable_usd = 0.0
            return
        if state.units > 0:
            gross = (float(price) - state.avg_price) * state.units
        else:
            gross = (state.avg_price - float(price)) * abs(state.units)
        usd = self._convert_to_usd(instrument, gross, price)
        state.max_favorable_usd = max(state.max_favorable_usd, usd)
        state.max_adverse_usd = min(state.max_adverse_usd, usd)

    def _rescale_excursions(self, state: PositionState, prev_units: int) -> None:
        prev_abs = max(1, abs(prev_units))
        remaining = abs(state.units)
        if remaining == 0:
            state.max_adverse_usd = 0.0
            state.max_favorable_usd = 0.0
            return
        scale = remaining / prev_abs
        state.max_adverse_usd *= scale
        state.max_favorable_usd *= scale

    def unrealized(self, instrument: str, price: Optional[float]) -> float:
        if price is None:
            return 0.0
        inst = instrument.upper()
        state = self.state(inst)
        if state.units == 0:
            return 0.0
        if state.units > 0:
            gross = (float(price) - state.avg_price) * state.units
        else:
            gross = (state.avg_price - float(price)) * abs(state.units)
        return self._convert_to_usd(inst, gross, price)

    def _commission(self, instrument: str, price: float, units: int) -> float:
        if self.cost_bps <= 0 or units <= 0:
            return 0.0
        notional = abs(float(price) * units)
        raw = notional * (self.cost_bps / 10_000.0)
        return self._convert_to_usd(instrument, raw, price)

    @staticmethod
    def _convert_to_usd(instrument: str, raw: float, price: float) -> float:
        inst = instrument.upper()
        if inst.endswith("_USD"):
            return raw
        if inst.startswith("USD_"):
            if math.isclose(price, 0.0):
                return 0.0
            return raw / price
        return raw


class BacktestSimulator:
    """Replay gate events across historical candles using live primitives."""

    def __init__(
        self,
        *,
        redis_url: Optional[str],
        granularity: str = "M1",
        profile_path: Optional[Path] = None,
        nav: float = 100_000.0,
        nav_risk_pct: float = 0.01,
        per_position_pct_cap: Optional[float] = None,
        cost_bps: float = 0.0,
        oanda: Optional[OandaConnector] = None,
        semantic_tagger: Callable[[Mapping[str, Any]], Sequence[str]] | None = None,
    ) -> None:
        self.redis_url = redis_url
        self.granularity = granularity
        self.nav = float(nav)
        self.nav_risk_pct = float(nav_risk_pct)
        self.per_position_pct_cap = per_position_pct_cap if per_position_pct_cap is not None else float(nav_risk_pct)
        self.cost_bps = float(cost_bps)
        self.connector = oanda or OandaConnector(read_only=True)
        self.profile = StrategyProfile.load(profile_path or Path("config/echo_strategy.yaml"))
        self.gate_reader = GateReader(redis_url)
        self.trade_state_store = TradeStateStore()
        self.trade_planner = TradePlanner(self.trade_state_store)
        self.risk_manager = RiskManager(RiskLimits())
        self.risk_manager.set_nav(self.nav)
        self.risk_sizer = RiskSizer(
            self.risk_manager,
            nav_risk_pct=self.nav_risk_pct,
            per_position_pct_cap=self.per_position_pct_cap,
            alloc_top_k=1,
        )
        self._redis_client = redis.from_url(redis_url) if redis_url and redis else None
        self._position_tracker = PositionTracker(cost_bps=self.cost_bps)
        self.semantic_tagger = semantic_tagger or generate_semantic_tags
        self._reset_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def simulate(
        self,
        instrument: str,
        *,
        start: datetime,
        end: datetime,
        params: Optional[SimulationParams] = None,
        profile_override: Optional[StrategyProfile] = None,
        semantic_filter: Optional[Sequence[str]] = None,
        semantic_overrides: Optional[Mapping[str, float]] = None,
    ) -> Optional[SimulationResult]:
        params = params or SimulationParams()
        profile = profile_override or copy.deepcopy(self.profile)
        inst_profile = profile.get(instrument)
        if params.hazard_override is not None:
            inst_profile.hazard_max = float(params.hazard_override)
        elif params.hazard_multiplier is not None and inst_profile.hazard_max is not None:
            inst_profile.hazard_max = (inst_profile.hazard_max or 0.0) * params.hazard_multiplier
        inst_profile.min_repetitions = max(1, params.min_repetitions)
        if semantic_filter is None and getattr(inst_profile, "semantic_filter", None):
            semantic_filter = list(getattr(inst_profile, "semantic_filter"))

        candles = self._load_candles(instrument, start, end)
        if not candles:
            return None
        gates = self._load_gate_events(instrument, start, end)
        source = "valkey" if gates else "synthetic"
        if not gates:
            gates = derive_signals(
                instrument,
                start=start,
                end=end,
                candles=candles,
                profile=inst_profile,
            )
        if not gates:
            return None

        decisions, equity_curve = self._replay(
            instrument,
            candles,
            gates,
            inst_profile,
            params=params,
            required_tags=semantic_filter,
            semantic_overrides=semantic_overrides,
        )
        tracker = self._position_tracker
        metrics = self._compute_metrics(equity_curve, tracker.trade_log)
        gate_coverage = self._coverage(decisions)
        result = SimulationResult(
            instrument=instrument.upper(),
            params=params,
            metrics=metrics,
            trades=list(tracker.trade_log),
            equity_curve=equity_curve,
            decisions=decisions,
            gate_coverage=gate_coverage,
            source="valkey" if gate_coverage.get("real_minutes") else source,
        )
        self._reset_state()
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        self.trade_state_store = TradeStateStore()
        self.trade_planner = TradePlanner(self.trade_state_store)
        self.risk_manager = RiskManager(RiskLimits())
        self.risk_manager.set_nav(self.nav)
        self.risk_sizer = RiskSizer(
            self.risk_manager,
            nav_risk_pct=self.nav_risk_pct,
            per_position_pct_cap=self.per_position_pct_cap,
            alloc_top_k=1,
        )
        self._position_tracker = PositionTracker(cost_bps=self.cost_bps)

    def _load_candles(self, instrument: str, start: datetime, end: datetime) -> List[Candle]:
        payload: List[Dict[str, Any]] = []
        granularity = self.granularity.upper()
        if chunked_fetch is not None:
            seconds = GRANULARITY_SECONDS.get(granularity, 60)
            # Aim for ~4500 samples per request to stay within OANDA's 5000 cap.
            step_seconds = max(seconds * 3000, 3600)
            step = timedelta(seconds=step_seconds)
            try:
                payload = chunked_fetch(
                    self.connector,
                    instrument,
                    granularity,
                    start,
                    end,
                    step,
                )
            except Exception:
                payload = []
        if not payload:
            payload = self.connector.get_candles(
                instrument,
                granularity=granularity,
                from_time=start.isoformat().replace("+00:00", "Z"),
                to_time=end.isoformat().replace("+00:00", "Z"),
            )
        if not payload:
            approx = int(max(1, min(5000, ((end - start).total_seconds() / 60) + 10)))
            payload = self.connector.get_candles(
                instrument,
                granularity=granularity,
                count=approx,
            )
        candles: List[Candle] = []
        for row in payload or []:
            mid = row.get("mid") or {}
            price = mid.get("c") or mid.get("close")
            if price is None:
                continue
            try:
                value = float(price)
            except Exception:
                continue
            ts_raw = row.get("time")
            if not ts_raw:
                continue
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(UTC)
            except Exception:
                continue
            if ts < start or ts > end:
                continue
            candles.append(Candle(time=ts, mid=value))
        candles.sort(key=lambda item: item.time)
        self._reset_state()
        return candles

    def _load_gate_events(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, Any]]:
        client = self._redis_client
        if client is None:
            return []
        key = f"gate:index:{instrument.upper()}"
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        entries: List[Dict[str, Any]] = []
        try:
            raw_entries = client.zrangebyscore(key, start_ms, end_ms, withscores=True)
        except Exception:
            raw_entries = []
        for raw, score in raw_entries:
            try:
                payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            except Exception:
                continue
            payload["ts_ms"] = int(score)
            payload.setdefault("source", "valkey")
            entries.append(payload)
        if not entries:
            try:
                raw_last = client.get(f"gate:last:{instrument.upper()}")
            except Exception:
                raw_last = None
            if raw_last:
                try:
                    payload = json.loads(raw_last if isinstance(raw_last, str) else raw_last.decode("utf-8"))
                    payload.setdefault("ts_ms", start_ms)
                    payload.setdefault("source", "valkey")
                    entries.append(payload)
                except Exception:
                    pass
        entries.sort(key=lambda row: row.get("ts_ms", 0))
        return entries

    def _replay(
        self,
        instrument: str,
        candles: Sequence[Candle],
        gate_events: Sequence[Dict[str, Any]],
        profile: StrategyInstrument,
        *,
        params: SimulationParams,
        required_tags: Optional[Sequence[str]] = None,
        semantic_overrides: Optional[Mapping[str, float]] = None,
    ) -> Tuple[List[GateDecision], List[Tuple[datetime, float]]]:
        tracker = PositionTracker(cost_bps=self.cost_bps)
        self._position_tracker = tracker
        sessions: Dict[str, Any] = {}
        if profile.session is not None:
            sessions[profile.symbol.upper()] = profile.session
        session_policy = SessionPolicy(sessions, exit_buffer_minutes=5)
        gate_events_sorted = sorted(gate_events, key=lambda row: row.get("ts_ms", 0))
        gate_index = 0
        current_gate: Optional[Dict[str, Any]] = None
        decisions: List[GateDecision] = []
        equity_curve: List[Tuple[datetime, float]] = []
        last_mid: Optional[float] = None
        required_tag_set = {
            tag.strip().lower()
            for tag in (required_tags or [])
            if isinstance(tag, str) and tag.strip()
        }
        semantic_tagger = self.semantic_tagger

        for candle in candles:
            while gate_index < len(gate_events_sorted) and gate_events_sorted[gate_index].get("ts_ms", 0) <= int(candle.time.timestamp() * 1000):
                current_gate = gate_events_sorted[gate_index]
                gate_index += 1

            tracker.mark(instrument, candle.mid)

            session = session_policy.evaluate(
                profile.symbol,
                candle.time,
                tracker.has_position(instrument.upper()),
            )
            gate_payload = dict(current_gate or {})
            gate_payload.setdefault("instrument", profile.symbol)
            gate_payload.setdefault("components", {})
            gate_payload.setdefault("source", "synthetic" if not current_gate else current_gate.get("source", "valkey"))

            semantic_tags: List[str] = []
            if semantic_tagger is not None or required_tag_set:
                tagger_callable = semantic_tagger or generate_semantic_tags
                if semantic_overrides is not None:
                    try:
                        candidate_tags = tagger_callable(
                            gate_payload,
                            overrides=semantic_overrides,
                        )
                    except TypeError:
                        candidate_tags = tagger_callable(gate_payload)
                else:
                    candidate_tags = tagger_callable(gate_payload)
                if candidate_tags:
                    seen: Dict[str, None] = {}
                    for tag in candidate_tags:
                        if not isinstance(tag, str):
                            continue
                        normalized = tag.strip()
                        if not normalized:
                            continue
                        if normalized.lower() in seen:
                            continue
                        seen[normalized.lower()] = None
                        semantic_tags.append(normalized)
            semantic_tag_set = {tag.lower() for tag in semantic_tags}

            admitted, gate_reasons = gate_evaluation(gate_payload, profile)
            reasons = list(gate_payload.get("reasons", []))
            reasons.extend(reason for reason in gate_reasons if reason not in reasons)
            hard_blocks: List[str] = []

            if not current_gate:
                reasons.append("missing_gate")
                hard_blocks.append("missing_gate")

            if not session.tradable:
                hard_blocks.append(session.reason)
                if session.reason not in reasons:
                    reasons.append(session.reason)

            if gate_payload.get("source") == "synthetic" and admitted and not reasons:
                reasons.append("synthetic:thresholds")

            if required_tag_set:
                missing = [tag for tag in required_tag_set if tag not in semantic_tag_set]
                if missing:
                    hard_blocks.append("semantic_filter")
                    reasons.append(
                        "semantic_filter_missing:" + ",".join(sorted(missing))
                    )

            direction = str(gate_payload.get("direction") or "").upper() or None
            if direction not in {"BUY", "SELL"}:
                direction = "BUY" if (last_mid is None or candle.mid >= last_mid) else "SELL"
            last_mid = candle.mid

            gate_entry_ready = admitted and not hard_blocks
            requested_side = 0
            if direction == "BUY":
                requested_side = 1
            elif direction == "SELL":
                requested_side = -1

            per_trade_caps = self.risk_sizer.compute_caps(self.nav)
            target_units_abs = 0
            if gate_entry_ready and requested_side != 0:
                try:
                    units, _, _ = self.risk_sizer.target_units(
                        instrument,
                        target_exposure=per_trade_caps.per_position_cap,
                        exposure_scale=params.exposure_scale,
                        price_data={"mid": candle.mid},
                        auxiliary_prices=None,
                    )
                except Exception:
                    units = 0
                target_units_abs = abs(units)

            current_units = tracker.net_units(instrument)
            outcome = self.trade_planner.plan_allocation(
                instrument,
                now_ts=candle.time.timestamp(),
                current_units=current_units,
                gate_entry_ready=gate_entry_ready,
                gate_reasons=list(reasons),
                direction=direction,
                requested_side=requested_side,
                scaled_units_abs=int(target_units_abs),
                hold_secs=max(60, params.hold_minutes * 60),
                max_hold_limit=None,
                hold_rearm_enabled=True,
                signal_key=str(gate_payload.get("signal_key") or ""),
                hard_blocks=list(hard_blocks),
            )

            delta_units = outcome.target_units - current_units
            if delta_units:
                tracker.execute(instrument, delta_units, candle.mid, candle.time)
                self.risk_manager.record_fill(instrument, delta_units, candle.mid)

            unrealized = tracker.unrealized(instrument, candle.mid)
            equity = self.nav + tracker.realized + unrealized
            equity_curve.append((candle.time, equity))

            decisions.append(
                GateDecision(
                    timestamp=candle.time,
                    instrument=instrument.upper(),
                    admit=gate_entry_ready,
                    hard_blocked=bool(hard_blocks),
                    reasons=reasons,
                    source=gate_payload.get("source", "unknown"),
                    metrics={
                        "lambda": gate_payload.get("lambda"),
                        "repetitions": gate_payload.get("repetitions"),
                        "components": gate_payload.get("components", {}),
                        "repetition_count": gate_payload.get("repetitions"),
                        "hazard_threshold": (
                            (gate_payload.get("structure") or {}).get("hazard_threshold")
                            or gate_payload.get("hazard_threshold")
                        ),
                        "regime": gate_payload.get("regime"),
                    },
                    direction=direction,
                    semantic_tags=semantic_tags,
                )
            )

        return decisions, equity_curve

    def _compute_metrics(
        self,
        equity_curve: Sequence[Tuple[datetime, float]],
        trades: Sequence[TradeRecord],
    ) -> SimulationMetrics:
        total_pnl = 0.0
        if equity_curve:
            total_pnl = equity_curve[-1][1] - self.nav
        return_pct = (total_pnl / self.nav) if self.nav else 0.0
        sharpe = self._compute_sharpe(equity_curve)
        max_dd = self._compute_drawdown(equity_curve)
        wins = sum(1 for trade in trades if trade.pnl > 0)
        num_trades = len(trades)
        win_rate = (wins / num_trades) if num_trades else 0.0
        avg_mae = statistics.mean(abs(trade.mae) for trade in trades) if trades else 0.0
        return SimulationMetrics(
            pnl=total_pnl,
            return_pct=return_pct,
            sharpe=sharpe,
            max_drawdown=max_dd,
            trades=num_trades,
            win_rate=win_rate,
            avg_mae=avg_mae,
        )

    @staticmethod
    def _compute_sharpe(equity_curve: Sequence[Tuple[datetime, float]]) -> float:
        if len(equity_curve) < 2:
            return 0.0
        daily_equity: Dict[date, float] = {}
        start_equity = equity_curve[0][1]
        for ts, value in equity_curve:
            daily_equity[ts.date()] = value
        returns: List[float] = []
        prev = start_equity
        for day in sorted(daily_equity.keys()):
            value = daily_equity[day]
            if math.isclose(prev, 0.0):
                prev = value
                continue
            returns.append((value - prev) / prev)
            prev = value
        if not returns:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(variance)
        if math.isclose(std, 0.0):
            return 0.0
        return (mean / std) * math.sqrt(252)

    @staticmethod
    def _compute_drawdown(equity_curve: Sequence[Tuple[datetime, float]]) -> float:
        peak = -float("inf")
        max_dd = 0.0
        for _, equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown
        return max_dd

    @staticmethod
    def _coverage(decisions: Sequence[GateDecision]) -> Dict[str, Any]:
        total = len(decisions)
        real = sum(1 for decision in decisions if decision.source == "valkey")
        synthetic = sum(1 for decision in decisions if decision.source == "synthetic")
        regime_counts: Dict[str, Dict[str, int]] = {}
        admitted_total = 0
        for decision in decisions:
            regime_payload = decision.metrics.get("regime") if isinstance(decision.metrics, Mapping) else None
            label = "unknown"
            if isinstance(regime_payload, Mapping):
                raw_label = regime_payload.get("label")
                if isinstance(raw_label, str) and raw_label.strip():
                    label = raw_label.strip().lower()
            bucket = regime_counts.setdefault(label, {"count": 0, "admitted": 0})
            bucket["count"] += 1
            if decision.admit:
                bucket["admitted"] += 1
                admitted_total += 1
        return {
            "minutes": total,
            "real_minutes": real,
            "synthetic_minutes": synthetic,
            "real_fraction": (real / total) if total else 0.0,
            "regimes": regime_counts,
            "regime_hit_rate": (admitted_total / total) if total else 0.0,
        }


__all__ = [
    "BacktestSimulator",
    "SimulationParams",
    "SimulationResult",
    "GateDecision",
    "TradeRecord",
]


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a backtest simulation for a single instrument")
    parser.add_argument("--instrument", required=True, help="Instrument symbol, e.g. USD_JPY")
    parser.add_argument("--start", required=True, help="ISO8601 start timestamp (UTC)")
    parser.add_argument("--end", required=True, help="ISO8601 end timestamp (UTC)")
    parser.add_argument("--balance", type=float, default=100_000.0, help="Starting NAV")
    parser.add_argument("--cost-bps", type=float, default=1.5, help="Commission/slippage in basis points")
    parser.add_argument("--redis", default=os.getenv("VALKEY_URL") or "redis://localhost:6379/0", help="Valkey URL")
    parser.add_argument("--granularity", default="M1", help="Candle granularity (default M1)")
    parser.add_argument("--nav-risk-pct", type=float, default=0.01, help="NAV risk percent")
    parser.add_argument("--profile", default="config/echo_strategy.yaml", help="Strategy profile path")
    parser.add_argument("--hazard", type=float, help="Override hazard cap value")
    parser.add_argument("--hazard-multiplier", type=float, help="Multiplier applied to profile hazard cap")
    parser.add_argument("--min-reps", type=int, help="Override minimum repetitions")
    parser.add_argument("--hold-minutes", type=int, help="Hold horizon in minutes")
    parser.add_argument("--exposure", type=float, default=0.02, help="Exposure scale")
    return parser.parse_args()


def main() -> int:
    args = _parse_cli_args()
    simulator = BacktestSimulator(
        redis_url=args.redis,
        granularity=args.granularity,
        profile_path=Path(args.profile),
        nav=args.balance,
        nav_risk_pct=args.nav_risk_pct,
        cost_bps=args.cost_bps,
    )
    params = SimulationParams(
        hazard_override=args.hazard,
        hazard_multiplier=args.hazard_multiplier if args.hazard_multiplier is not None else 1.0,
        min_repetitions=args.min_reps or 1,
        hold_minutes=args.hold_minutes or 30,
        exposure_scale=args.exposure,
    )
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(UTC)
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(UTC)
    result = simulator.simulate(args.instrument, start=start, end=end, params=params)
    if result is None:
        print(json.dumps({"error": "no_result"}, indent=2))
        return 1
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    raise SystemExit(main())
