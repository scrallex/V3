#!/usr/bin/env python3
"""Risk sizing and trade planning primitives for the lean SEP stack."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =============================================================================
# Risk management
# =============================================================================


@dataclass
class RiskLimits:
    """Configuration for core portfolio guardrails."""

    max_position_size: float = 0.0
    max_total_exposure: float = 0.0
    max_positions_per_pair: int = 1
    max_total_positions: int = 4
    max_net_units_per_pair: int = 100_000
    max_total_units: Optional[int] = None


@dataclass
class RiskSizerCaps:
    """Margin caps returned by :class:`RiskSizer`."""

    nav_risk_cap: float
    per_position_cap: float
    portfolio_cap: float


class RiskManager:
    """Minimal inventory of open exposure used by the portfolio manager."""

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self._positions: Dict[str, int] = {}
        self._exposure: Dict[str, float] = {}
        self._nav_snapshot = 0.0
        self._last_updated = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # NAV helpers
    # ------------------------------------------------------------------
    def set_nav(self, nav: float) -> None:
        self._nav_snapshot = max(0.0, float(nav or 0.0))
        self._last_updated = datetime.now(timezone.utc)

    def record_fill(
        self, instrument: str, units: int, price: Optional[float] = None
    ) -> None:
        inst = instrument.upper()
        current = self._positions.get(inst, 0)
        new_units = current + int(units)
        if new_units == 0:
            self._positions.pop(inst, None)
            self._exposure.pop(inst, None)
        else:
            self._positions[inst] = new_units
            notional = self._usd_notional(inst, new_units, price)
            if notional is not None:
                self._exposure[inst] = notional
        self._last_updated = datetime.now(timezone.utc)

    def flatten(self, instrument: str) -> None:
        inst = instrument.upper()
        self._positions.pop(inst, None)
        self._exposure.pop(inst, None)
        self._last_updated = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def net_units(self, instrument: str) -> int:
        return self._positions.get(instrument.upper(), 0)

    def positions(self) -> Dict[str, int]:
        return dict(self._positions)

    def total_units(self) -> int:
        return sum(abs(units) for units in self._positions.values())

    def exposure(self) -> float:
        return sum(self._exposure.values())

    def position_breakdown(self) -> List[Dict[str, object]]:
        """Return a per-instrument snapshot of units and estimated exposure."""

        return [
            {
                "instrument": instrument,
                "net_units": int(units),
                "exposure": float(self._exposure.get(instrument, 0.0)),
            }
            for instrument, units in self._positions.items()
        ]

    def get_risk_summary(self) -> Dict[str, float]:
        return {
            "nav_snapshot": self._nav_snapshot,
            "total_units": float(self.total_units()),
            "exposure_usd": float(self.exposure()),
            "last_updated": self._last_updated.timestamp(),
        }

    # ------------------------------------------------------------------
    # Guardrail checks
    # ------------------------------------------------------------------
    def can_add(self, instrument: str, planned_units: int) -> bool:
        proposed = self.net_units(instrument) + int(planned_units)
        if (
            self.limits.max_net_units_per_pair
            and abs(proposed) > self.limits.max_net_units_per_pair
        ):
            return False
        if (
            self.limits.max_total_units is not None
            and (self.total_units() + abs(planned_units)) > self.limits.max_total_units
        ):
            return False
        if (
            proposed != 0
            and instrument.upper() not in self._positions
            and self.limits.max_positions_per_pair <= 0
        ):
            return False
        if (
            proposed != 0
            and instrument.upper() not in self._positions
            and len(self._positions) >= self.limits.max_total_positions
        ):
            return False
        if (
            self.limits.max_position_size > 0
            and self.exposure() >= self.limits.max_position_size
        ):
            return False
        return True

    def _usd_notional(
        self, instrument: str, net_units: int, price: Optional[float]
    ) -> Optional[float]:
        """Return the absolute USD notional for the given exposure, if derivable."""

        units = abs(int(net_units))
        if units == 0:
            return 0.0
        parts = instrument.split("_", 1)
        base = parts[0].upper() if parts else ""
        quote = parts[1].upper() if len(parts) > 1 else ""
        if base == "USD":
            return float(units)
        if quote == "USD":
            if price is None:
                return None
            return float(units) * float(price)
        if price is None:
            return None
        # Fall back to quote-denominated notional when we cannot normalise to USD.
        return float(units) * float(price)


class RiskSizer:
    """Convert target exposure into units under simple leverage assumptions."""

    def __init__(
        self,
        risk_manager: RiskManager,
        *,
        nav_risk_pct: float,
        per_position_pct_cap: float,
        alloc_top_k: int,
    ) -> None:
        self._risk_manager = risk_manager
        self._nav_risk_pct = max(0.0, min(1.0, float(nav_risk_pct)))
        self._per_position_pct_cap = max(0.0, min(1.0, float(per_position_pct_cap)))
        self._alloc_top_k = max(1, int(alloc_top_k))

    def compute_caps(self, nav_snapshot: float) -> RiskSizerCaps:
        nav = max(0.0, float(nav_snapshot or 0.0))
        nav_risk_cap = nav * self._nav_risk_pct
        per_position_cap = nav_risk_cap
        if self._per_position_pct_cap > 0:
            per_position_cap = min(per_position_cap, nav * self._per_position_pct_cap)
        portfolio_cap = nav_risk_cap * self._alloc_top_k
        return RiskSizerCaps(
            nav_risk_cap=nav_risk_cap,
            per_position_cap=per_position_cap,
            portfolio_cap=portfolio_cap,
        )

    def target_units(
        self,
        instrument: str,
        *,
        target_exposure: float,
        exposure_scale: float,
        price_data: Dict[str, float],
        auxiliary_prices: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[int, float, float]:
        live_price = float(price_data.get("mid", 0.0)) if price_data else 0.0
        margin_per_unit = self._margin_per_unit(
            instrument,
            live_price,
            float(exposure_scale or 0.0),
            auxiliary_prices or {},
        )
        if margin_per_unit <= 0:
            return 0, 0.0, 0.0

        adjusted_exposure = max(0.0, float(target_exposure or 0.0))

        # ------------------------------------------------------------------
        # V3 Risk Check: Margin Cap (Solvency Guard)
        # Broker Leverage: 50:1
        # Safety Limit: 50% of NAV used for Margin
        # Max Allowed Exposure = NAV * 50 * 0.5 = NAV * 25.0
        # ------------------------------------------------------------------
        nav_curr = self._risk_manager._nav_snapshot
        max_allowed_exposure = nav_curr * 25.0
        current_total_exposure = self._risk_manager.exposure()

        # If adding this exposure breaches the limit, clamp it
        # Note: We assume 'target_exposure' is additive. If we are reducing, logic handles it naturally (target < current notional in diff contexts),
        # but here 'target_units' implies a NEW sizing request.
        if current_total_exposure + adjusted_exposure > max_allowed_exposure:
            available_exposure = max(0.0, max_allowed_exposure - current_total_exposure)
            adjusted_exposure = min(adjusted_exposure, available_exposure)

        raw_units = adjusted_exposure / margin_per_unit
        units = (
            int(raw_units) if raw_units >= 1 else (1 if adjusted_exposure > 0 else 0)
        )
        return units, margin_per_unit, adjusted_exposure

    def _margin_per_unit(
        self,
        instrument: str,
        live_price: float,
        exposure_scale: float,
        auxiliary_prices: Dict[str, Dict[str, float]],
    ) -> float:
        if exposure_scale <= 0:
            return 0.0
        base, quote = (instrument.upper().split("_", 1) + [""])[:2]
        if live_price <= 0:
            live_price = 0.0
        if base == "USD":
            return exposure_scale
        if quote == "USD":
            return live_price * exposure_scale if live_price > 0 else 0.0
        bpair = f"{base}_USD"
        if auxiliary_prices.get(bpair, {}).get("mid"):
            return float(auxiliary_prices[bpair]["mid"]) * exposure_scale
        qpair = f"USD_{quote}"
        if auxiliary_prices.get(qpair, {}).get("mid") and live_price > 0:
            return float(auxiliary_prices[qpair]["mid"]) * live_price * exposure_scale
        return live_price * exposure_scale if live_price > 0 else 0.0


# =============================================================================
# Trade planner
# =============================================================================


@dataclass
class ActiveTrade:
    direction: int  # +1 long, -1 short
    units: int
    entry_ts: float
    hold_until: float
    max_hold_until: Optional[float]

    def net_units(self) -> int:
        return self.direction * self.units

    def extend_hold(self, new_hold_until: float, max_limit: Optional[float]) -> bool:
        if max_limit is not None and new_hold_until > max_limit:
            new_hold_until = max_limit
        if new_hold_until > self.hold_until:
            self.hold_until = new_hold_until
            return True
        return False


class TradeStateStore:
    """In-memory trade stack with optional persistence helpers."""

    def __init__(self) -> None:
        self._trades: Dict[str, List[ActiveTrade]] = {}
        self._pending_close: set[str] = set()
        self._last_entry_signal: Dict[str, str] = {}

    def get_trades(self, instrument: str) -> List[ActiveTrade]:
        trades = self._trades.get(instrument.upper(), [])
        return [
            ActiveTrade(
                t.direction, t.units, t.entry_ts, t.hold_until, t.max_hold_until
            )
            for t in trades
        ]

    def replace_trades(self, instrument: str, trades: List[ActiveTrade]) -> None:
        inst = instrument.upper()
        if trades:
            self._trades[inst] = trades
        else:
            self._trades.pop(inst, None)

    def remove_trades(self, instrument: str) -> None:
        inst = instrument.upper()
        self._trades.pop(inst, None)
        self._pending_close.discard(inst)
        self._last_entry_signal.pop(inst, None)

    def has_trades(self, instrument: str) -> bool:
        return bool(self._trades.get(instrument.upper()))

    def mark_pending_close(self, instrument: str) -> None:
        self._pending_close.add(instrument.upper())

    def clear_pending_close(self, instrument: str) -> None:
        self._pending_close.discard(instrument.upper())

    def is_pending_close(self, instrument: str) -> bool:
        return instrument.upper() in self._pending_close

    def set_last_signal(self, instrument: str, signal_key: str) -> None:
        if signal_key:
            self._last_entry_signal[instrument.upper()] = signal_key

    def clear_last_signal(self, instrument: str) -> None:
        self._last_entry_signal.pop(instrument.upper(), None)


@dataclass
class TradePlanOutcome:
    target_units: int
    gate_entry_ready: bool
    gate_reasons: List[str]
    direction: Optional[str]
    requested_side: int
    state_changed: bool


class TradePlanner:
    """Build trade stacks and net target units for an instrument."""

    def __init__(self, store: TradeStateStore) -> None:
        self._store = store

    def plan_allocation(
        self,
        instrument: str,
        *,
        now_ts: float,
        current_units: int,
        gate_entry_ready: bool,
        gate_reasons: List[str],
        direction: Optional[str],
        requested_side: int,
        scaled_units_abs: int,
        hold_secs: int,
        max_hold_limit: Optional[float],
        hold_rearm_enabled: bool,
        signal_key: str,
        hard_blocks: List[str],
    ) -> TradePlanOutcome:
        inst = instrument.upper()
        trades = self._store.get_trades(inst)
        pending = self._store.is_pending_close(inst)
        state_changed = False

        net_from_trades = sum(trade.net_units() for trade in trades)
        if pending and net_from_trades == current_units:
            self._store.clear_pending_close(inst)
            pending = False
            state_changed = True

        if not pending and net_from_trades != current_units:
            if trades:
                trades = []
                state_changed = True
            if current_units != 0:
                hold_until = now_ts + hold_secs if hold_secs > 0 else now_ts
                trades.append(
                    ActiveTrade(
                        direction=1 if current_units > 0 else -1,
                        units=abs(current_units),
                        entry_ts=now_ts,
                        hold_until=hold_until,
                        max_hold_until=max_hold_limit,
                    )
                )
                state_changed = True

        if hard_blocks:
            if trades:
                self._store.mark_pending_close(inst)
                self._store.clear_last_signal(inst)
            self._store.remove_trades(inst)
            return TradePlanOutcome(
                target_units=0,
                gate_entry_ready=False,
                gate_reasons=gate_reasons + hard_blocks,
                direction=direction,
                requested_side=requested_side,
                state_changed=bool(trades) or state_changed,
            )

        net_from_trades = sum(trade.net_units() for trade in trades)

        if not gate_entry_ready or requested_side == 0:
            target_units = net_from_trades
            if trades:
                self._store.replace_trades(inst, trades)
            else:
                self._store.remove_trades(inst)
            return TradePlanOutcome(
                target_units=target_units,
                gate_entry_ready=gate_entry_ready,
                gate_reasons=gate_reasons,
                direction=direction,
                requested_side=requested_side,
                state_changed=state_changed,
            )

        current_side = 1 if net_from_trades > 0 else (-1 if net_from_trades < 0 else 0)
        if current_side and requested_side and requested_side != current_side:
            gate_entry_ready = False
            gate_reasons.append("opposite_side_blocked")

        last_signal = self._store._last_entry_signal.get(inst)
        if gate_entry_ready and scaled_units_abs > 0 and requested_side:
            if signal_key and signal_key == last_signal:
                gate_entry_ready = False
                gate_reasons.append("duplicate_signal")
            else:
                trade_units = int(scaled_units_abs)
                hold_until = now_ts + hold_secs if hold_secs > 0 else now_ts
                trades.append(
                    ActiveTrade(
                        direction=requested_side,
                        units=trade_units,
                        entry_ts=now_ts,
                        hold_until=hold_until,
                        max_hold_until=max_hold_limit,
                    )
                )
                self._store.set_last_signal(inst, signal_key)
                state_changed = True

        if trades:
            remaining: List[ActiveTrade] = []
            for trade in trades:
                max_limit = trade.max_hold_until
                if (
                    max_limit is not None and now_ts >= max_limit
                ) or now_ts >= trade.hold_until:
                    continue
                remaining.append(trade)
            if len(remaining) != len(trades):
                self._store.mark_pending_close(inst)
                state_changed = True
            trades = remaining
            net_from_trades = sum(trade.net_units() for trade in trades)

        if trades:
            self._store.replace_trades(inst, trades)
        else:
            self._store.remove_trades(inst)

        return TradePlanOutcome(
            target_units=net_from_trades,
            gate_entry_ready=gate_entry_ready,
            gate_reasons=gate_reasons,
            direction=direction,
            requested_side=requested_side,
            state_changed=state_changed,
        )


__all__ = [
    "ActiveTrade",
    "RiskLimits",
    "RiskManager",
    "RiskSizer",
    "RiskSizerCaps",
    "TradePlanOutcome",
    "TradePlanner",
    "TradeStateStore",
]
