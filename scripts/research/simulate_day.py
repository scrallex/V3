#!/usr/bin/env python3
"""Baseline gate playback to evaluate admit frequency and session filtering."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from scripts.trading.portfolio_manager import (
    StrategyProfile,
    SessionPolicy,
    gate_is_admitted,
)
from scripts.trading.risk_planner import (
    RiskLimits,
    RiskManager,
    RiskSizer,
    TradePlanner,
    TradeStateStore,
)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [dict(row) for row in reader]
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline gate playback simulator")
    parser.add_argument("input", help="Gate export file (CSV or JSONL)")
    parser.add_argument("--profile", default="config/echo_strategy.yaml", help="Strategy profile path")
    parser.add_argument("--nav", type=float, default=10000.0, help="Starting NAV for sizing")
    parser.add_argument("--exposure-scale", type=float, default=0.02, help="Exposure scale (margin per unit)")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    if not rows:
        print("No rows loaded")
        return 1

    profile = StrategyProfile.load(Path(args.profile))
    sessions = {symbol: inst.session for symbol, inst in profile.instruments.items() if inst.session is not None}
    session_policy = SessionPolicy(sessions, exit_buffer_minutes=int(profile.global_defaults.get("session_exit_minutes", 5) or 5))

    risk = RiskManager(RiskLimits())
    sizer = RiskSizer(risk, nav_risk_pct=0.01, per_position_pct_cap=0.01, alloc_top_k=3)
    trade_state = TradeStateStore()
    trade_planner = TradePlanner(trade_state)

    admits_total = 0
    admitted_by_instrument: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()

    # Sort by timestamp if available
    def sort_key(row: Dict[str, Any]) -> float:
        return maybe_float(row.get("ts_ms")) or maybe_float(row.get("ts")) or 0.0

    rows.sort(key=sort_key)

    for row in rows:
        instrument = str(row.get("instrument") or row.get("__key", "").split(":")[-1]).upper()
        gate_info = row
        profile_inst = profile.get(instrument)
        admitted = gate_is_admitted(gate_info, profile_inst)
        ts = sort_key(row) / 1000.0
        now = None
        if ts > 0:
            from datetime import datetime, timezone

            now = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)

        decision = session_policy.evaluate(instrument, now, trade_state.has_trades(instrument))
        if not admitted:
            rejected_reasons["gate_block"] += 1
            continue
        if not decision.tradable:
            rejected_reasons[decision.reason] += 1
            continue
        admits_total += 1
        admitted_by_instrument[instrument] += 1

    print(f"Processed {len(rows)} gate records")
    print(f"Total admits: {admits_total}")
    for inst, count in admitted_by_instrument.items():
        print(f"  {inst}: {count}")
    if rejected_reasons:
        print("Rejected events:")
        for reason, count in rejected_reasons.most_common():
            print(f"  {reason}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
