#!/usr/bin/env python3
"""Backtest grid orchestration using the live-aligned simulator."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import pathlib
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.simulator.backtest_simulator import (
    BacktestSimulator,
    SimulationParams,
    SimulationResult,
)


UTC = timezone.utc


@dataclass(frozen=True)
class GridParams:
    hazard: float
    min_repetitions: int
    hold_minutes: int
    exposure: float


@dataclass
class GridRow:
    instrument: str
    params: GridParams
    source: str
    metrics: Dict[str, Any]
    qualified: bool
    insufficient_data: bool
    detail: Optional[Dict[str, Any]] = None
    score: float = 0.0
    penalties: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "instrument": self.instrument,
            "params": {
                "hazard_x": self.params.hazard,
                "min_repetitions": self.params.min_repetitions,
                "hold_minutes": self.params.hold_minutes,
                "exposure": self.params.exposure,
            },
            "metrics": dict(self.metrics),
            "source": self.source,
            "qualified": self.qualified,
            "insufficient_data": self.insufficient_data,
        }
        payload["score"] = self.score
        if self.penalties:
            payload["penalties"] = list(self.penalties)
        if self.detail is not None:
            payload["detail"] = dict(self.detail)
        return payload


class BacktestRunner:
    """Coordinate grid evaluations across instruments using the simulator."""

    def __init__(
        self,
        *,
        instruments: Sequence[str],
        redis_url: str,
        granularity: str = "M1",
        profile_path: pathlib.Path | None = None,
        nav: float = 100_000.0,
        nav_risk_pct: float = 0.01,
        cost_bps: float = 1.5,
        output_path: pathlib.Path | None = None,
        semantic_filters: Mapping[str, Sequence[str]] | None = None,
        semantic_overrides: Mapping[str, float] | None = None,
    ) -> None:
        self.instruments = [inst.upper() for inst in instruments]
        self.output_path = output_path or pathlib.Path("output/backtests/latest.json")
        self.partial_path = self.output_path.with_suffix(".partial.json")
        self.error_path = self.output_path.with_suffix(".error.json")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        profile_file = profile_path or pathlib.Path("config/echo_strategy.yaml")
        self.simulator = BacktestSimulator(
            redis_url=redis_url,
            granularity=granularity,
            profile_path=profile_file,
            nav=nav,
            nav_risk_pct=nav_risk_pct,
            per_position_pct_cap=nav_risk_pct,
            cost_bps=cost_bps,
        )
        self.semantic_filters = {
            key.upper(): list(value)
            for key, value in (semantic_filters or {}).items()
            if isinstance(value, Sequence)
        }
        defaults = {
            inst: list(getattr(self.simulator.profile.get(inst), "semantic_filter", []))
            for inst in self.instruments
        }
        for inst, default in defaults.items():
            current = self.semantic_filters.get(inst)
            if not current:
                self.semantic_filters[inst] = default
        self.semantic_overrides = dict(semantic_overrides) if semantic_overrides else None
        self.profile = self.simulator.profile
        if yaml is not None and profile_file.exists():
            self.profile_raw = yaml.safe_load(profile_file.read_text(encoding="utf-8")) or {}
        else:
            self.profile_raw = {}

    # ------------------------------------------------------------------
    # Public runner
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        start: datetime,
        end: datetime,
        grid: Dict[str, Dict[str, Sequence[float] | Sequence[int]]],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "instruments": {},
            "grid": {},
        }

        progress_payload = {
            "state": "running",
            "generated_at": summary["generated_at"],
            "window": summary["window"],
            "instruments": {},
        }

        total = len(self.instruments)
        if self.error_path.exists():
            self.error_path.unlink()
        try:
            default_cfg = cast(Dict[str, Sequence[float] | Sequence[int]], grid.get("default") or {}) if isinstance(grid, dict) else {}
            for idx, instrument in enumerate(self.instruments, start=1):
                instrument_cfg = cast(Dict[str, Sequence[float] | Sequence[int]], grid.get(instrument.upper()) or {}) if isinstance(grid, dict) else {}
                merged_cfg: Dict[str, Sequence[float] | Sequence[int]] = {}
                if default_cfg:
                    merged_cfg.update(default_cfg)
                if instrument_cfg:
                    merged_cfg.update(instrument_cfg)
                rows = self._evaluate_instrument(instrument, start, end, merged_cfg)
                summary["grid"][instrument] = [row.to_dict() for row in rows]
                best = self._select_best(instrument, rows)
                progress_payload["instruments"][instrument] = self._instrument_view(best)
                summary["instruments"][instrument] = self._instrument_view(best)
                progress_payload["progress"] = {
                    "completed": idx,
                    "total": total,
                    "percent": round((idx / total) * 100.0, 2),
                }
                self._write_atomic(self.partial_path, progress_payload)

            summary["state"] = "completed"
            self._write_atomic(self.output_path, summary)
            if self.partial_path.exists():
                self.partial_path.unlink()
            if self.error_path.exists():
                self.error_path.unlink()
            return summary
        except Exception as exc:  # pragma: no cover - failure path
            error_payload = {
                "state": "error",
                "generated_at": datetime.now(UTC).isoformat(),
                "window": summary["window"],
                "message": str(exc),
            }
            self._write_atomic(self.error_path, error_payload)
            raise

    # ------------------------------------------------------------------
    # Instrument helpers
    # ------------------------------------------------------------------
    def _evaluate_instrument(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
        overrides: Dict[str, Sequence[float] | Sequence[int]],
    ) -> List[GridRow]:
        params_grid = self._build_grid(instrument, overrides)
        rows: List[GridRow] = []
        required_tags = self.semantic_filters.get(instrument.upper()) if hasattr(self, "semantic_filters") else None
        for params in params_grid:
            sim_params = SimulationParams(
                hazard_override=params.hazard,
                min_repetitions=params.min_repetitions,
                hold_minutes=params.hold_minutes,
                exposure_scale=params.exposure,
            )
            result = self.simulator.simulate(
                instrument,
                start=start,
                end=end,
                params=sim_params,
                semantic_filter=required_tags,
                semantic_overrides=self.semantic_overrides,
            )
            if result is None:
                continue
            metrics = self._compose_metrics(result)
            qualified = metrics["trades"] >= 10
            row = GridRow(
                instrument=instrument,
                params=params,
                source=result.source,
                metrics=metrics,
                qualified=qualified,
                insufficient_data=not qualified,
                detail=result.to_dict(),
            )
            row.score, row.penalties = self._score_row(row)
            if row.detail is not None:
                row.detail.setdefault("analysis", {})
                row.detail["analysis"]["score"] = row.score
                if row.penalties:
                    row.detail["analysis"]["penalties"] = list(row.penalties)
            rows.append(row)
        return rows

    def _build_grid(self, instrument: str, overrides: Dict[str, Sequence[float] | Sequence[int]]) -> List[GridParams]:
        profile_inst = self.profile.get(instrument)
        hazard_cap = profile_inst.hazard_max or float(self.profile_raw.get("global", {}).get("hazard_max", 0.08) or 0.08)
        hazard_values: set[float] = set()

        for offset in (-0.02, 0.0, 0.02):
            hazard = min(0.14, max(0.06, hazard_cap + offset))
            hazard_values.add(round(hazard, 6))

        override_mults = overrides.get("hazard_multiplier") or []
        for mult in override_mults:
            try:
                hazard = float(hazard_cap) * float(mult)
            except Exception:
                continue
            hazard = min(0.14, max(0.06, hazard))
            hazard_values.add(round(hazard, 6))

        hazard_list = sorted(hazard_values)

        min_reps_values: set[int] = {max(1, profile_inst.min_repetitions)}
        override_reps = overrides.get("min_repetitions") or []
        for value in override_reps:
            try:
                min_reps_values.add(max(1, int(value)))
            except Exception:
                continue

        hold_allowed = self._allowed_hold_minutes(instrument)
        hold_values: set[int] = set(hold_allowed)
        override_hold = overrides.get("hold_minutes") or []
        if override_hold:
            filtered = {int(value) for value in override_hold if int(value) in hold_allowed}
            if filtered:
                hold_values = filtered
            else:
                hold_values.update(int(value) for value in override_hold)
        if not hold_values:
            hold_values = {30}

        exposure_values: set[float] = {0.02}
        override_exposure = overrides.get("exposure_scale") or overrides.get("exposure") or []
        if override_exposure:
            exposure_values = set()
            for value in override_exposure:
                try:
                    exposure_values.add(max(0.001, float(value)))
                except Exception:
                    continue

        grid_params: List[GridParams] = []
        for hazard in hazard_list:
            for reps in sorted(min_reps_values):
                for hold in sorted(hold_values):
                    for exposure in sorted(exposure_values):
                        grid_params.append(
                            GridParams(
                                hazard=hazard,
                                min_repetitions=reps,
                                hold_minutes=hold,
                                exposure=exposure,
                            )
                        )
        return grid_params

    def _allowed_hold_minutes(self, instrument: str) -> List[int]:
        instruments = self.profile_raw.get("instruments", {}) if isinstance(self.profile_raw, dict) else {}
        entry = instruments.get(instrument.upper()) or {}
        exit_cfg = entry.get("exit") or {}
        values: set[int] = set()
        for key in ("exit_horizon", "max_hold_minutes"):
            value = exit_cfg.get(key)
            if value is None:
                continue
            try:
                values.add(int(value))
            except Exception:
                continue
        if not values:
            global_exit = self.profile_raw.get("global", {}).get("exit_horizon") if isinstance(self.profile_raw, dict) else None
            if global_exit:
                try:
                    values.add(int(global_exit))
                except Exception:
                    pass
        if not values:
            values.add(30)
        return sorted(values)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def _compose_metrics(self, result: SimulationResult) -> Dict[str, Any]:
        trades = result.metrics.trades
        wins = 0.0
        losses = 0.0
        win_count = 0
        loss_count = 0
        trade_pnls: List[float] = []
        hold_minutes: List[float] = []
        for trade in result.trades:
            pnl = float(trade.pnl)
            if pnl > 0:
                wins += pnl
                win_count += 1
            elif pnl < 0:
                losses += abs(pnl)
                loss_count += 1
            trade_pnls.append(pnl)
            try:
                hold_minutes.append(
                    max(0.0, (trade.exit_time - trade.entry_time).total_seconds() / 60.0)
                )
            except Exception:
                continue
        loss_denom = max(losses, 1e-6)
        nav = getattr(self.simulator, "nav", 100_000.0)
        profit_factor = wins / loss_denom
        try:
            median_pnl = statistics.median(trade_pnls) if trade_pnls else 0.0
        except statistics.StatisticsError:
            median_pnl = 0.0
        avg_hold = sum(hold_minutes) / len(hold_minutes) if hold_minutes else 0.0
        win_loss_ratio = (win_count / loss_count) if loss_count else (float("inf") if win_count else 0.0)
        metrics = {
            "trades": int(trades),
            "sharpe": self._finite(result.metrics.sharpe),
            "pnl": self._finite(result.metrics.pnl),
            "win_rate": self._finite(result.metrics.win_rate),
            "max_dd": self._finite(result.metrics.max_drawdown),
            "profit_factor": self._finite(profit_factor),
            "return_pct": self._finite(result.metrics.return_pct),
            "gate_coverage": dict(result.gate_coverage or {}),
            "source": result.source,
            "avg_hold_minutes": round(avg_hold, 2),
            "median_trade_pnl": self._finite(median_pnl),
            "win_loss_ratio": self._finite(win_loss_ratio if math.isfinite(win_loss_ratio) else 0.0),
            "pnl_to_drawdown": self._finite(
                (result.metrics.pnl / max(1.0, result.metrics.max_drawdown))
                if result.metrics.max_drawdown > 0
                else 0.0
            ),
            "pnl_per_trade": self._finite(result.metrics.pnl / max(1, trades)),
            "nav": self._finite(nav),
        }
        return metrics

    def _select_best(self, instrument: str, rows: Sequence[GridRow]) -> GridRow:
        if not rows:
            return GridRow(
                instrument=instrument,
                params=GridParams(hazard=0.0, min_repetitions=0, hold_minutes=0, exposure=0.0),
                source="empty",
                metrics=self._empty_metrics(),
                qualified=False,
                insufficient_data=True,
            )
        qualified = [row for row in rows if row.qualified]
        target = qualified or rows
        best = max(
            target,
            key=lambda row: (
                row.score,
                row.metrics.get("pnl", 0.0),
                row.metrics.get("profit_factor", 0.0),
                row.metrics.get("sharpe", 0.0),
            ),
        )
        if not qualified:
            best = dataclasses.replace(best, qualified=False, insufficient_data=True)
        return best

    def _score_row(self, row: GridRow) -> Tuple[float, List[str]]:
        metrics = row.metrics
        nav = max(self.simulator.nav, 1.0) if hasattr(self.simulator, "nav") else 100_000.0
        pnl = float(metrics.get("pnl", 0.0))
        sharpe = float(metrics.get("sharpe", 0.0))
        return_pct = float(metrics.get("return_pct", 0.0))
        profit_factor = float(metrics.get("profit_factor", 0.0))
        max_dd = float(metrics.get("max_dd", 0.0)) / nav
        trades = int(metrics.get("trades", 0))
        coverage = metrics.get("gate_coverage") or {}
        real_fraction = float(coverage.get("real_fraction") or 0.0)
        synthetic_fraction = float(coverage.get("synthetic_minutes") or 0.0)
        total_minutes = float(coverage.get("minutes") or 0.0)
        penalties: List[str] = []

        score = 0.0
        score += sharpe * 0.5
        score += return_pct * 1.2
        score += (pnl / nav) * 1.0
        score += max(0.0, profit_factor - 1.0) * 0.75
        score += math.log1p(max(trades - 9, 0)) * 0.15 if trades else 0.0
        score -= max_dd * 0.8
        if total_minutes > 0 and synthetic_fraction:
            synthetic_ratio = synthetic_fraction / total_minutes
            score -= max(0.0, synthetic_ratio - 0.4) * 0.5
            if synthetic_ratio > 0.6:
                penalties.append("high_synthetic_coverage")
        score += real_fraction * 0.25

        if trades < 10:
            penalties.append("low_trade_count")
            score -= 0.6
        if pnl <= 0:
            penalties.append("negative_pnl")
            score -= 0.8
        if profit_factor < 1.0:
            penalties.append("profit_factor_below_one")
            score -= 0.5
        if sharpe <= 0:
            penalties.append("non_positive_sharpe")
            score -= 0.4
        if max_dd > 0 and pnl > 0 and pnl / nav < max_dd:
            penalties.append("drawdown_exceeds_pnl")
            score -= 0.3

        return round(score, 4), penalties

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def _instrument_view(self, row: GridRow) -> Dict[str, Any]:
        metrics = dict(row.metrics)
        metrics.pop("gate_coverage", None)
        metrics.pop("source", None)
        payload = {
            "params": {
                "hazard_x": row.params.hazard,
                "min_reps": row.params.min_repetitions,
                "hold_minutes": row.params.hold_minutes,
                "exposure": row.params.exposure,
            },
            "metrics": metrics,
            "qualified": bool(row.qualified),
            "insufficient_data": bool(row.insufficient_data),
            "source": row.source,
        }
        payload["score"] = row.score
        if row.penalties:
            payload["penalties"] = list(row.penalties)
        if row.metrics.get("gate_coverage"):
            payload["gate_coverage"] = row.metrics.get("gate_coverage")
        return payload

    def _write_atomic(self, path: pathlib.Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _finite(self, value: Any) -> float:
        try:
            num = float(value)
        except Exception:
            return 0.0
        if not math.isfinite(num):
            return 0.0
        return num

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "trades": 0,
            "sharpe": 0.0,
            "pnl": 0.0,
            "win_rate": 0.0,
            "max_dd": 0.0,
            "profit_factor": 0.0,
            "return_pct": 0.0,
            "gate_coverage": {},
            "source": "empty",
        }

def compute_week_range(reference: Optional[date] = None) -> Tuple[datetime, datetime]:
    today = reference or datetime.now(UTC).date()
    monday = today - timedelta(days=today.weekday())
    start_dt = datetime.combine(monday, time(0, 0, tzinfo=UTC))
    end_dt = start_dt + timedelta(days=5)
    now = datetime.now(UTC)
    if end_dt > now:
        end_dt = now
    return start_dt, end_dt


def load_grid_config(path: pathlib.Path) -> Dict[str, Dict[str, Sequence[float] | Sequence[int]]]:
    if not path.exists():
        return {
            "default": {
                "hazard_multiplier": [0.8, 1.0, 1.2],
                "min_repetitions": [1, 2],
                "hold_minutes": [20, 30, 45],
                "exposure_scale": [0.015, 0.02, 0.03],
            }
        }
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() == ".json" or yaml is None:
            data = json.load(fh)
        else:
            data = yaml.safe_load(fh)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Run grid-search backtests for configured instruments")
    parser.add_argument("--redis", default="redis://localhost:6379/0", help="Valkey URL")
    parser.add_argument("--start", help="ISO start override (default = current week)")
    parser.add_argument("--end", help="ISO end override (default = current week)")
    parser.add_argument("--output", default="output/backtests/latest.json", help="Where to write summary JSON")
    parser.add_argument("--grid-config", default="config/backtest_grid.json", help="Grid configuration file (JSON/YAML)")
    parser.add_argument("--profile", default="config/echo_strategy.yaml", help="Strategy profile path")
    parser.add_argument("--granularity", default="M1", help="Candle granularity for simulator")
    parser.add_argument("--balance", type=float, default=100_000.0, help="Starting balance for backtest NAV")
    parser.add_argument("--nav-risk-pct", type=float, default=0.01, help="NAV risk percent used for sizing")
    parser.add_argument("--cost-bps", type=float, default=1.5, help="Commission/slippage in basis points")
    parser.add_argument("--instruments", nargs="*", default=["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CHF", "USD_CAD"], help="Instrument whitelist")
    args = parser.parse_args()

    if args.start and args.end:
        start = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(UTC)
        end = datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(UTC)
    else:
        start, end = compute_week_range()

    grid_config = load_grid_config(pathlib.Path(args.grid_config))

    runner = BacktestRunner(
        instruments=args.instruments,
        redis_url=args.redis,
        granularity=args.granularity,
        profile_path=pathlib.Path(args.profile),
        nav=args.balance,
        nav_risk_pct=args.nav_risk_pct,
        cost_bps=args.cost_bps,
        output_path=pathlib.Path(args.output),
    )
    summary = runner.run(start=start, end=end, grid=grid_config)
    print(f"Wrote backtest summary to {runner.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
