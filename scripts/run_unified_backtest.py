#!/usr/bin/env python3
"""Unified SEP backtesting entrypoint with semantic regime filters."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.simulator.backtest_simulator import (  # noqa: E402
    BacktestSimulator,
    SimulationParams,
)

UTC = timezone.utc

DEFAULT_NAV = 100_000.0
DEFAULT_NAV_RISK_PCT = 0.01
DEFAULT_COST_BPS = 1.5
DEFAULT_EXPOSURE = 0.02
DEFAULT_HOLD_MINUTES = 30
DEFAULT_MIN_REPETITIONS = 1


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _parse_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _parse_threshold_overrides(entries: Sequence[str]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for entry in entries:
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid semantic threshold override '{entry}'")
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing key in semantic threshold override '{entry}'")
        try:
            overrides[key] = float(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid numeric value in semantic threshold override '{entry}'") from exc
    return overrides


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Configuration must evaluate to a dictionary")
        return data


def _resolve_instruments(
    cli_instruments: Sequence[str] | None,
    config: Mapping[str, Any],
) -> List[str]:
    if cli_instruments:
        return [inst.upper() for inst in cli_instruments]
    semantic_filters = config.get("semantic_filters")
    if isinstance(semantic_filters, Mapping):
        return [inst.upper() for inst in semantic_filters.keys()]
    instruments = config.get("instruments")
    if isinstance(instruments, Iterable):
        return [str(inst).upper() for inst in instruments]
    raise ValueError("No instruments provided via CLI or config")


def _semantic_overrides(
    cli_overrides: Mapping[str, float],
    config: Mapping[str, Any],
) -> Dict[str, float]:
    config_overrides = config.get("semantic_thresholds")
    payload: Dict[str, float] = {}
    if isinstance(config_overrides, Mapping):
        for key, value in config_overrides.items():
            try:
                payload[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid semantic threshold value for '{key}' in config") from exc
    payload.update(cli_overrides)
    return payload


def _semantic_filters_for(
    instrument: str,
    default_tags: Sequence[str],
    config_filters: Mapping[str, Any] | None,
    profile: Any,
) -> List[str]:
    if config_filters and instrument in config_filters:
        tags = config_filters[instrument]
        if isinstance(tags, str):
            return _parse_tags(tags)
        if isinstance(tags, Iterable):
            return [str(tag).strip() for tag in tags if str(tag).strip()]
    if default_tags:
        return list(default_tags)
    inst_profile = profile.get(instrument)
    return list(getattr(inst_profile, "semantic_filter", []))


def _merge_params(
    base: SimulationParams,
    override: Mapping[str, Any] | None,
) -> SimulationParams:
    if not override:
        return base
    params_dict = {
        "hazard_override": override.get("hazard_override", base.hazard_override),
        "hazard_multiplier": override.get("hazard_multiplier", base.hazard_multiplier),
        "min_repetitions": int(override.get("min_repetitions", base.min_repetitions)),
        "hold_minutes": int(override.get("hold_minutes", base.hold_minutes)),
        "exposure_scale": float(override.get("exposure_scale", base.exposure_scale)),
    }
    return SimulationParams(**params_dict)


def _summarise_semantic_usage(decisions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tag_counter: Counter[str] = Counter()
    filtered = 0
    for decision in decisions:
        tags = decision.get("semantic_tags")
        if isinstance(tags, Iterable):
            tag_counter.update(tag for tag in tags if isinstance(tag, str))
        reasons = decision.get("reasons")
        if isinstance(reasons, Iterable):
            if any(isinstance(reason, str) and reason.startswith("semantic_filter") for reason in reasons):
                filtered += 1
    return {
        "total_minutes": len(decisions),
        "tag_counts": dict(tag_counter),
        "filtered_minutes": filtered,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unified SEP backtests with semantic regime filters")
    parser.add_argument("--config", type=Path, help="Optional JSON/YAML config file")
    parser.add_argument("--instruments", nargs="*", help="Instrument symbols (e.g., EUR_USD)")
    parser.add_argument("--start", help="Backtest window start (ISO8601 UTC)")
    parser.add_argument("--end", help="Backtest window end (ISO8601 UTC)")
    parser.add_argument("--profile", type=Path, default=Path("config/echo_strategy.yaml"), help="Strategy profile path")
    parser.add_argument("--redis-url", default=os.getenv("VALKEY_URL") or "redis://localhost:6379/0", help="Valkey/Redis URL")
    parser.add_argument("--granularity", default="M1", help="Candle granularity")
    parser.add_argument("--nav", type=float, help="Account NAV (default 100000)")
    parser.add_argument("--nav-risk-pct", type=float, help="NAV risk percent (default 1%)")
    parser.add_argument("--cost-bps", type=float, help="Basis-point cost assumption (default 1.5)")
    parser.add_argument("--exposure", type=float, help="Exposure scale passed to risk sizer (default 0.02)")
    parser.add_argument("--hold-minutes", type=int, help="Hold minutes per position (default 30)")
    parser.add_argument("--min-reps", type=int, help="Override minimum repetition count (default 1)")
    parser.add_argument("--hazard", type=float, help="Hazard override applied to the profile")
    parser.add_argument("--hazard-multiplier", type=float, help="Hazard multiplier applied to profile settings")
    parser.add_argument("--semantic-regime-filter", help="Comma-separated semantic tags applied to every instrument")
    parser.add_argument(
        "--semantic-threshold",
        action="append",
        default=[],
        help="Override semantic thresholds with key=value syntax (e.g. high_coherence=0.8)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/backtests/unified_latest.json"),
        help="Where to write the JSON summary",
    )
    parser.add_argument(
        "--instrument-overrides",
        type=Path,
        help="Optional JSON/YAML file providing per-instrument parameter overrides",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without running backtests",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    inst_overrides = _load_config(args.instrument_overrides)
    if isinstance(inst_overrides, Mapping):
        inst_overrides = {
            str(key).upper(): value for key, value in inst_overrides.items()
        }

    instruments = _resolve_instruments(args.instruments, config)

    window_cfg = config.get("window") if isinstance(config.get("window"), Mapping) else {}
    try:
        start_value = args.start or window_cfg.get("start")
        end_value = args.end or window_cfg.get("end")
    except AttributeError as exc:
        raise ValueError("Window configuration must be a mapping") from exc
    if not start_value or not end_value:
        raise ValueError("Both start and end timestamps must be supplied via CLI or config window")
    start = _parse_time(str(start_value))
    end = _parse_time(str(end_value))

    semantic_defaults = _parse_tags(args.semantic_regime_filter)
    raw_filters = config.get("semantic_filters")
    config_filters = None
    if isinstance(raw_filters, Mapping):
        config_filters = {
            str(key).upper(): value for key, value in raw_filters.items()
        }

    cli_semantic_overrides = _parse_threshold_overrides(args.semantic_threshold)
    overrides = _semantic_overrides(cli_semantic_overrides, config)

    nav = float(config.get("nav", DEFAULT_NAV))
    if args.nav is not None:
        nav = float(args.nav)

    nav_risk_pct = float(config.get("nav_risk_pct", DEFAULT_NAV_RISK_PCT))
    if args.nav_risk_pct is not None:
        nav_risk_pct = float(args.nav_risk_pct)

    cost_bps = float(config.get("cost_bps", DEFAULT_COST_BPS))
    if args.cost_bps is not None:
        cost_bps = float(args.cost_bps)

    exposure_scale = float(config.get("exposure", DEFAULT_EXPOSURE))
    if args.exposure is not None:
        exposure_scale = float(args.exposure)

    hold_minutes = int(config.get("hold_minutes", DEFAULT_HOLD_MINUTES))
    if args.hold_minutes is not None:
        hold_minutes = int(args.hold_minutes)

    cfg_min_reps = config.get("min_repetitions", config.get("min_reps"))
    min_repetitions = int(cfg_min_reps) if cfg_min_reps is not None else DEFAULT_MIN_REPETITIONS
    if args.min_reps is not None:
        min_repetitions = int(args.min_reps)

    hazard_override_value = config.get("hazard")
    if hazard_override_value is not None:
        hazard_override_value = float(hazard_override_value)
    if args.hazard is not None:
        hazard_override_value = float(args.hazard)

    hazard_multiplier_value = config.get("hazard_multiplier")
    if hazard_multiplier_value is not None:
        hazard_multiplier_value = float(hazard_multiplier_value)
    if args.hazard_multiplier is not None:
        hazard_multiplier_value = float(args.hazard_multiplier)

    redis_url = str(config.get("redis_url", args.redis_url))
    granularity = str(config.get("granularity", args.granularity))
    profile_path = Path(config.get("profile", args.profile))

    simulator = BacktestSimulator(
        redis_url=redis_url,
        granularity=granularity,
        profile_path=profile_path,
        nav=nav,
        nav_risk_pct=nav_risk_pct,
        cost_bps=cost_bps,
    )

    base_params = SimulationParams(
        hazard_override=hazard_override_value,
        hazard_multiplier=hazard_multiplier_value,
        min_repetitions=min_repetitions,
        hold_minutes=hold_minutes,
        exposure_scale=exposure_scale,
    )
    profile_override = None

    if args.dry_run:
        snapshot = {
            "instruments": instruments,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "semantic_defaults": semantic_defaults,
            "semantic_filters": config_filters,
            "semantic_thresholds": overrides,
            "nav": args.nav,
            "nav_risk_pct": args.nav_risk_pct,
            "cost_bps": args.cost_bps,
        }
        print(json.dumps(snapshot, indent=2))
        return 0

    summary: Dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "semantic_regime_filter": semantic_defaults,
        "semantic_thresholds": overrides,
        "nav": nav,
        "nav_risk_pct": nav_risk_pct,
        "cost_bps": cost_bps,
        "hold_minutes": hold_minutes,
        "exposure_scale": exposure_scale,
        "instruments": {},
    }

    results: Dict[str, Any] = {}
    for instrument in instruments:
        inst_filters = _semantic_filters_for(instrument, semantic_defaults, config_filters, simulator.profile)
        per_inst_override = None
        if isinstance(inst_overrides, Mapping):
            per_inst_override = inst_overrides.get(instrument) or inst_overrides.get(instrument.upper())
        params = _merge_params(base_params, per_inst_override if isinstance(per_inst_override, Mapping) else None)
        sim_result = simulator.simulate(
            instrument,
            start=start,
            end=end,
            params=params,
            profile_override=profile_override,
            semantic_filter=inst_filters,
            semantic_overrides=overrides,
        )
        if sim_result is None:
            results[instrument] = {
                "error": "no_result",
                "applied_semantic_filter": inst_filters,
            }
            continue
        payload = sim_result.to_dict()
        payload["semantic_summary"] = _summarise_semantic_usage(payload.get("decisions", []))
        payload["applied_semantic_filter"] = inst_filters
        results[instrument] = payload

    summary["instruments"] = results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI tool
    raise SystemExit(main())
