#!/usr/bin/env python3
"""Lean bootstrapper for the simplified SEP trading system."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from logging import handlers
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None

# Ensure the repository root is on sys.path when executed as ``python scripts/...``
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trading import oanda as oanda_service
from scripts.trading.api import start_http_server
from scripts.trading.portfolio_manager import PortfolioManager, StrategyProfile, gate_evaluation, structural_metric
from scripts.trading.risk_planner import RiskLimits, RiskManager
from scripts.research.backtest_grid import BacktestRunner, compute_week_range, load_grid_config

def _configure_logging() -> logging.Logger:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()
    log_to_file = os.getenv("LOG_TO_FILE", "1").lower() not in {"0", "false", "off"}
    log_file_path = os.getenv("LOG_FILE_PATH", "/app/logs/backend.log")
    max_bytes = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting only
            payload = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            if record.__dict__:
                extras = {k: v for k, v in record.__dict__.items() if k not in logging.LogRecord.__dict__}
                if extras:
                    payload.update(extras)
            return json.dumps(payload, ensure_ascii=False)

    class PlainFormatter(logging.Formatter):
        def __init__(self) -> None:
            super().__init__("%(asctime)s %(levelname)s %(name)s :: %(message)s")

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    formatter: logging.Formatter
    if log_format == "json":
        formatter = JsonFormatter()
    else:
        formatter = PlainFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_to_file:
        try:
            Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
            file_handler = handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except Exception as exc:  # pragma: no cover - file system error path
            root.warning("Failed to configure file logging: %s", exc)

    return logging.getLogger(__name__)


logger = _configure_logging()


class PriceHistoryCache:
    """Cache small pricing series snapshots in Valkey (or memory fallback)."""

    def __init__(self, redis_url: Optional[str], *, ttl_seconds: int = 300, max_points: int = 500) -> None:
        self.ttl_seconds = max(30, int(ttl_seconds))
        self.max_points = max(10, int(max_points))
        self._client = None
        if redis and redis_url:
            try:
                self._client = redis.from_url(redis_url)
            except Exception:  # pragma: no cover - network error path
                self._client = None
        self._memory: Dict[str, Tuple[float, List[Dict[str, object]]]] = {}

    def _key(self, instrument: str, granularity: str) -> str:
        return f"pricing:history:{granularity.upper()}:{instrument.upper()}"

    def get(self, instrument: str, granularity: str) -> Tuple[List[Dict[str, object]], Optional[float]]:
        key = self._key(instrument, granularity)
        payload: Optional[Dict[str, object]] = None
        if self._client:
            try:
                raw = self._client.get(key)
            except Exception:  # pragma: no cover - network error path
                raw = None
            if raw:
                try:
                    payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                except Exception:
                    payload = None
        if not payload:
            entry = self._memory.get(key)
            if entry:
                ts, points = entry
                return [dict(p) for p in points], ts
            return [], None
        points = payload.get("points") if isinstance(payload, dict) else None
        fetched_at = payload.get("fetched_at") if isinstance(payload, dict) else None
        if not isinstance(points, list):
            points = []
        ts_value: Optional[float]
        try:
            ts_value = float(fetched_at) if fetched_at is not None else None
        except Exception:
            ts_value = None
        return [dict(p) for p in points], ts_value

    def set(self, instrument: str, granularity: str, points: List[Dict[str, object]]) -> None:
        key = self._key(instrument, granularity)
        trimmed = [dict(p) for p in points][-self.max_points :]
        record = {
            "points": trimmed,
            "fetched_at": time.time(),
        }
        if self._client:
            try:
                self._client.set(key, json.dumps(record), ex=self.ttl_seconds)
            except Exception:  # pragma: no cover - network error path
                pass
        self._memory[key] = (record["fetched_at"], trimmed)


class TradingService:
    """Orchestrates the OANDA connector, portfolio manager, and HTTP API."""

    def __init__(self, *, read_only: bool = False, enabled_pairs: Optional[Iterable[str]] = None) -> None:
        self.read_only = read_only
        self.kill_switch_enabled = True
        self.trading_active = False
        self.running = False

        profile = StrategyProfile.load(Path(os.getenv("STRATEGY_PROFILE", "config/echo_strategy.yaml")))
        default_pairs = sorted(profile.instruments.keys()) or ["EUR_USD", "GBP_USD", "USD_JPY"]
        self.enabled_pairs = list({inst.upper() for inst in (enabled_pairs or default_pairs)})

        self.oanda = oanda_service.OandaConnector(read_only=read_only)
        self.risk_manager = RiskManager(RiskLimits())
        self.portfolio_manager = PortfolioManager(self)

        redis_url = os.getenv("VALKEY_URL") or os.getenv("REDIS_URL")
        self._valkey_client = self._connect_valkey(redis_url)
        self.kill_switch_key = os.getenv("KILL_SWITCH_KEY", "ops:kill_switch")
        self.kill_switch_enabled = self._load_kill_switch_state(default=True)
        ttl_seconds = int(os.getenv("PRICING_HISTORY_TTL", "300") or 300)
        max_points = int(os.getenv("PRICING_HISTORY_MAX_POINTS", "500") or 500)
        self.price_history_cache = PriceHistoryCache(redis_url, ttl_seconds=ttl_seconds, max_points=max_points)
        self.backtest_results_path = Path(os.getenv("BACKTEST_RESULTS_PATH", "output/backtests/latest.json"))
        self.backtest_partial_path = self.backtest_results_path.with_suffix(".partial.json")
        self.backtest_error_path = self.backtest_results_path.with_suffix(".error.json")
        self.backtest_grid_config = Path(os.getenv("BACKTEST_GRID_CONFIG", "config/backtest_grid.json"))
        self._backtest_lock = threading.Lock()
        self._backtest_status: Dict[str, Any] = {"state": "idle"}
        self._backtest_thread: Optional[threading.Thread] = None

        self.signal_evidence_path = Path(os.getenv("SIGNAL_EVIDENCE_PATH", "docs/evidence/outcome_weekly_costs.json"))
        self._signal_evidence_cache: Optional[Dict[str, Any]] = None
        self._signal_evidence_mtime: Optional[float] = None
        self.roc_summary_path = Path(os.getenv("ROC_REGIME_SUMMARY_PATH", "docs/evidence/roc_regime_summary.json"))
        self._roc_summary_cache: Optional[Dict[str, Any]] = None
        self._roc_summary_mtime: Optional[float] = None
        self.bundle_evidence_path = Path(os.getenv("BUNDLE_EVIDENCE_PATH", "docs/evidence/bundle_outcomes.json"))
        self._bundle_evidence_cache: Optional[Dict[str, Any]] = None
        self._bundle_evidence_mtime: Optional[float] = None

        self._api_server = None
        self._shutdown = False

    def _connect_valkey(self, redis_url: Optional[str]):
        if not redis or not redis_url:
            return None
        try:
            return redis.from_url(redis_url)
        except Exception:
            logger.warning("Unable to connect to Valkey at %s", redis_url)
            return None

    def _load_kill_switch_state(self, *, default: bool) -> bool:
        if not self._valkey_client or not self.kill_switch_key:
            return default
        try:
            raw = self._valkey_client.get(self.kill_switch_key)
        except Exception:
            logger.warning("Failed to read kill switch key %s", self.kill_switch_key)
            return default
        if raw is None:
            return default
        value = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        return value.strip() not in {"0", "false", "False"}

    def _persist_kill_switch(self, flag: bool) -> None:
        if not self._valkey_client or not self.kill_switch_key:
            return
        try:
            self._valkey_client.set(self.kill_switch_key, "1" if flag else "0")
        except Exception:
            logger.warning("Failed to persist kill switch state to %s", self.kill_switch_key)

    def _sync_trading_state(self) -> None:
        self.trading_active = bool(self.running and not self.read_only and not self.kill_switch_enabled)

    def set_kill_switch(self, enabled: bool) -> bool:
        flag = bool(enabled)
        self.kill_switch_enabled = flag
        self._persist_kill_switch(flag)
        self._sync_trading_state()
        return flag

    # ------------------------------------------------------------------
    # Metrics + diagnostics
    # ------------------------------------------------------------------
    def nav_metrics(self) -> Dict[str, object]:
        summary = self.risk_manager.get_risk_summary()
        read_only = str(os.getenv("READ_ONLY", "1")).lower() in {"1", "true", "yes", "on"}
        show_positions = self.trading_active and not self.kill_switch_enabled and not read_only
        positions = self.risk_manager.position_breakdown() if show_positions else []
        if not show_positions:
            summary["total_units"] = 0.0
            summary["exposure_usd"] = 0.0
        summary.update(
            {
                "positions": positions,
                "kill_switch": self.kill_switch_enabled,
                "trading_active": self.trading_active,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return summary

    def gate_metrics(self) -> Dict[str, object]:
        instruments = list(self.enabled_pairs or [])
        payloads: Dict[str, Dict[str, object]] = {}
        try:
            payloads = self.portfolio_manager.latest_gate_payloads()
            if not payloads and getattr(self.portfolio_manager, "gate_reader", None):
                payloads = self.portfolio_manager.gate_reader.load(instruments)
        except Exception:
            payloads = {}

        entries: List[Dict[str, object]] = []
        now = datetime.now(timezone.utc)
        for inst in instruments:
            payload = payloads.get(inst.upper()) or {}
            ts_raw = payload.get("ts_ms") or payload.get("ts")
            updated_at = None
            age = None
            if ts_raw is not None:
                updated: Optional[datetime] = None
                try:
                    if isinstance(ts_raw, (int, float)) or (isinstance(ts_raw, str) and ts_raw.replace(".", "", 1).isdigit()):
                        ts_val = float(ts_raw)
                        if ts_val > 10_000:
                            ts_val /= 1000.0
                        updated = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                    elif isinstance(ts_raw, str):
                        updated = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except Exception:
                    updated = None
                if updated:
                    updated_at = updated.isoformat()
                    age = max(0.0, (now - updated).total_seconds())
            strategy_profile = self.portfolio_manager.strategy.get(inst)
            admitted, reasons = gate_evaluation(payload, strategy_profile)
            coh_tau_slope = structural_metric(payload, "coherence_tau_slope")
            domain_wall_slope = structural_metric(payload, "domain_wall_slope")
            spectral_lowf_share = structural_metric(payload, "spectral_lowf_share")
            entries.append(
                {
                    "instrument": inst,
                    "admit": admitted,
                    "age_seconds": age,
                    "updated_at": updated_at,
                    "hazard": payload.get("lambda"),
                    "hazard_threshold": (
                        (payload.get("structure") or {}).get("hazard_threshold")
                        or payload.get("hazard_threshold")
                    ),
                    "repetitions": payload.get("repetitions"),
                    "reasons": reasons,
                    "regime": payload.get("regime"),
                    "components": payload.get("components"),
                    "structure_metrics": {
                        "coherence_tau_slope": coh_tau_slope,
                        "domain_wall_slope": domain_wall_slope,
                        "spectral_lowf_share": spectral_lowf_share,
                    },
                    "bundle_hits": payload.get("bundle_hits"),
                    "bundle_blocks": payload.get("bundle_blocks"),
                    "bundle_readiness": payload.get("bundle_readiness"),
                    "raw": payload,
                }
            )

        return {"as_of": now.isoformat(), "gates": entries}

    def price_history(
        self,
        instrument: str,
        *,
        granularity: str = "M5",
        count: int = 200,
    ) -> Dict[str, object]:
        instrument_code = (instrument or "").upper()
        if not instrument_code:
            return {"instrument": instrument_code, "points": []}
        count = max(1, min(int(count or 0), 500))
        granularity_code = granularity.upper() if granularity else "M5"
        cache: Optional[PriceHistoryCache] = getattr(self, "price_history_cache", None)
        cached_points: List[Dict[str, object]] = []
        cached_ts: Optional[float] = None
        if cache:
            cached_points, cached_ts = cache.get(instrument_code, granularity_code)

        now_ts = time.time()
        if cached_points and cached_ts is not None and cache and (now_ts - cached_ts) < cache.ttl_seconds:
            return {
                "instrument": instrument_code,
                "granularity": granularity_code,
                "points": cached_points[-count:],
                "source": "cache",
            }

        candles: List[Dict[str, object]] = []
        connector = getattr(self, "oanda", None)
        if connector:
            try:
                candles = connector.get_candles(instrument_code, granularity=granularity_code, count=max(count, 200))
            except Exception:
                candles = []
        series: List[Dict[str, object]] = []
        for candle in candles or []:
            mid = candle.get("mid") or {}
            close = mid.get("c") or mid.get("close") or mid.get("C")
            try:
                price_val = float(close) if close is not None else None
            except Exception:
                price_val = None
            time_str = candle.get("time") or None
            if price_val is None or time_str is None:
                continue
            series.append({"time": time_str, "close": price_val})

        if series:
            if cache:
                cache.set(instrument_code, granularity_code, series)
            return {
                "instrument": instrument_code,
                "granularity": granularity_code,
                "points": series[-count:],
                "source": "oanda",
            }

        if cached_points:
            return {
                "instrument": instrument_code,
                "granularity": granularity_code,
                "points": cached_points[-count:],
                "source": "cache_stale",
            }

        return {
            "instrument": instrument_code,
            "granularity": granularity_code,
            "points": [],
            "source": "empty",
        }

    def latest_backtests(self) -> Dict[str, Any]:
        base_path = self.backtest_results_path
        partial = self.backtest_partial_path
        error_file = self.backtest_error_path
        target_path = base_path
        if partial.exists():
            target_path = partial
        elif not base_path.exists() and error_file.exists():
            try:
                return json.loads(error_file.read_text(encoding="utf-8"))
            except Exception:
                return {"error": "unreadable"}
        elif not base_path.exists():
            return {"error": "not_found"}
        try:
            payload = json.loads(target_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read backtest results: %s", exc)
            return {"error": "unreadable"}
        return payload

    def backtest_status(self) -> Dict[str, Any]:
        with self._backtest_lock:
            status = dict(self._backtest_status)
        if status.get("state") == "running" and self.backtest_partial_path.exists():
            try:
                partial = json.loads(self.backtest_partial_path.read_text(encoding="utf-8"))
                progress = partial.get("progress")
                if progress:
                    status["progress"] = progress
                window = partial.get("window")
                if window and "window" not in status:
                    status["window"] = window
            except Exception:
                pass
        return status

    def trigger_backtest(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        instruments: Optional[Iterable[str]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        with self._backtest_lock:
            if self._backtest_status.get("state") == "running":
                return False, dict(self._backtest_status)

            try:
                if start and end:
                    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(timezone.utc)
                    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00")).astimezone(timezone.utc)
                else:
                    start_dt, end_dt = compute_week_range()
            except Exception as exc:
                logger.warning("Invalid backtest window: %s", exc)
                return False, {"state": "idle", "error": "invalid_time_range"}

            selected_instruments = [inst.upper() for inst in (instruments or self.enabled_pairs)]
            if not selected_instruments:
                selected_instruments = ["EUR_USD"]

            job_id = datetime.now(timezone.utc).isoformat()
            self._backtest_status = {
                "state": "running",
                "job_id": job_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "message": "Backtest grid running",
                "window": {"start": start_dt.isoformat(), "end": end_dt.isoformat()},
                "instruments": selected_instruments,
            }
            for stale in (self.backtest_partial_path, self.backtest_error_path):
                try:
                    if stale.exists():
                        stale.unlink()
                except Exception:
                    logger.debug("Failed to remove stale backtest artifact: %s", stale)

            self._backtest_thread = threading.Thread(
                target=self._run_backtest_job,
                args=(start_dt, end_dt, tuple(selected_instruments), job_id),
                name="BacktestGrid",
                daemon=True,
            )
            self._backtest_thread.start()
            return True, dict(self._backtest_status)

    def _run_backtest_job(
        self,
        start: datetime,
        end: datetime,
        instruments: Tuple[str, ...],
        job_id: str,
    ) -> None:
        try:
            gate_client = getattr(getattr(self.portfolio_manager, "gate_reader", None), "_client", None)
            if gate_client and instruments:
                latest_ts = None
                for inst in instruments:
                    try:
                        row = gate_client.zrevrange(f"gate:index:{inst}", 0, 0, withscores=True)
                    except Exception:
                        row = None
                    if row:
                        _, score = row[0]
                        if latest_ts is None or score > latest_ts:
                            latest_ts = score
                if latest_ts:
                    latest_dt = datetime.fromtimestamp(float(latest_ts) / 1000.0, tz=timezone.utc)
                    if latest_dt < end:
                        end = latest_dt
                    start_candidate = latest_dt - timedelta(days=5)
                    if start_candidate > start:
                        start = start_candidate
            if end <= start:
                end = start + timedelta(days=5)
            now = datetime.now(timezone.utc)
            if end > now:
                end = now
            grid = load_grid_config(self.backtest_grid_config)
            redis_url = os.getenv("VALKEY_URL") or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
            nav = float(os.getenv("BACKTEST_NAV", "100000") or 100000)
            nav_risk_pct = float(os.getenv("BACKTEST_NAV_RISK_PCT", "0.01") or 0.01)
            cost_bps = float(os.getenv("BACKTEST_COST_BPS", "1.5") or 1.5)
            granularity = os.getenv("BACKTEST_GRANULARITY", "M1") or "M1"

            runner = BacktestRunner(
                instruments=list(instruments),
                redis_url=redis_url,
                granularity=granularity,
                profile_path=Path(os.getenv("STRATEGY_PROFILE", "config/echo_strategy.yaml")),
                nav=nav,
                nav_risk_pct=nav_risk_pct,
                cost_bps=cost_bps,
                output_path=self.backtest_results_path,
            )
            summary = runner.run(start=start, end=end, grid=grid)
            with self._backtest_lock:
                self._backtest_status = {
                    "state": "completed",
                    "job_id": job_id,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "message": "Backtest grid completed",
                    "window": summary.get("window", {"start": start.isoformat(), "end": end.isoformat()}),
                    "instruments": list(instruments),
                    "generated_at": summary.get("generated_at"),
                }
        except Exception as exc:
            logger.exception("Backtest grid failed")
            with self._backtest_lock:
                self._backtest_status = {
                    "state": "error",
                    "job_id": job_id,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "message": str(exc),
                    "window": {"start": start.isoformat(), "end": end.isoformat()},
                }

    # ------------------------------------------------------------------
    # Public API used by HTTP layer and portfolio manager
    # ------------------------------------------------------------------
    def get_pricing(self, instruments: Iterable[str]) -> Dict[str, Dict[str, float]]:
        return oanda_service.pricing(self, list(instruments or []))

    def get_oanda_positions(self) -> List[Dict[str, object]]:
        return oanda_service.positions(self)

    def get_oanda_account_info(self) -> Dict[str, object]:
        return oanda_service.account_info(self)

    def fetch_and_store_candles(self, instrument: str, granularity: str = "M5", count: int = 200) -> bool:
        return oanda_service.fetch_and_store_candles(self, instrument, granularity, count)

    def fetch_candles_for_enabled_pairs(self, granularity: str = "M5", count: int = 200) -> None:
        oanda_service.fetch_candles_for_enabled_pairs(self, granularity, count)

    def place_order(
        self,
        instrument: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> Dict[str, object]:
        if self.kill_switch_enabled:
            return {"ok": False, "error": "kill_switch"}
        if self.read_only or self.oanda.read_only:
            return {"ok": False, "error": "read_only"}
        response = oanda_service.submit_market_order(self, instrument, units, stop_loss, take_profit) or {}
        if current_price is not None:
            self.risk_manager.record_fill(instrument, units, current_price)
        return {"ok": bool(response), "response": response}

    def close_position(self, instrument: str, units: Optional[str] = None) -> Dict[str, object]:
        if self.kill_switch_enabled:
            return {"ok": False, "error": "kill_switch"}
        response = oanda_service.close_position(self, instrument, units)
        return {"ok": bool(response), "response": response}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, *, start_api: bool = True) -> None:
        if self.running:
            return
        self.kill_switch_enabled = self._load_kill_switch_state(default=self.kill_switch_enabled)
        self._persist_kill_switch(self.kill_switch_enabled)
        if self.read_only and not self.kill_switch_enabled:
            logger.info("READ_ONLY mode active; kill switch cleared but trading loop remains disabled")
        elif self.read_only and self.kill_switch_enabled:
            logger.info("READ_ONLY mode active; kill switch engaged")
        self.running = True
        self.portfolio_manager.start()
        try:
            self.portfolio_manager.reconcile_portfolio()
        except Exception:
            logger.warning("Portfolio reconciliation failed during startup", exc_info=True)
        self._sync_trading_state()
        if start_api:
            host = os.getenv("HTTP_HOST", "0.0.0.0")
            port = int(os.getenv("HTTP_PORT", "8000") or 8000)
            self._api_server = start_http_server(self, host, port)
        logger.info("Trading service started (API=%s)", bool(start_api))

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.set_kill_switch(True)
        self.portfolio_manager.stop()
        if self._api_server:
            try:
                self._api_server.shutdown()
            except Exception:
                pass
        logger.info("Trading service stopped")

    def signal_outcomes(self) -> Dict[str, Any]:
        path = self.signal_evidence_path
        if not path.exists():
            return {"error": "evidence_missing", "path": str(path)}
        try:
            mtime = path.stat().st_mtime
        except Exception as exc:
            return {"error": "evidence_unreadable", "detail": str(exc)}
        if self._signal_evidence_cache is None or self._signal_evidence_mtime != mtime:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                return {"error": "evidence_invalid", "detail": str(exc)}
            self._signal_evidence_cache = data
            self._signal_evidence_mtime = mtime
        return dict(self._signal_evidence_cache)

    def regime_roc_summary(self) -> Dict[str, Any]:
        path = self.roc_summary_path
        if not path.exists():
            return {"error": "roc_summary_missing", "path": str(path)}
        try:
            mtime = path.stat().st_mtime
        except Exception as exc:
            return {"error": "roc_summary_unreadable", "detail": str(exc)}
        if self._roc_summary_cache is None or self._roc_summary_mtime != mtime:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                return {"error": "roc_summary_invalid", "detail": str(exc)}
            self._roc_summary_cache = data
            self._roc_summary_mtime = mtime
        return dict(self._roc_summary_cache)

    def bundle_outcomes(self) -> Dict[str, Any]:
        path = self.bundle_evidence_path
        if not path.exists():
            return {"error": "bundle_evidence_missing", "path": str(path)}
        try:
            mtime = path.stat().st_mtime
        except Exception as exc:
            return {"error": "bundle_evidence_unreadable", "detail": str(exc)}
        if self._bundle_evidence_cache is None or self._bundle_evidence_mtime != mtime:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                return {"error": "bundle_evidence_invalid", "detail": str(exc)}
            self._bundle_evidence_cache = data
            self._bundle_evidence_mtime = mtime
        return dict(self._bundle_evidence_cache)


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEP trading service")
    parser.add_argument("--read-only", action="store_true", help="disable live order placement")
    parser.add_argument("--pairs", type=str, help="comma separated instrument whitelist")
    parser.add_argument("--no-api", action="store_true", help="do not start HTTP API server")
    return parser.parse_args()


def _install_signal_handlers(service: TradingService) -> None:
    def _shutdown(signum, frame):  # type: ignore[override]
        logger.info("Signal %s received; shutting down", signum)
        service.stop()
        service._shutdown = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _shutdown)
        except Exception:  # pragma: no cover
            pass


def main() -> int:
    args = _parse_cli()
    pairs = args.pairs.split(",") if args.pairs else None
    service = TradingService(read_only=args.read_only, enabled_pairs=pairs)
    _install_signal_handlers(service)
    service.start(start_api=not args.no_api)
    logger.info("Service running. Press Ctrl+C to stop.")
    try:
        while not service._shutdown:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
