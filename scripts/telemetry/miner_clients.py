"""Miner API clients for pulling live hashrate/share metrics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import requests

from .config import MinerAPIConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MinerStats:
    hashrate_5s: Optional[float] = None
    hashrate_1m: Optional[float] = None
    accepted_shares: Optional[int] = None
    rejected_shares: Optional[int] = None
    stale_shares: Optional[int] = None
    net_rtt_ms: Optional[float] = None
    stratum_latency_ms: Optional[float] = None
    submit_latency_ms: Optional[float] = None
    pool_diff: Optional[float] = None
    job_rate_hz: Optional[float] = None

    def apply_defaults(self, other: "MinerStats") -> "MinerStats":
        """Return a copy with ``other`` values used as fallback when ``self`` is None."""

        return MinerStats(
            hashrate_5s=self.hashrate_5s if self.hashrate_5s is not None else other.hashrate_5s,
            hashrate_1m=self.hashrate_1m if self.hashrate_1m is not None else other.hashrate_1m,
            accepted_shares=self.accepted_shares if self.accepted_shares is not None else other.accepted_shares,
            rejected_shares=self.rejected_shares if self.rejected_shares is not None else other.rejected_shares,
            stale_shares=self.stale_shares if self.stale_shares is not None else other.stale_shares,
            net_rtt_ms=self.net_rtt_ms if self.net_rtt_ms is not None else other.net_rtt_ms,
            stratum_latency_ms=self.stratum_latency_ms if self.stratum_latency_ms is not None else other.stratum_latency_ms,
            submit_latency_ms=self.submit_latency_ms if self.submit_latency_ms is not None else other.submit_latency_ms,
            pool_diff=self.pool_diff if self.pool_diff is not None else other.pool_diff,
            job_rate_hz=self.job_rate_hz if self.job_rate_hz is not None else other.job_rate_hz,
        )


class MinerClient:
    """Base class for miner telemetry clients."""

    def fetch(self) -> Optional[MinerStats]:
        raise NotImplementedError


class NullMinerClient(MinerClient):
    """Return no stats when no miner API is configured."""

    def fetch(self) -> Optional[MinerStats]:
        return None


class TrexMinerClient(MinerClient):
    """Fetch telemetry from the T-Rex miner HTTP API."""

    def __init__(self, config: MinerAPIConfig) -> None:
        self._url = config.url.rstrip("/")
        self._timeout = config.timeout
        self._session = requests.Session()

    def fetch(self) -> Optional[MinerStats]:
        try:
            response = self._session.get(f"{self._url}", timeout=self._timeout)
            response.raise_for_status()
        except Exception as exc:
            logger.debug("T-Rex API fetch failed: %s", exc)
            return None

        try:
            data = response.json()
        except json.JSONDecodeError:
            logger.warning("T-Rex API returned non-JSON payload")
            return None

        hashrate_5s = _coerce_float(data.get("hashrate"))
        hashrate_1m = _coerce_float(
            data.get("hashrate_minute") or data.get("hashrate_10s") or data.get("hashrate")
        )

        shares = data.get("shares") or {}
        accepted = _coerce_int(data.get("accepted_count") or shares.get("accepted_count"))
        rejected = _coerce_int(
            data.get("rejected_count")
            or shares.get("rejected_count")
            or shares.get("invalid_count")
        )
        stale = _coerce_int(shares.get("stale_count"))

        latency = data.get("latency") or data.get("avg_latency")
        if isinstance(latency, dict):
            net_rtt_ms = _coerce_float(latency.get("current") or latency.get("avg"))
        else:
            net_rtt_ms = _coerce_float(latency)

        return MinerStats(
            hashrate_5s=hashrate_5s,
            hashrate_1m=hashrate_1m,
            accepted_shares=accepted,
            rejected_shares=rejected,
            stale_shares=stale,
            net_rtt_ms=net_rtt_ms,
            stratum_latency_ms=_coerce_float(data.get("short_rtt") or data.get("ping")),
            submit_latency_ms=_coerce_float(data.get("avg_submit_ms")),
            pool_diff=_coerce_float((data.get("pool") or {}).get("difficulty")),
            job_rate_hz=None,
        )


def _coerce_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def create_miner_client(config: Optional[MinerAPIConfig]) -> MinerClient:
    if not config:
        return NullMinerClient()
    if config.kind == "trex":
        return TrexMinerClient(config)
    logger.warning("Unsupported miner API kind '%s'; falling back to null client", config.kind)
    return NullMinerClient()
