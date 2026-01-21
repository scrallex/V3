"""Codec utilities for converting market microstructure windows into structural manifolds.

The goal is to mirror the structural manifold metrics described in ``stm/docs/manifold_vs_optical``
while operating directly on price/volume windows.  Each short window is represented by a reversible
bit plane and scored with the (q, φ, h, λ) tuple plus ancillary canonical features so downstream
gates can reason about profitable regimes.
"""

from __future__ import annotations

import base64
import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BITS_PER_CANDLE = 8  # direction (1) + |Δ| bucket (3) + ATR bucket (2) + liquidity (1) + volume flag (1)


@dataclass
class Candle:
    """Minimal candle representation used by the codec."""

    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: Optional[float] = None


@dataclass
class CanonicalFeatures:
    realized_vol: float
    atr_mean: float
    autocorr: float
    trend_strength: float
    volume_zscore: float
    regime: str
    regime_confidence: float


@dataclass
class EncodedWindow:
    instrument: str
    start_ms: int
    end_ms: int
    bits: bytes
    bit_length: int
    signature: str
    metrics: Dict[str, float]
    canonical: CanonicalFeatures
    codec_meta: Dict[str, float]

    def bits_b64(self) -> str:
        return base64.b64encode(self.bits).decode("ascii")

    def to_json(self) -> Dict[str, object]:
        return {
            "instrument": self.instrument,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "bits_b64": self.bits_b64(),
            "bit_length": self.bit_length,
            "signature": self.signature,
            "metrics": self.metrics,
            "canonical": self.canonical.__dict__,
            "codec_meta": self.codec_meta,
        }


def decode_window_bits(window: EncodedWindow) -> List[Dict[str, float]]:
    """Reconstruct the bucket-level signals for inspection."""

    bit_values = _bytes_to_bits(window.bits, window.bit_length)
    records: List[Dict[str, float]] = []
    idx = 0
    delta_scale = window.codec_meta.get("delta_scale", 1.0)
    atr_scale = window.codec_meta.get("atr_scale", 1.0)
    volume_split = window.codec_meta.get("volume_split", 1.0)

    while idx + BITS_PER_CANDLE <= len(bit_values):
        direction = bit_values[idx]
        delta_bucket = _bits_to_int(bit_values[idx + 1 : idx + 4])
        atr_bucket = _bits_to_int(bit_values[idx + 4 : idx + 6])
        liquidity_flag = bit_values[idx + 6]
        volume_flag = bit_values[idx + 7]
        idx += BITS_PER_CANDLE

        reconstructed_delta = ((delta_bucket + 0.5) / 8.0) * delta_scale
        reconstructed_range = ((atr_bucket + 0.5) / 4.0) * atr_scale
        reconstructed_volume = volume_split * (1.25 if volume_flag else 0.75)

        records.append(
            {
                "direction": direction,
                "abs_delta_est": reconstructed_delta,
                "atr_ratio_est": reconstructed_range,
                "liquidity_flag": liquidity_flag,
                "volume_est": reconstructed_volume,
            }
        )
    return records


class MarketManifoldCodec:
    """Convert rolling candle windows into reversible structural manifolds."""

    def __init__(
        self,
        *,
        window_candles: int = 64,
        stride_candles: int = 16,
        atr_period: int = 14,
    ) -> None:
        if window_candles < 8:
            raise ValueError("window_candles must be >= 8")
        if stride_candles < 1:
            raise ValueError("stride_candles must be >= 1")
        self.window_candles = window_candles
        self.stride_candles = stride_candles
        self.atr_period = atr_period

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(
        self,
        candles: Sequence[Candle],
        *,
        instrument: str,
    ) -> List[EncodedWindow]:
        if len(candles) < self.window_candles:
            return []

        atr_series = _ema_true_range(candles, period=self.atr_period)
        log_returns = _log_returns(candles)
        spread_values = [c.spread if c.spread and c.spread > 0 else c.high - c.low for c in candles]
        spread_median = statistics.median(spread_values) if spread_values else 0.0
        volume_values = [max(1e-12, c.volume) for c in candles]
        volume_median = statistics.median(volume_values) if volume_values else 1.0

        windows: List[EncodedWindow] = []
        start = 0
        while start + self.window_candles <= len(candles):
            end = start + self.window_candles
            subset = candles[start:end]
            subset_atr = atr_series[start:end]
            subset_returns = log_returns[start + 1 : end]  # aligned with closes
            bits, meta = self._encode_window_bits(
                subset,
                subset_atr,
                volume_median,
                spread_median,
                prev_close=candles[start - 1].close if start > 0 else subset[0].open,
            )
            metrics = _structural_metrics(bits)
            canonical = self._canonical_features(subset, subset_returns, subset_atr, volume_median)
            signature = _signature(metrics)
            windows.append(
                EncodedWindow(
                    instrument=instrument,
                    start_ms=subset[0].timestamp_ms,
                    end_ms=subset[-1].timestamp_ms,
                    bits=_bits_to_bytes(bits),
                    bit_length=len(bits),
                    signature=signature,
                    metrics=metrics,
                    canonical=canonical,
                    codec_meta=meta,
                )
            )
            start += self.stride_candles
        return windows

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _encode_window_bits(
        self,
        subset: Sequence[Candle],
        atr_values: Sequence[float],
        volume_split: float,
        spread_split: float,
        *,
        prev_close: float,
    ) -> Tuple[List[int], Dict[str, float]]:
        bits: List[int] = []
        last_close = prev_close
        for candle, atr in zip(subset, atr_values):
            atr_safe = max(atr, 1e-8)
            delta = candle.close - last_close
            direction = 1 if delta >= 0 else 0
            abs_ratio = min(1.0, abs(delta) / atr_safe)
            delta_bucket = min(7, int(round(abs_ratio * 7)))

            tr = max(candle.high - candle.low, abs(candle.high - last_close), abs(candle.low - last_close))
            tr_ratio = min(1.0, tr / max(atr_safe, 1e-8))
            atr_bucket = min(3, int(round(tr_ratio * 3)))

            spread_value = candle.spread if candle.spread and candle.spread > 0 else candle.high - candle.low
            liquidity_flag = 1 if spread_split <= 0 or spread_value <= spread_split else 0

            volume_flag = 1 if candle.volume >= volume_split else 0

            bits.extend(_int_to_bits(direction, 1))
            bits.extend(_int_to_bits(delta_bucket, 3))
            bits.extend(_int_to_bits(atr_bucket, 2))
            bits.extend(_int_to_bits(liquidity_flag, 1))
            bits.extend(_int_to_bits(volume_flag, 1))

            last_close = candle.close

        meta = {
            "delta_scale": float(statistics.fmean(atr_values)) if atr_values else 1.0,
            "atr_scale": float(statistics.fmean(atr_values)) if atr_values else 1.0,
            "volume_split": float(volume_split),
            "spread_split": float(spread_split),
        }
        return bits, meta

    def _canonical_features(
        self,
        subset: Sequence[Candle],
        returns: Sequence[float],
        atr_values: Sequence[float],
        volume_split: float,
    ) -> CanonicalFeatures:
        if len(subset) < 2:
            return CanonicalFeatures(0.0, 0.0, 0.0, 0.0, 0.0, "insufficient", 0.0)

        realized_vol = statistics.pstdev(returns) if len(returns) >= 2 else 0.0
        atr_mean = statistics.fmean(atr_values) if atr_values else 0.0
        autocorr = _lag1_autocorr(returns)

        xs = list(range(len(subset)))
        closes = [c.close for c in subset]
        slope = _ols_slope(xs, closes)
        volatility = statistics.pstdev(closes) if len(closes) >= 2 else 1e-8
        trend_strength = slope / max(1e-8, volatility)

        volume_avg = statistics.fmean(c.volume for c in subset) if subset else 0.0
        volume_zscore = (volume_avg - volume_split) / max(1e-6, volume_split)

        regime, confidence = _classify_regime(trend_strength, autocorr, realized_vol, atr_mean)

        return CanonicalFeatures(
            realized_vol=realized_vol,
            atr_mean=atr_mean,
            autocorr=autocorr,
            trend_strength=trend_strength,
            volume_zscore=volume_zscore,
            regime=regime,
            regime_confidence=confidence,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _ema_true_range(candles: Sequence[Candle], *, period: int) -> List[float]:
    if not candles:
        return []
    alpha = 2.0 / (period + 1.0)
    atr: List[float] = []
    prev_close = candles[0].close
    ema = candles[0].high - candles[0].low
    for candle in candles:
        tr = max(candle.high - candle.low, abs(candle.high - prev_close), abs(candle.low - prev_close))
        ema = (alpha * tr) + (1 - alpha) * ema
        atr.append(max(ema, 1e-8))
        prev_close = candle.close
    return atr


def _log_returns(candles: Sequence[Candle]) -> List[float]:
    rets: List[float] = [0.0]
    for idx in range(1, len(candles)):
        prev = candles[idx - 1].close
        curr = candles[idx].close
        if prev <= 0 or curr <= 0:
            rets.append(0.0)
            continue
        rets.append(math.log(curr / prev))
    return rets


def _structural_metrics(bits: Sequence[int]) -> Dict[str, float]:
    if not bits:
        return {
            "coherence": 0.0,
            "stability": 0.0,
            "entropy": 0.0,
            "hazard": 0.0,
            "rupture": 0.0,
            "coherence_tau_1": 1.0,
            "coherence_tau_4": 1.0,
            "coherence_tau_slope": 0.0,
            "domain_wall_ratio": 0.0,
            "domain_wall_slope": 0.0,
            "spectral_lowf_share": 0.0,
            "reynolds_ratio": 0.0,
            "temporal_half_life": 0.0,
            "spatial_corr_length": 0.0,
            "pinned_alignment": 1.0,
        }

    ones = sum(bits)
    total = len(bits)
    p1 = ones / total
    p0 = 1.0 - p1
    entropy = 0.0
    for p in (p0, p1):
        if p > 0:
            entropy -= p * math.log2(p)
    coherence = 1.0 - min(entropy, 1.0)

    transitions = sum(1 for i in range(1, total) if bits[i] != bits[i - 1])
    rupture = transitions / max(1, total - 1)
    stability = 1.0 - rupture
    hazard = rupture

    coh_tau_1 = _coherence_tau(bits, 1)
    coh_tau_4 = _coherence_tau(bits, 4)
    coh_tau_slope = (coh_tau_4 - coh_tau_1) / 3.0
    wall_ratio, wall_slope = _domain_wall_stats(bits)
    spectral_low = _low_frequency_share(bits)
    reynolds = abs(rupture) / max(1e-6, abs(wall_slope)) if wall_slope else 0.0
    temporal_half_life = math.log(2.0) / max(1e-6, abs(coh_tau_slope)) if coh_tau_slope else 0.0
    spatial_corr_length = 1.0 / max(1e-6, abs(wall_slope)) if wall_slope else 0.0

    return {
        "coherence": round(coherence, 6),
        "stability": round(stability, 6),
        "entropy": round(entropy, 6),
        "hazard": round(hazard, 6),
        "rupture": round(rupture, 6),
        "coherence_tau_1": round(coh_tau_1, 6),
        "coherence_tau_4": round(coh_tau_4, 6),
        "coherence_tau_slope": round(coh_tau_slope, 6),
        "domain_wall_ratio": round(wall_ratio, 6),
        "domain_wall_slope": round(wall_slope, 6),
        "spectral_lowf_share": round(spectral_low, 6),
        "reynolds_ratio": round(reynolds, 6),
        "temporal_half_life": round(temporal_half_life, 6),
        "spatial_corr_length": round(spatial_corr_length, 6),
        "pinned_alignment": 1.0,
    }


def _signature(metrics: Dict[str, float]) -> str:
    coh = metrics.get("coherence", 0.0)
    stab = metrics.get("stability", 0.0)
    ent = metrics.get("entropy", 0.0)
    return f"c{coh:.3f}_s{stab:.3f}_e{ent:.3f}"


def _int_to_bits(value: int, width: int) -> List[int]:
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def _bits_to_int(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (bit & 1)
    return value


def _bits_to_bytes(bits: Sequence[int]) -> bytes:
    buf = bytearray()
    for idx in range(0, len(bits), 8):
        chunk = bits[idx : idx + 8]
        value = 0
        for bit in chunk:
            value = (value << 1) | (bit & 1)
        value <<= max(0, 8 - len(chunk))
        buf.append(value & 0xFF)
    return bytes(buf)


def _bytes_to_bits(data: bytes, bit_length: int) -> List[int]:
    bits: List[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
            if len(bits) == bit_length:
                return bits
    return bits[:bit_length]


def _lag1_autocorr(series: Sequence[float]) -> float:
    if len(series) < 2:
        return 0.0
    mean = statistics.fmean(series)
    num = 0.0
    denom = 0.0
    for idx in range(1, len(series)):
        x0 = series[idx - 1] - mean
        x1 = series[idx] - mean
        num += x0 * x1
        denom += x0 * x0
    return num / denom if denom else 0.0


def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom = sum((x - mean_x) ** 2 for x in xs)
    return num / denom if denom else 0.0


def _classify_regime(
    trend_strength: float,
    autocorr: float,
    realized_vol: float,
    atr_mean: float,
) -> Tuple[str, float]:
    if atr_mean <= 0:
        atr_mean = 1e-6
    vol_ratio = realized_vol / max(1e-6, atr_mean)
    if trend_strength >= 1.5 and autocorr >= 0.1:
        return "trend_bull", min(1.0, trend_strength / 3.0)
    if trend_strength <= -1.5 and autocorr >= 0.1:
        return "trend_bear", min(1.0, abs(trend_strength) / 3.0)
    if vol_ratio < 0.75 and abs(autocorr) < 0.25:
        return "mean_revert", 1.0 - vol_ratio
    if vol_ratio >= 1.5 and abs(autocorr) < 0.1:
        return "chaotic", min(1.0, vol_ratio / 3.0)
    return "neutral", 0.5


def _coherence_tau(bits: Sequence[int], tau: int) -> float:
    if tau <= 0 or len(bits) <= tau:
        return 1.0
    matches = 0
    comparisons = 0
    for idx in range(tau, len(bits)):
        comparisons += 1
        if bits[idx] == bits[idx - tau]:
            matches += 1
    return matches / comparisons if comparisons else 1.0


def _domain_wall_stats(bits: Sequence[int]) -> Tuple[float, float]:
    if len(bits) < 2:
        return 0.0, 0.0
    total_ratio = _domain_wall_ratio(bits, 0, len(bits))
    mid = len(bits) // 2
    first = _domain_wall_ratio(bits, 0, mid)
    second = _domain_wall_ratio(bits, mid, len(bits))
    return total_ratio, second - first


def _domain_wall_ratio(bits: Sequence[int], begin: int, end: int) -> float:
    if end <= begin + 1:
        return 0.0
    transitions = 0
    for idx in range(begin + 1, end):
        if bits[idx] != bits[idx - 1]:
            transitions += 1
    pairs = (end - begin) - 1
    return transitions / pairs if pairs else 0.0


def _low_frequency_share(bits: Sequence[int]) -> float:
    n = len(bits)
    if n < 4:
        return 0.0
    centred = [float(b) for b in bits]
    mean = sum(centred) / n
    centred = [value - mean for value in centred]
    max_k = n // 2
    if max_k == 0:
        return 0.0
    low_cut = max(1, max_k // 4)
    two_pi_over_n = (2.0 * math.pi) / n
    total_power = 0.0
    low_power = 0.0
    for k in range(1, max_k + 1):
        real = 0.0
        imag = 0.0
        for t, value in enumerate(centred):
            angle = two_pi_over_n * (k * t)
            real += value * math.cos(angle)
            imag += value * math.sin(angle)
        power = real * real + imag * imag
        total_power += power
        if k <= low_cut:
            low_power += power
    if total_power <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, low_power / total_power))


def window_summary(windows: Iterable[EncodedWindow]) -> Dict[str, object]:
    """Utility used by validation scripts to aggregate correlations."""

    rows = list(windows)
    if not rows:
        return {"count": 0}
    coh = [row.metrics["coherence"] for row in rows]
    stab = [row.metrics["stability"] for row in rows]
    ent = [row.metrics["entropy"] for row in rows]
    hazard = [row.metrics["hazard"] for row in rows]
    vol = [row.canonical.realized_vol for row in rows]
    autocorr = [row.canonical.autocorr for row in rows]
    trend = [row.canonical.trend_strength for row in rows]

    def corr(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b) or len(a) < 2:
            return 0.0
        mean_a = statistics.fmean(a)
        mean_b = statistics.fmean(b)
        num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        den = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
        return num / den if den else 0.0

    return {
        "count": len(rows),
        "corr": {
            "coherence_vs_autocorr": corr(coh, autocorr),
            "hazard_vs_vol": corr(hazard, vol),
            "stability_vs_trend": corr(stab, trend),
            "entropy_vs_vol": corr(ent, vol),
        },
        "regime_breakdown": {
            regime: sum(1 for row in rows if row.canonical.regime == regime) / len(rows)
            for regime in {"trend_bull", "trend_bear", "mean_revert", "chaotic", "neutral"}
        },
    }


def windows_to_jsonl(windows: Iterable[EncodedWindow]) -> str:
    """Serialize encoded windows into newline-delimited JSON for inspection."""

    return "\n".join(json.dumps(window.to_json()) for window in windows)
