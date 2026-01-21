"""Helpers for optional native (C++) accelerated routines.

This module wraps the :mod:`sep_quantum` pybind11 bindings so callers in the
STM codebase can consume the full manifold metrics whenever the native kernel
is available.  Each helper raises ``RuntimeError`` if the extension has not
been built, allowing callers to fall back to the pure-Python implementations.
"""

from __future__ import annotations

import os
import struct
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import grpc  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    grpc = None

try:  # pragma: no cover - optional dependency
    from sep_quantum import (  # type: ignore
        QFHAggregateEvent as _QFHAggregateEvent,
        QFHEvent as _QFHEvent,
        QFHResult as _QFHResult,
        QFHState as _QFHState,
        aggregate_events as _native_aggregate_events,
        analyze_bits as _native_analyze_bits,
        analyze_window as _native_analyze_window,
        analyze_window_batch as _native_analyze_window_batch,
        transform_rich as _native_transform_rich,
    )
    HAVE_NATIVE = True
except ImportError:  # pragma: no cover - optional dependency
    _native_aggregate_events = None
    _native_analyze_bits = None
    _native_analyze_window = None
    _native_analyze_window_batch = None
    _native_transform_rich = None
    _QFHAggregateEvent = None
    _QFHEvent = None
    _QFHResult = None
    _QFHState = None
    HAVE_NATIVE = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sep_quantum import QFHAggregateEvent, QFHEvent, QFHResult, QFHState
else:
    QFHAggregateEvent = _QFHAggregateEvent  # type: ignore
    QFHEvent = _QFHEvent  # type: ignore
    QFHResult = _QFHResult  # type: ignore
    QFHState = _QFHState  # type: ignore

__all__ = [
    "HAVE_NATIVE",
    "QFHAggregateEvent",
    "QFHEvent",
    "QFHResult",
    "QFHState",
    "aggregate_events",
    "analyze_bits",
    "analyze_window",
    "analyze_window_batch",
    "bits_from_bytes",
    "grpc_enabled",
    "score_bytes",
    "set_grpc_target",
    "set_use_native",
    "transform_rich",
    "use_native",
]


_USE_NATIVE = False
_GRPC_TARGET = os.getenv("STRUCTURAL_RPC_TARGET")
_GRPC_STUBS: dict[str, object] = {}


def _require_native(name: str) -> None:
    if not HAVE_NATIVE:
        raise RuntimeError("Native quantum bindings are not available")


def grpc_enabled() -> bool:
    """Return ``True`` when grpc is importable (client stubs can run)."""

    return grpc is not None


def analyze_bits(bits: Sequence[int]) -> dict[str, float]:
    """Return aggregate metrics for ``bits`` using the native kernel."""

    _require_native("analyze_bits")
    if _native_analyze_bits is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Native build missing analyze_bits")
    metrics = _native_analyze_bits(list(bits))
    return {
        "coherence": float(metrics.coherence),
        "stability": float(metrics.stability),
        "entropy": float(metrics.entropy),
        "rupture": float(metrics.rupture),
        "lambda_hazard": float(metrics.lambda_hazard),
        "sig_c": int(metrics.sig_c),
        "sig_s": int(metrics.sig_s),
        "sig_e": int(metrics.sig_e),
    }


def analyze_window(bits: Sequence[int]):
    """Return the full :class:`sep_quantum.QFHResult` for ``bits``."""

    _require_native("analyze_window")
    if _native_analyze_window is None:
        raise RuntimeError("Native build missing analyze_window")
    return _native_analyze_window(list(bits))


def analyze_window_batch(windows: Sequence[bytes]) -> list[dict[str, float]]:
    """Return aggregate metrics for a batch of byte windows."""

    _require_native("analyze_window_batch")
    if _native_analyze_window_batch is None:
        raise RuntimeError("Native build missing analyze_window_batch")
    metrics_batch = _native_analyze_window_batch(list(windows))
    results: list[dict[str, float]] = []
    for metrics in metrics_batch:
        results.append(
            {
                "coherence": float(metrics.coherence),
                "stability": float(metrics.stability),
                "entropy": float(metrics.entropy),
                "rupture": float(metrics.rupture),
                "lambda_hazard": float(metrics.lambda_hazard),
            }
        )
    return results


def transform_rich(bits: Sequence[int]):  # type: ignore[override]
    """Return the sequence of :class:`QFHEvent` instances for ``bits``."""

    _require_native("transform_rich")
    if _native_transform_rich is None:
        raise RuntimeError("Native build missing transform_rich")
    return list(_native_transform_rich(list(bits)))


def aggregate_events(events):  # type: ignore[override]
    """Aggregate events into :class:`QFHAggregateEvent` spans."""

    _require_native("aggregate_events")
    if _native_aggregate_events is None:
        raise RuntimeError("Native build missing aggregate_events")
    return list(_native_aggregate_events(list(events)))


def bits_from_bytes(data: bytes) -> Iterable[int]:
    for byte in data:
        for shift in range(7, -1, -1):
            yield (byte >> shift) & 1


def set_use_native(enabled: bool) -> None:
    """Toggle whether high-level helpers should prefer the native kernel."""

    global _USE_NATIVE
    _USE_NATIVE = bool(enabled) and HAVE_NATIVE


def use_native() -> bool:
    """Return ``True`` when the native engine should be used by defaults."""

    return HAVE_NATIVE and _USE_NATIVE


# =============================================================================
# gRPC client helper
# =============================================================================


def _bucket(value: float) -> int:
    return int(round(max(0.0, min(1.0, value)) * 1000.0))


def _signature_bytes(sig_c: int, sig_s: int, sig_e: int, sig_h: int) -> bytes:
    return struct.pack(">HHHHB", sig_c & 0xFFFF, sig_s & 0xFFFF, sig_e & 0xFFFF, sig_h & 0xFFFF, 0)


def set_grpc_target(target: str | None) -> None:
    """Update the default structural verifier endpoint (e.g., ``localhost:8600``)."""

    global _GRPC_TARGET, _GRPC_STUBS
    _GRPC_TARGET = target
    _GRPC_STUBS = {}


def _grpc_stub(target: str):
    if grpc is None:
        raise RuntimeError("grpc is not available; install grpcio to use the remote verifier")
    if target in _GRPC_STUBS:
        return _GRPC_STUBS[target]
    from .structural_verification_pb2 import StructuralRequest, StructuralResponse

    channel = grpc.insecure_channel(target)
    call = channel.unary_unary(
        "/sep.structural.StructuralVerifier/ScoreWindow",
        request_serializer=StructuralRequest.SerializeToString,
        response_deserializer=StructuralResponse.FromString,
    )
    _GRPC_STUBS[target] = call
    return call


def _local_score_bytes(payload: bytes, *, bit_length: int | None = None) -> dict[str, object]:
    bits = list(bits_from_bytes(payload))
    if bit_length is not None and bit_length > 0 and bit_length < len(bits):
        bits = bits[:bit_length]
    metrics = analyze_bits(bits)
    coherence = float(metrics.get("coherence", 0.0))
    stability = float(metrics.get("stability", 0.0))
    entropy = float(metrics.get("entropy", 0.0))
    hazard = float(metrics.get("lambda_hazard", metrics.get("rupture", 0.0)))
    sig_c = int(metrics.get("sig_c", _bucket(coherence)))
    sig_s = int(metrics.get("sig_s", _bucket(stability)))
    sig_e = int(metrics.get("sig_e", _bucket(entropy)))
    signature = _signature_bytes(sig_c, sig_s, sig_e, _bucket(hazard))
    return {
        "coherence": coherence,
        "stability": stability,
        "entropy": entropy,
        "hazard": hazard,
        "signature": signature,
        "sig_c": sig_c,
        "sig_s": sig_s,
        "sig_e": sig_e,
    }


def score_bytes(
    payload: bytes,
    *,
    target: str | None = None,
    prefer_grpc: bool = True,
    timeout: float | None = 1.0,
    input_type: str | None = None,
    bit_length: int | None = None,
) -> Mapping[str, object]:
    """Score a raw byte window via the shared gRPC service or local kernel.

    When ``prefer_grpc`` is True and a target is configured (via argument or
    ``STRUCTURAL_RPC_TARGET``), the request is sent to the remote verifier and
    falls back to local native execution on RPC failure.
    """

    target_host = target or _GRPC_TARGET
    last_error: Exception | None = None
    if prefer_grpc and target_host:
        try:
            stub = _grpc_stub(target_host)
            from .structural_verification_pb2 import StructuralRequest

            request = StructuralRequest(payload=payload or b"", input_type=input_type or "", bit_length=int(bit_length or 0))
            response = stub(request, timeout=timeout)
            return {
                "coherence": float(response.coherence),
                "stability": float(response.stability),
                "entropy": float(response.entropy),
                "hazard": float(response.hazard),
                "signature": bytes(response.signature),
                "sig_c": int(response.sig_c),
                "sig_s": int(response.sig_s),
                "sig_e": int(response.sig_e),
            }
        except Exception as exc:  # pragma: no cover - network/availability
            last_error = exc

    try:
        return _local_score_bytes(payload, bit_length=bit_length)
    except Exception as exc:
        if last_error:
            raise RuntimeError(f"gRPC scoring failed ({last_error}) and local scoring failed") from exc
        raise
