"""gRPC front-end for the native QFH kernel.

This server wraps the :mod:`sep_quantum` bindings (C++ kernel) and exposes a
single unary RPC that accepts a raw byte window (text or price ticks) and
returns the compact structural signature plus core metrics.
"""

from __future__ import annotations

import logging
import os
import struct
from concurrent import futures
from typing import Optional

import grpc  # type: ignore

from . import native
from .structural_verification_pb2 import StructuralRequest, StructuralResponse

LOGGER = logging.getLogger(__name__)


def _bucket(value: float) -> int:
    return int(round(max(0.0, min(1.0, value)) * 1000.0))


def _signature_bytes(sig_c: int, sig_s: int, sig_e: int, sig_h: int) -> bytes:
    """Pack the four quantised components into 9 bytes (big-endian)."""

    return struct.pack(">HHHHB", sig_c & 0xFFFF, sig_s & 0xFFFF, sig_e & 0xFFFF, sig_h & 0xFFFF, 0)


class StructuralVerifierServicer:
    """Unary servicer that runs the native kernel."""

    def __init__(self) -> None:
        if not native.HAVE_NATIVE:
            raise RuntimeError("sep_quantum native bindings are required to run the structural verifier service")

    def ScoreWindow(self, request: StructuralRequest, context) -> StructuralResponse:  # noqa: N802 - gRPC naming
        payload = bytes(request.payload or b"")
        if not payload:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "payload is required")

        bits = list(native.bits_from_bytes(payload))
        if request.bit_length and request.bit_length > 0 and request.bit_length < len(bits):
            bits = bits[: int(request.bit_length)]

        try:
            metrics = native.analyze_bits(bits)
        except Exception as exc:  # pragma: no cover - runtime path
            LOGGER.exception(
                "Failed to analyse payload: input_type=%s len=%d bit_length=%d",
                request.input_type,
                len(payload),
                request.bit_length,
            )
            context.abort(grpc.StatusCode.INTERNAL, f"analysis failed: {exc}")

        coherence = float(metrics.get("coherence", 0.0))
        stability = float(metrics.get("stability", 0.0))
        entropy = float(metrics.get("entropy", 0.0))
        hazard = float(metrics.get("lambda_hazard", metrics.get("rupture", 0.0)))

        sig_c = _bucket(coherence)
        sig_s = _bucket(stability)
        sig_e = _bucket(entropy)
        sig_h = _bucket(hazard)

        return StructuralResponse(
            signature=_signature_bytes(sig_c, sig_s, sig_e, sig_h),
            hazard=hazard,
            coherence=coherence,
            stability=stability,
            entropy=entropy,
            sig_c=sig_c,
            sig_s=sig_s,
            sig_e=sig_e,
        )


def add_structural_verifier(server: grpc.Server, servicer: StructuralVerifierServicer) -> None:
    handler = grpc.unary_unary_rpc_method_handler(
        servicer.ScoreWindow,
        request_deserializer=StructuralRequest.FromString,
        response_serializer=StructuralResponse.SerializeToString,
    )
    service = grpc.method_handlers_generic_handler(
        "sep.structural.StructuralVerifier",
        {"ScoreWindow": handler},
    )
    server.add_generic_rpc_handlers((service,))


def serve(
    *,
    host: str = "0.0.0.0",
    port: int = 8600,
    max_workers: int = 8,
    grpc_options: Optional[list[tuple[str, object]]] = None,
) -> grpc.Server:
    """Start the structural verifier service and return the running server."""

    options = grpc_options or [
        ("grpc.max_send_message_length", 16 * 1024 * 1024),
        ("grpc.max_receive_message_length", 16 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
    add_structural_verifier(server, StructuralVerifierServicer())
    bind_addr = f"{host}:{port}"
    server.add_insecure_port(bind_addr)
    server.start()
    LOGGER.info("Structural verifier gRPC service listening on %s", bind_addr)
    return server


def main() -> int:  # pragma: no cover - thin CLI wrapper
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    host = os.getenv("STRUCTURAL_RPC_HOST", "0.0.0.0")
    port = int(os.getenv("STRUCTURAL_RPC_PORT", "8600") or 8600)
    max_workers = int(os.getenv("STRUCTURAL_RPC_WORKERS", "8") or 8)
    server = serve(host=host, port=port, max_workers=max_workers)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down structural verifier")
        server.stop(grace=None)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
