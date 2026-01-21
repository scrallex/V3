#!/usr/bin/env python3
"""Lightweight mock miner HTTP endpoint (T-Rex compatible).

Run with::

    python -m scripts.telemetry.mock_miner --port 4067

The server exposes ``/summary`` with a subset of the T-Rex API fields, so the
telemetry collector can ingest realistic hashrate/share statistics while no
real miner is running.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict


@dataclass
class MinerState:
    hashrate_gh: float
    accepted: float
    rejected: float
    stale: float
    latency_ms: float
    submit_ms: float
    pool_diff: float
    job_rate_hz: float

    lock: threading.Lock

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "hashrate": self.hashrate_gh * 1_000_000_000.0,
                "hashrate_minute": self.hashrate_gh * 1_000_000_000.0,
                "accepted_count": int(self.accepted),
                "rejected_count": int(self.rejected),
                "shares": {
                    "accepted_count": int(self.accepted),
                    "rejected_count": int(self.rejected),
                    "stale_count": int(self.stale),
                },
                "latency": {"avg": self.latency_ms},
                "avg_latency": self.latency_ms,
                "ping": self.latency_ms,
                "avg_submit_ms": self.submit_ms,
                "pool": {"difficulty": self.pool_diff},
                "job_rate_hz": self.job_rate_hz,
            }

    def tick(self, dt: float) -> None:
        with self.lock:
            # Hashrate noise (smooth sine + random jitter)
            t = time.time()
            base = self.hashrate_gh
            noise = math.sin(t / 30.0) * 0.02 * base
            rand = random.uniform(-0.01, 0.01) * base
            self.hashrate_gh = max(0.0, base + noise + rand)

            shares_per_sec = self.hashrate_gh / 1_000.0  # arbitrary scale
            accepted_inc = max(0.0, random.gauss(shares_per_sec * dt, 0.05))
            rejected_inc = max(0.0, random.gauss(shares_per_sec * 0.002 * dt, 0.01))
            stale_inc = max(0.0, random.gauss(shares_per_sec * 0.001 * dt, 0.01))

            self.accepted += accepted_inc
            self.rejected += rejected_inc
            self.stale += stale_inc

            # Latency wiggle
            self.latency_ms = max(10.0, self.latency_ms + random.uniform(-1.0, 1.0))
            self.submit_ms = max(5.0, self.submit_ms + random.uniform(-0.5, 0.5))


class MinerRequestHandler(BaseHTTPRequestHandler):
    state: MinerState  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - silence
        return

    def do_GET(self) -> None:  # pragma: no cover - exercised via manual run
        if self.path not in {"/summary", "/summary/"}:
            self.send_response(404)
            self.end_headers()
            return

        payload = self.state.snapshot()
        body = json.dumps(payload).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_tick_loop(state: MinerState, interval: float) -> None:
    while True:
        time.sleep(interval)
        state.tick(interval)


def serve(port: int, initial_hashrate_gh: float) -> None:
    state = MinerState(
        hashrate_gh=initial_hashrate_gh,
        accepted=0.0,
        rejected=0.0,
        stale=0.0,
        latency_ms=85.0,
        submit_ms=12.0,
        pool_diff=5.5,
        job_rate_hz=0.8,
        lock=threading.Lock(),
    )

    tick_thread = threading.Thread(target=run_tick_loop, args=(state, 1.0), daemon=True)
    tick_thread.start()

    handler = MinerRequestHandler
    handler.state = state

    with HTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Mock miner listening on http://127.0.0.1:{port}/summary")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopping mock miner...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock miner HTTP endpoint")
    parser.add_argument("--port", type=int, default=4067, help="Port to bind (default: 4067)")
    parser.add_argument("--hashrate-gh", type=float, default=150.0, help="Initial hashrate in GH/s")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve(args.port, args.hashrate_gh)


if __name__ == "__main__":
    main()
