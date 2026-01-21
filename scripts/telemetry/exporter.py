#!/usr/bin/env python3
"""Prometheus exporter for telemetry optimizer metrics."""

from __future__ import annotations

import json
import time
from typing import Dict, List

from prometheus_client import Counter, Gauge, start_http_server

from .runtime import bootstrap_from_yaml


def _setup_gauges() -> Dict[str, Gauge]:
    gauges = {
        "sep_score": Gauge("telemetry_sep_score", "SEP score", ["rig"]),
        "lambda_hazard": Gauge("telemetry_lambda_hazard", "Lambda hazard", ["rig"]),
        "rupture_flag": Gauge("telemetry_rupture_flag", "Rupture flag encoded as 0-3", ["rig"]),
        "efficiency": Gauge("telemetry_efficiency_gh_per_j", "Efficiency (GH/J)", ["rig"]),
        "reject_rate": Gauge("telemetry_reject_rate", "Reject rate", ["rig"]),
        "thermal_headroom": Gauge("telemetry_thermal_headroom", "Thermal headroom (C)", ["rig"]),
        "latency_budget": Gauge("telemetry_latency_budget_ms", "Latency budget (ms)", ["rig"]),
        "temp_core": Gauge("telemetry_temp_core_c", "Core temperature (C)", ["rig"]),
        "power": Gauge("telemetry_power_w", "Power draw (W)", ["rig"]),
        "autotune": Gauge("telemetry_autotune_enabled", "Auto-tune controller status (1=enabled)", ["rig"]),
    }
    return gauges


def _setup_rupture_gauge() -> Gauge:
    return Gauge("telemetry_rupture_class", "Rupture class flag", ["rig", "class"])


def _setup_action_counter() -> Counter:
    return Counter("telemetry_action_total", "Auto-tune actions applied", ["rig", "actuator", "delta"])


def _flag_to_int(flag: str) -> int:
    return {"none": 0, "micro": 1, "meso": 2, "macro": 3}.get(flag, -1)


def run_exporter(config_path: str, *, port: int = 9108) -> None:
    pipeline = bootstrap_from_yaml(config_path)
    gauges = _setup_gauges()
    rupture_gauge = _setup_rupture_gauge()
    action_counter = _setup_action_counter()
    start_http_server(port)

    pipeline.start()
    try:
        while True:
            for rig_id in pipeline.collectors:
                score = pipeline._latest_scores.get(rig_id)  # pylint: disable=protected-access
                collector = pipeline.collectors[rig_id]
                if not score:
                    continue
                labels = {"rig": rig_id}
                gauges["sep_score"].labels(**labels).set(score.sep_score)
                gauges["lambda_hazard"].labels(**labels).set(score.lambda_hazard)
                gauges["rupture_flag"].labels(**labels).set(_flag_to_int(score.rupture_flag))
                gauges["efficiency"].labels(**labels).set(score.efficiency)
                gauges["reject_rate"].labels(**labels).set(score.reject_rate)
                gauges["thermal_headroom"].labels(**labels).set(score.thermal_headroom)
                gauges["latency_budget"].labels(**labels).set(score.latency_budget_ms)
                gauges["autotune"].labels(**labels).set(1.0 if pipeline.autotune_enabled else 0.0)

                snapshot = collector.latest()
                if snapshot:
                    gauges["temp_core"].labels(**labels).set(snapshot.temp_core_c)
                    gauges["power"].labels(**labels).set(snapshot.power_w)

                for cls in ("none", "micro", "meso", "macro"):
                    rupture_gauge.labels(rig=rig_id, class=cls).set(1 if score.rupture_flag == cls else 0)

            for event in pipeline.pop_action_events():
                action_counter.labels(
                    rig=event.rig_id,
                    actuator=event.actuator,
                    delta=f"{event.delta:+.1f}",
                ).inc()
            time.sleep(2.0)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run telemetry Prometheus exporter")
    parser.add_argument("--config", default="config/telemetry.yaml", help="Path to telemetry YAML config")
    parser.add_argument("--port", type=int, default=9108, help="HTTP port for metrics")
    args = parser.parse_args()
    run_exporter(args.config, port=args.port)
