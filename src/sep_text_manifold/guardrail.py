"""Guardrail calibration helpers for STM logistics demos."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Mapping, Sequence


Number = float | int


def calibrate_threshold(
    values: Sequence[Number],
    *,
    padding: float,
    ceiling: float | None = None,
) -> float:
    """Return a calibrated threshold slightly above the baseline distribution.

    Parameters
    ----------
    values:
        Historical or baseline metric values used to estimate the
        foreground threshold.
    padding:
        Additive margin applied on top of the maximum baseline value.
    ceiling:
        Optional upper bound that clamps the resulting threshold.
    """
    if not values:
        baseline = 0.0
    else:
        baseline = max(float(v) for v in values)
    threshold = baseline + float(padding)
    if ceiling is not None:
        threshold = min(threshold, float(ceiling))
    return threshold


def summarise_guardrail(
    *,
    signals: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    event_index: int,
    hazard_padding: float = 0.0015,
    hazard_ceiling: float | None = 0.95,
    path_padding: float = 0.05,
    path_ceiling: float | None = 1.2,
    signal_padding: float = 0.01,
    signal_ceiling: float | None = None,
) -> Dict[str, Any]:
    """Aggregate per-step metrics and derive calibrated alert thresholds.

    The helper returns the per-step metric rows, calibrated thresholds, and
    lead-time measurements that downstream dashboards and docs can reuse.
    """
    limit = min(len(signals), len(transitions))
    rows: list[Dict[str, Any]] = []
    for idx in range(limit):
        signal = signals[idx]
        transition = transitions[idx]
        metrics: Mapping[str, Any]
        metrics = signal.get("metrics") if isinstance(signal.get("metrics"), Mapping) else {}
        lambda_hazard = float(signal.get("lambda_hazard", metrics.get("lambda_hazard", metrics.get("rupture", 0.0))))
        coherence = float(metrics.get("coherence", signal.get("coherence", 0.0)))
        stability = float(metrics.get("stability", signal.get("stability", 0.0)))
        entropy = float(metrics.get("entropy", signal.get("entropy", 0.0)))
        dilution_payload = signal.get("dilution")
        path_dilution = 0.0
        signal_dilution = 0.0
        if isinstance(dilution_payload, Mapping):
            path_dilution = float(dilution_payload.get("path", 0.0))
            signal_dilution = float(dilution_payload.get("signal", 0.0))
        signature = signal.get("signature")
        try:
            window_id = int(signal.get("id", signal.get("index", idx)))
        except (TypeError, ValueError):
            window_id = idx
        row: Dict[str, Any] = {
            "step": idx,
            "action": transition.get("action"),
            "status": transition.get("status", "valid"),
            "lambda_hazard": lambda_hazard,
            "coherence": coherence,
            "stability": stability,
            "entropy": entropy,
            "path_dilution": path_dilution,
            "signal_dilution": signal_dilution,
            "errors": transition.get("errors", []),
            "note": transition.get("note"),
            "window_id": window_id,
            "signature": signature if isinstance(signature, str) else None,
        }
        rows.append(row)

    baseline_rows = [row for row in rows if row["step"] <= event_index]
    baseline_hazard = [row["lambda_hazard"] for row in baseline_rows]
    baseline_path = [row["path_dilution"] for row in baseline_rows]
    baseline_signal = [row["signal_dilution"] for row in baseline_rows]

    hazard_threshold = calibrate_threshold(baseline_hazard, padding=hazard_padding, ceiling=hazard_ceiling)
    path_threshold = calibrate_threshold(baseline_path, padding=path_padding, ceiling=path_ceiling)
    signal_threshold = calibrate_threshold(baseline_signal, padding=signal_padding, ceiling=signal_ceiling)

    alert_steps: list[int] = []
    for row in rows:
        hazard_alert = row["lambda_hazard"] >= hazard_threshold
        path_alert = row["path_dilution"] >= path_threshold
        signal_alert = row["signal_dilution"] >= signal_threshold
        row["hazard_alert"] = hazard_alert
        row["dilution_alert"] = path_alert or signal_alert
        row["alert"] = hazard_alert or row["dilution_alert"]
        if row["alert"]:
            alert_steps.append(row["step"])

    failure_steps = [row["step"] for row in rows if row.get("status") != "valid"]
    first_alert = alert_steps[0] if alert_steps else None
    first_failure = failure_steps[0] if failure_steps else None
    lead_time = None
    if first_alert is not None and first_failure is not None:
        lead_time = max(0, first_failure - first_alert)

    summary = {
        "rows": rows,
        "thresholds": {
            "lambda_hazard": hazard_threshold,
            "path_dilution": path_threshold,
            "signal_dilution": signal_threshold,
        },
        "alert_steps": alert_steps,
        "failure_steps": failure_steps,
        "first_alert": first_alert,
        "first_failure": first_failure,
        "lead_time": lead_time,
        "baseline": {
            "hazard_mean": mean(baseline_hazard) if baseline_hazard else 0.0,
            "path_mean": mean(baseline_path) if baseline_path else 0.0,
            "signal_mean": mean(baseline_signal) if baseline_signal else 0.0,
        },
    }
    return summary
