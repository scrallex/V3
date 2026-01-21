"""Adapters for ingesting external datasets into the research stack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional

from scripts.enrich_features import enrich_with_causal_features
from scripts.features import CausalFeatureExtractor


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _step_metrics(step: Mapping[str, object]) -> Dict[str, float]:
    duration_ms = float(step.get("duration_ms", 0) or 0.0)
    duration_ratio = _clamp(duration_ms / 60000.0)
    success = str(step.get("conclusion", "")).lower() == "success"
    coherence = _clamp(1.0 - duration_ratio * 0.6)
    entropy = _clamp(0.2 + duration_ratio * 0.6)
    stability = _clamp(0.5 + (0.25 if success else -0.15))
    hazard = _clamp(duration_ratio if not success else duration_ratio * 0.5)
    return {
        "coherence": coherence,
        "entropy": entropy,
        "stability": stability,
        "lambda_hazard": hazard,
        "rupture": hazard,
    }


def _step_state(step: Mapping[str, object]) -> Dict[str, object]:
    locked = 1 if str(step.get("conclusion", "")).lower() != "success" else 0
    return {"resources": {step.get("name", "step"): {"locked": locked}}}


def _step_dilution(step: Mapping[str, object]) -> Dict[str, float]:
    duration_ms = float(step.get("duration_ms", 0) or 0.0)
    ratio = _clamp(duration_ms / 90000.0)
    path = _clamp(0.3 + ratio * 0.5 if str(step.get("conclusion", "")).lower() != "success" else ratio * 0.4)
    signal = _clamp(ratio * 0.6)
    return {"path": path, "signal": signal}


class RealWorldAdapter:
    """Derive STM-compatible state from operational data feeds."""

    def __init__(self, *, extractor: CausalFeatureExtractor | None = None) -> None:
        self.extractor = extractor or CausalFeatureExtractor()

    def from_github_actions(self, path: str | Path) -> Dict[str, object]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        runs = payload.get("workflow_runs") if isinstance(payload, Mapping) else []
        if not isinstance(runs, list):
            runs = []

        signals: List[MutableMapping[str, object]] = []
        failure_index: Optional[int] = None
        for run in runs:
            steps = run.get("steps") if isinstance(run, Mapping) else []
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, Mapping):
                    continue
                metrics = _step_metrics(step)
                window: MutableMapping[str, object] = {
                    "step": step.get("name"),
                    "metrics": metrics,
                    "dilution": _step_dilution(step),
                    "state": _step_state(step),
                    "features": {},
                    "failure": str(step.get("conclusion", "")).lower() not in {"success", "skipped"},
                    "url": step.get("html_url"),
                }
                features = self.extractor.extract(window, history=signals)
                window["features"]["causal"] = features
                if window["failure"] and failure_index is None:
                    failure_index = len(signals)
                signals.append(window)

        state: Dict[str, object] = {
            "metadata": {
                "domain": "github_actions",
                "source": str(path),
            },
            "signals": signals,
            "failure_index": failure_index if failure_index is not None else -1,
        }
        enrich_with_causal_features(state, extractor=self.extractor)
        return state


__all__ = ["RealWorldAdapter"]
