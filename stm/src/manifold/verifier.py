"""Helpers to inspect verification metrics."""

from __future__ import annotations

from typing import Dict


def score_documents(summary: Dict[str, object]) -> Dict[str, float]:
    """Return aggregate verification stats from an evaluate_manifold summary."""

    verification = summary.get("verification", {})
    return {
        "precision": float(verification.get("precision", 0.0)),
        "recall": float(verification.get("recall", 0.0)),
        "false_positive_rate": float(verification.get("false_positive_rate", 0.0)),
    }
