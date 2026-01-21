"""Lightweight Flask demo exposing manifold discovery endpoints."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request

from .propose import load_state, propose_from_state

app = Flask(__name__)


@functools.lru_cache(maxsize=8)
def _load_cached_state(path: str) -> Dict[str, Any]:
    return load_state(Path(path))


def _resolve_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "state" in payload and payload["state"]:
        return payload["state"]
    state_path = payload.get("state_path")
    if state_path:
        return _load_cached_state(state_path)
    raise ValueError("Request must include 'state' or 'state_path'.")


def _top_strings(state: Dict[str, Any], metric: str, limit: int = 5) -> List[str]:
    scores = state.get("string_scores", {})
    ranked = sorted(
        ((float(data.get(metric, 0.0)), text) for text, data in scores.items()),
        reverse=True,
    )
    return [text for _, text in ranked[:limit]]


def adaptive_surface(context: str, state: Dict[str, Any], *, k: int = 15) -> Dict[str, Any]:
    context = (context or "").strip().lower()
    candidates = []
    if context:
        scores = state.get("string_scores", {})
        for text, payload in scores.items():
            if context in text.lower():
                candidates.append((float(payload.get("patternability", 0.0)), text))
        candidates.sort(reverse=True)
    if not candidates:
        seeds = _top_strings(state, "patternability", limit=3)
    else:
        seeds = [text for _, text in candidates[:3]]
    result = propose_from_state(state, seeds=seeds, k=k, min_connector=0.25)
    return {"seeds": seeds, "proposals": result["proposals"]}


def bridge_surface(context: str, state: Dict[str, Any], *, k: int = 15) -> Dict[str, Any]:
    seeds = _top_strings(state, "connector", limit=4)
    result = propose_from_state(state, seeds=seeds, k=k, min_connector=0.35)
    return {"seeds": seeds, "proposals": result["proposals"]}


def generate_why(entries: List[Dict[str, Any]]) -> List[str]:
    explanations: List[str] = []
    for item in entries:
        explanation = (
            f"{item['string']} — pattern={item['patternability']:.3f}, "
            f"connector={item['connector']:.3f}, occ={item['occurrences']}"
        )
        explanations.append(explanation)
    return explanations


@app.route("/api/surface", methods=["POST"])
def surface_strings():
    payload = request.get_json(force=True) or {}
    context = payload.get("context", "")
    mode = payload.get("mode", "adaptive")
    k = int(payload.get("k", 15))
    try:
        state = _resolve_state(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if mode == "adaptive":
        data = adaptive_surface(context, state, k=k)
    else:
        data = bridge_surface(context, state, k=k)
    return jsonify({
        "seeds": data["seeds"],
        "discovered": data["proposals"],
        "explanation": generate_why(data["proposals"][:10]),
    })


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=8080, debug=True)
