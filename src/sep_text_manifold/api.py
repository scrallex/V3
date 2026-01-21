"""
FastAPI server scaffolding for the Sep Text Manifold.

This module defines a simple REST API exposing string metrics,
themes and raw event data.  It is intended as a starting point and
does not yet implement persistence or authentication.

To run the API locally install the ``fastapi`` and ``uvicorn``
packages and start the server with ``uvicorn sep_text_manifold.api:app``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .propose import propose_from_state, load_state as load_analysis_state
from .seen import get_engine


class Metrics(BaseModel):
    coherence: float
    stability: float
    entropy: float
    rupture: float
    lambda_hazard: float


class StringMetricsOut(BaseModel):
    string: str
    metrics: Metrics


class ThemeOut(BaseModel):
    id: int
    members: List[str]


class DiscoverRequest(BaseModel):
    seeds: List[str]
    k: int = 25
    min_connector: float = 0.0
    min_patternability: float = 0.0
    target_profile: Optional[str] = None
    state_path: Optional[str] = None
    state: Optional[Dict[str, Any]] = None


# In-memory stores – these should be replaced by a proper database or
# state management layer in a production system.
_string_metrics: Dict[str, Metrics] = {}
_themes: List[List[str]] = []

app = FastAPI(title="Sep Text Manifold API")


@app.get("/strings/search", response_model=List[StringMetricsOut])
def search_strings(q: Optional[str] = None, min_patternability: float = 0.0) -> List[StringMetricsOut]:
    """Search for strings by substring and patternability threshold."""
    from .scoring import patternability_score
    results: List[StringMetricsOut] = []
    for s, metrics in _string_metrics.items():
        if q is None or q.lower() in s:
            p_score = patternability_score(metrics.coherence, metrics.stability, metrics.entropy, metrics.rupture)
            if p_score >= min_patternability:
                results.append(StringMetricsOut(string=s, metrics=metrics))
    return results


@app.get("/strings/{string}", response_model=StringMetricsOut)
def get_string(string: str) -> StringMetricsOut:
    """Get metrics for a single string."""
    metrics = _string_metrics.get(string)
    if metrics is None:
        raise HTTPException(status_code=404, detail="String not found")
    return StringMetricsOut(string=string, metrics=metrics)


@app.get("/themes", response_model=List[ThemeOut])
def list_themes() -> List[ThemeOut]:
    """List all detected themes."""
    return [ThemeOut(id=i, members=members) for i, members in enumerate(_themes)]


@app.get("/themes/{theme_id}", response_model=ThemeOut)
def get_theme(theme_id: int) -> ThemeOut:
    """Get a single theme by ID."""
    if 0 <= theme_id < len(_themes):
        return ThemeOut(id=theme_id, members=_themes[theme_id])
    raise HTTPException(status_code=404, detail="Theme not found")


@app.post("/load")
def load_data(strings: Dict[str, Dict[str, float]], themes: List[List[str]]) -> Dict[str, str]:
    """Load string metrics and themes into the API.

    This endpoint is a simple way to populate the in‑memory stores
    during development.  In a real application data would be loaded
    from a database or file system.
    """
    global _string_metrics, _themes
    _string_metrics = {s: Metrics(**m) for s, m in strings.items()}
    _themes = themes
    return {"status": "loaded", "strings": len(_string_metrics), "themes": len(_themes)}


@app.post("/discover")
def discover_strings(request: DiscoverRequest) -> Dict[str, Any]:
    """Generate bridge-string proposals from a loaded or supplied state."""
    if request.state is not None:
        state = request.state
    elif request.state_path:
        state = load_analysis_state(request.state_path)
    elif _string_metrics:
        state = {
            "string_scores": {
                text: {
                    "metrics": metric.dict(),
                    "occurrences": 0,
                    "window_ids": [],
                    "patternability": 0.0,
                    "connector": 0.0,
                }
                for text, metric in _string_metrics.items()
            },
            "themes": _themes,
        }
    else:
        raise HTTPException(status_code=400, detail="State not provided and in-memory store is empty")
    try:
        result = propose_from_state(
            state,
            seeds=request.seeds,
            k=request.k,
            min_connector=request.min_connector,
            min_patternability=request.min_patternability,
            target_profile=request.target_profile,
        )
    except ValueError as exc:  # pragma: no cover - guarded by validation
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/seen")
def seen_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    trigger = payload.get("trigger")
    if not trigger:
        raise HTTPException(status_code=400, detail="Missing 'trigger' field")
    engine = get_engine()
    return engine.seen(str(trigger))
