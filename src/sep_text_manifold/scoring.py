"""
Scoring functions for strings and connectors.

These functions compute high‑level scores that quantify how
"patternable" or "connector‑like" a string is.  The formulas
implemented here are intentionally simple; you can refine them or
replace them entirely to better suit your use case.
"""

from __future__ import annotations

import math
from typing import Optional


def _logistic(x: float) -> float:
    """Return the logistic function 1/(1+exp(-x))."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def patternability_score(coherence: float, stability: float, entropy: float, rupture: float) -> float:
    """Compute a simple patternability score for a string.

    The patternability score reflects how consistently a string appears
    in coherent and stable contexts with low entropy and low rupture.
    A higher score indicates that the string is more likely to be part
    of a repeating pattern.

    The formula implemented here is a normalised linear combination
    followed by a logistic squashing.  It is loosely inspired by the
    full SEP scoring formula but simplified for clarity.

    Parameters
    ----------
    coherence, stability, entropy, rupture: float
        Mean metric values for the string computed over all of its
        occurrences.

    Returns
    -------
    float
        A value between 0 and 1.  Values closer to 1 indicate higher
        patternability.
    """
    # Weights for each term.  You can tune these to bias towards
    # coherence or stability more heavily.
    alpha = 0.5  # coherence weight
    beta = 0.3   # stability weight
    gamma = 0.2  # entropy penalty weight
    delta = 0.2  # rupture penalty weight
    # Compute linear combination.  Subtract mean values for entropy and
    # rupture because higher values indicate randomness and instability.
    score = alpha * coherence + beta * stability - gamma * entropy - delta * rupture
    # Apply logistic to bound in [0, 1]
    return _logistic(score)


def connector_score(
    betweenness: float,
    bridging: float,
    pmi_across_themes: float,
    theme_entropy_neighbors: float,
    redundant_degree: float,
) -> float:
    """Compute a connector score for a string.

    A connector string bridges otherwise separate themes.  This score
    combines centrality measures (betweenness, bridging) with pointwise
    mutual information (PMI) and penalises strings that appear in
    nearly every theme (high redundant degree).  It is a simple
    heuristic and can be refined.

    Parameters
    ----------
    betweenness, bridging, pmi_across_themes, theme_entropy_neighbors, redundant_degree : float
        Normalised graph analytics values for the string.  See
        `themes.py` for how these are computed.

    Returns
    -------
    float
        A value between 0 and 1.  Higher values indicate stronger
        connector behaviour.
    """
    a = 0.4
    b = 0.3
    c = 0.2
    d = 0.1
    e = 0.2
    linear = (
        a * betweenness +
        b * bridging +
        c * pmi_across_themes +
        d * theme_entropy_neighbors -
        e * redundant_degree
    )
    return _logistic(linear)