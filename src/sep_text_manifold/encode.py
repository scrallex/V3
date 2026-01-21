"""
Encoding and quantum metric computation for the Sep Text Manifold.

This module exposes functions to convert a window of bytes into a
bitstream and compute informational metrics on that stream.  The
metrics implemented here are simplified stand‑ins for the full
native manifold algorithms used by the SEP Engine.  They provide a
reasonable approximation for experimentation and can be replaced by
calls into the C++ implementation when performance or fidelity is
required.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

from . import native


def bytes_to_bits(data: bytes) -> List[int]:
    """Convert a sequence of bytes into a list of bits (0 or 1).

    The most significant bit of each byte is emitted first.
    """
    bits: List[int] = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def compute_metrics(bits: List[int]) -> Dict[str, float]:
    """Compute simplified quantum‑inspired metrics on a bitstream.

    Parameters
    ----------
    bits:
        Sequence of 0/1 values representing the bitstream.

    Returns
    -------
    dict
        A dictionary containing `coherence`, `stability`, `entropy`,
        `rupture` and `lambda_hazard` fields.  The definitions used
        here are inspired by the SEP Engine but are simplified:

        - **Entropy** is the binary Shannon entropy of the distribution
          of zeros and ones, normalised to lie in [0, 1].
        - **Coherence** is `1 - entropy`, so it approaches 1 when the
          bitstream is highly regular and 0 when random.
        - **Rupture** is the fraction of adjacent bit transitions
          (i.e. the rate of change).  A high rupture value implies
          instability.
        - **Stability** is `1 - rupture` and mirrors rupture.
        - **lambda_hazard** is set equal to the rupture value; in the
          SEP Engine this captures the instantaneous hazard of a
          pattern collapsing.
    """
    n = len(bits)
    if n == 0:
        return {
            "coherence": 0.0,
            "stability": 0.0,
            "entropy": 0.0,
            "rupture": 0.0,
            "lambda_hazard": 0.0,
        }
    # Compute distribution of bits
    count_ones = sum(bits)
    count_zeros = n - count_ones
    p_zero = count_zeros / n
    p_one = count_ones / n
    # Binary Shannon entropy in bits
    entropy = 0.0
    for p in (p_zero, p_one):
        if p > 0:
            entropy -= p * math.log2(p)
    # Normalise entropy: max for binary distribution is 1.0
    entropy_norm = entropy
    if entropy_norm > 1.0:
        entropy_norm = 1.0
    coherence = 1.0 - entropy_norm
    # Compute rupture (transition rate)
    transitions = sum(1 for i in range(1, n) if bits[i] != bits[i - 1])
    rupture = transitions / (n - 1) if n > 1 else 0.0
    stability = 1.0 - rupture
    lambda_hazard = rupture
    return {
        "coherence": coherence,
        "stability": stability,
        "entropy": entropy_norm,
        "rupture": rupture,
        "lambda_hazard": lambda_hazard,
    }


def encode_window(window: bytes) -> Dict[str, float]:
    """Encode a window of bytes and compute metrics.

    This function wraps `bytes_to_bits` and `compute_metrics` to
    produce a single metrics dictionary for a window of data.
    """
    try:
        scored = native.score_bytes(window, input_type="text", prefer_grpc=True)
        return {
            "coherence": float(scored.get("coherence", 0.0)),
            "stability": float(scored.get("stability", 0.0)),
            "entropy": float(scored.get("entropy", 0.0)),
            "rupture": float(scored.get("hazard", scored.get("lambda_hazard", 0.0))),
            "lambda_hazard": float(scored.get("hazard", scored.get("lambda_hazard", 0.0))),
        }
    except Exception:
        bits = bytes_to_bits(window)
        if native.use_native():
            try:
                result = native.analyze_window(bits)
                rupture_ratio = float(result.rupture_ratio)
                return {
                    "coherence": float(result.coherence),
                    "stability": 1.0 - rupture_ratio,
                    "entropy": float(result.entropy),
                    "rupture": rupture_ratio,
                    "lambda_hazard": rupture_ratio,
                }
            except Exception:  # pragma: no cover - fall back to simplified metrics
                if native.HAVE_NATIVE:
                    try:
                        metrics = native.analyze_bits(bits)
                    except Exception:
                        pass
                    else:
                        return {
                            "coherence": metrics.get("coherence", 0.0),
                            "stability": metrics.get("stability", 0.0),
                            "entropy": metrics.get("entropy", 0.0),
                            "rupture": metrics.get("rupture", 0.0),
                            "lambda_hazard": metrics.get("lambda_hazard", metrics.get("rupture", 0.0)),
                        }
        return compute_metrics(bits)


def signature_from_metrics(coherence: float, stability: float, entropy: float, precision: int = 2) -> str:
    """Generate a repetition signature string from the core metrics.

    The signature buckets each metric to a given decimal precision and
    concatenates them.  This mirrors the behaviour of the SEP
    `make_signature` helper in `manifold_builder.cpp`【739285426356909†L60-L75】.
    """
    scale = 10 ** precision
    def bucket(value: float) -> float:
        value_clamped = min(max(value, 0.0), 1.0)
        return round(value_clamped * scale) / scale
    c = bucket(coherence)
    s = bucket(stability)
    e = bucket(entropy)
    return f"c{c}_s{s}_e{e}"
