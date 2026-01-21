"""Regime manifold research utilities for encoding market microstructure into structural signatures."""

from .codec import (
    Candle,
    EncodedWindow,
    MarketManifoldCodec,
    decode_window_bits,
)

__all__ = [
    "Candle",
    "EncodedWindow",
    "MarketManifoldCodec",
    "decode_window_bits",
]
