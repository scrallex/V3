"""Convenience helpers to expose structural manifold functionality as a library."""

from .encoder import build_signature_index
from .verifier import score_documents

__all__ = ["build_signature_index", "score_documents"]
