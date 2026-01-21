"""Semantic adapters for the structural manifold pipeline.

The core STM stack focuses on byte-level repetition. This module adds
lightweight embedding plumbing so callers can project strings into a
semantic vector space and compare those vectors against the structural
scores already emitted by the manifold.  The design keeps dependencies
optional: SentenceTransformer works when installed, otherwise a
deterministic hashing fallback provides stable pseudo-embeddings that
still allow experimentation with the structural/semantic bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


def _normalise(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def _hash_embedding(texts: Sequence[str], *, dims: int = 256) -> np.ndarray:
    """Return deterministic pseudo-embeddings when no model is available."""

    import hashlib

    vectors = np.zeros((len(texts), dims), dtype=np.float32)
    for idx, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand digest to fill the requested dimensionality.
        repeats = (dims + len(digest) - 1) // len(digest)
        blob = (digest * repeats)[:dims]
        vectors[idx] = np.frombuffer(blob, dtype=np.uint8) / 255.0
        # Mean centre for basic cosine geometry.
        vectors[idx] -= vectors[idx].mean()
    # Avoid zero vectors.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


@dataclass
class EmbeddingConfig:
    method: str = "auto"
    model_name: str = "all-MiniLM-L6-v2"
    dims: int = 256


class SemanticEmbedder:
    """Encodes strings into vectors using optional transformer models."""

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = None
        method = self.config.method.lower()
        if method == "transformer" or (method == "auto" and SentenceTransformer):
            self._initialise_transformer()
        elif method not in {"auto", "hash"}:
            raise ValueError(f"Unknown embedding method: {self.config.method}")

    def _initialise_transformer(self) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        self._model = SentenceTransformer(self.config.model_name)  # type: ignore[arg-type]

    def encode(self, strings: Sequence[str]) -> np.ndarray:
        cleaned = [_normalise(s) for s in strings]
        if self._model is not None:
            vectors = self._model.encode(cleaned, show_progress_bar=False)  # type: ignore[no-any-return]
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            return vectors / norms
        return _hash_embedding(cleaned, dims=self.config.dims)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)
    a_norms[a_norms == 0.0] = 1.0
    b_norms[b_norms == 0.0] = 1.0
    return (a @ b.T) / (a_norms * b_norms.T)


def seed_similarity(
    tokens: Sequence[str],
    *,
    embedder: SemanticEmbedder,
    seeds: Sequence[str],
) -> np.ndarray:
    token_vecs = embedder.encode(tokens)
    seed_vecs = embedder.encode(seeds)
    sims = cosine_similarity(token_vecs, seed_vecs)
    return sims.mean(axis=1)


def aggregate_semantic_scores(
    strings: Sequence[str],
    *,
    embedder: SemanticEmbedder,
    seeds: Sequence[str],
) -> List[float]:
    return seed_similarity(strings, embedder=embedder, seeds=seeds).tolist()

