"""Wrapper utilities for structural manifold encoding."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from scripts.experiments.manifold_compression_eval import evaluate_manifold


def build_signature_index(
    text_root: Path,
    window_bytes: int = 512,
    stride_bytes: int = 384,
    precision: int = 3,
    tokenizer_name: str = "gpt2",
    tokenizer_trust_remote_code: bool = False,
    max_documents: Optional[int] = None,
) -> Dict[str, object]:
    """Return the same summary dictionary exposed by benchmark_eval."""

    return evaluate_manifold(
        text_root=text_root,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=precision,
        tokenizer_name=tokenizer_name,
        tokenizer_trust_remote_code=tokenizer_trust_remote_code,
        max_documents=max_documents,
    )
