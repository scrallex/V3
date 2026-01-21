"""
Top‑level package for the Sep Text Manifold library.

This package provides modules for ingesting text data, computing
quantum‑inspired metrics over sliding windows of bytes, building a
manifold of signals, extracting and scoring strings, detecting themes
and exposing the results via a simple API and command line interface.

The heavy numerical work is delegated to the SEP Engine's structural manifold
implementation.  See `docs/integration_with_sep.md` for details on
how to integrate with the existing C++ codebase.
"""

from .ingest import ingest_directory
from .encode import encode_window
from .gpu_windows import gpu_window_metrics  # type: ignore  # Optional CUDA backend
from .guardrail import calibrate_threshold, summarise_guardrail
from .manifold import build_manifold
from .pipeline import AnalysisResult, analyse_directory
from .strings import extract_strings, aggregate_string_metrics
from .scoring import patternability_score, connector_score
from .themes import build_theme_graph, detect_themes
from .dilution import (
    path_dilution,
    signal_dilution,
    semantic_dilution,
    compute_dilution_metrics,
)
from .comparison import stm_val_alignment, detection_lead_time, AlignmentResult, LeadTimeResult
from .feedback import suggest_twin_action, TwinSuggestion
from .semantic import SemanticEmbedder, EmbeddingConfig, seed_similarity

try:  # Optional transformer-based reliability model
    from .attn_ospace import OspaceTransformer, OspaceTransformerConfig
except ImportError:  # pragma: no cover - torch may be unavailable
    OspaceTransformer = None  # type: ignore[assignment]
    OspaceTransformerConfig = None  # type: ignore[assignment]

__all__ = [
    "ingest_directory",
    "encode_window",
    "build_manifold",
    "extract_strings",
    "aggregate_string_metrics",
    "analyse_directory",
    "AnalysisResult",
    "patternability_score",
    "connector_score",
    "build_theme_graph",
    "detect_themes",
    "path_dilution",
    "signal_dilution",
    "semantic_dilution",
    "compute_dilution_metrics",
    "SemanticEmbedder",
    "EmbeddingConfig",
    "seed_similarity",
    "stm_val_alignment",
    "detection_lead_time",
    "AlignmentResult",
    "LeadTimeResult",
    "suggest_twin_action",
    "TwinSuggestion",
    "calibrate_threshold",
    "summarise_guardrail",
    "OspaceTransformer",
    "OspaceTransformerConfig",
    "gpu_window_metrics",
]
