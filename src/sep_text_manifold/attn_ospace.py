"""Transformer-based reliability model for O-space manifolds.

This module implements a lightweight attention backbone with optional
cross-attention into an evidence memory and a reliability head that
predicts admit probability plus support margin.

The implementation is intentionally lean so it can be iterated without
locking us into a specific training recipe.  It supports three core use
cases:

* Embedding incoming span tokens with sinusoidal + phase channels.
* Running a stack of Transformer encoder blocks over the sequence.
* Optionally attending into an evidence memory (truth-pack windows,
  historical price manifolds) before emitting reliability scores.

If PyTorch is not available at import time, the module exposes a
`torch_available()` helper so callers can short circuit gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

try:  # PyTorch is an optional dependency for this feature set
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - allows repo to function without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass
class OspaceTransformerConfig:
    """Configuration for the O-space Transformer backbone."""

    vocab_size: int
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_hidden_dim: int = 512
    dropout: float = 0.1
    max_position_embeddings: int = 2048
    use_phase_channel: bool = True
    phase_scale: float = 1.0
    cross_attention_heads: Optional[int] = None
    use_cross_attention: bool = True
    max_evidence_len: int = 256
    evidence_encoder_layers: int = 1
    evidence_encoder_heads: int = 4
    evidence_feature_dim: int = 0
    evidence_dropout: float = 0.1


@dataclass
class OspaceTransformerOutput:
    """Convenience container for forward pass outputs."""

    admit_logits: "torch.Tensor"
    support_margin: "torch.Tensor"
    span_representations: "torch.Tensor"
    evidence_attention: Optional["torch.Tensor"]


def torch_available() -> bool:
    """Return True if torch is importable."""

    return torch is not None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for sep_text_manifold.attn_ospace. "
            "Install sep-text-manifold with the 'attn' extra or add torch to your environment."
        )


def _build_position_encoding(length: int, dim: int, device: "torch.device") -> "torch.Tensor":
    """Create sinusoidal position encodings as described in Vaswani et al."""

    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-torch.log(torch.tensor(10_000.0)) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


if torch is not None:

    class EvidenceEncoder(nn.Module):
        """Encode evidence sentences into memory vectors with optional feature fusion."""

        def __init__(self, config: OspaceTransformerConfig) -> None:
            super().__init__()
            self.config = config
            self.dropout = nn.Dropout(config.evidence_dropout)
            self.layer_norm = nn.LayerNorm(config.d_model)

            if config.evidence_encoder_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.evidence_encoder_heads,
                    dim_feedforward=config.ffn_hidden_dim,
                    dropout=config.dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=config.evidence_encoder_layers,
                )
            else:
                self.encoder = None

            if config.evidence_feature_dim > 0:
                self.feature_proj = nn.Linear(config.evidence_feature_dim, config.d_model)
            else:
                self.feature_proj = None

            pe = _build_position_encoding(
                config.max_evidence_len,
                config.d_model,
                device=torch.device("cpu"),
            )
            self.register_buffer("positional_encoding", pe, persistent=False)

        def forward(
            self,
            embeddings: "torch.Tensor",
            *,
            token_mask: Optional["torch.Tensor"] = None,
            features: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            """Encode evidence tokens and fuse structural features."""

            batch, evidences, seq_len, dim = embeddings.shape
            if evidences == 0:
                return embeddings.new_zeros(batch, 0, dim)
            if features is not None and features.shape[:2] != (batch, evidences):
                raise ValueError(
                    "evidence_features must align with embeddings on batch and evidence dimensions"
                )
            if seq_len > self.config.max_evidence_len:
                embeddings = embeddings[:, :, : self.config.max_evidence_len, :]
                if token_mask is not None:
                    token_mask = token_mask[:, :, : self.config.max_evidence_len]
                seq_len = self.config.max_evidence_len

            pos = self.positional_encoding[:seq_len].to(embeddings.device)
            enriched = embeddings + pos
            enriched = self.dropout(enriched)

            flat = enriched.view(batch * evidences, seq_len, dim)
            if token_mask is not None:
                flat_mask = token_mask.view(batch * evidences, seq_len)
                src_key_padding_mask = flat_mask == 0
            else:
                flat_mask = None
                src_key_padding_mask = None

            if self.encoder is not None:
                encoded = self.encoder(flat, src_key_padding_mask=src_key_padding_mask)
            else:
                encoded = flat

            if flat_mask is not None:
                mask = flat_mask.float().unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                pooled = (encoded * mask).sum(dim=1) / denom
            else:
                pooled = encoded.mean(dim=1)

            if features is not None and self.feature_proj is not None:
                feature_proj = self.feature_proj(features.view(batch * evidences, -1))
                pooled = pooled + feature_proj

            pooled = self.layer_norm(pooled)
            return pooled.view(batch, evidences, dim)

    class ReliabilityHead(nn.Module):
        """Small MLP projecting the pooled representation to admit logits + margin."""

        def __init__(self, dim: int, dropout: float) -> None:
            super().__init__()
            hidden = max(dim, 128)
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.admit_head = nn.Linear(hidden // 2, 1)
            self.margin_head = nn.Linear(hidden // 2, 1)

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            features = self.net(x)
            admit_logits = self.admit_head(features).squeeze(-1)
            margin = self.margin_head(features).squeeze(-1)
            return admit_logits, margin


    class OspaceTransformer(nn.Module):
        """Transformer backbone with optional evidence cross-attention."""

        def __init__(self, config: OspaceTransformerConfig) -> None:
            super().__init__()
            self.config = config
            self.token_embed = nn.Embedding(
                config.vocab_size,
                config.d_model,
                padding_idx=0,
            )
            self.dropout = nn.Dropout(config.dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.ffn_hidden_dim,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

            if config.use_cross_attention:
                cross_heads = config.cross_attention_heads or config.num_heads
                self.cross_attention = nn.MultiheadAttention(
                    config.d_model,
                    cross_heads,
                    dropout=config.dropout,
                    batch_first=True,
                )
            else:
                self.cross_attention = None

            self.evidence_encoder = EvidenceEncoder(config)

            self.reliability_head = ReliabilityHead(config.d_model * 2, config.dropout)

            # Register a buffer for positional encodings to avoid recomputation at inference time.
            pe = _build_position_encoding(config.max_position_embeddings, config.d_model, device=torch.device("cpu"))
            self.register_buffer("positional_encoding", pe, persistent=False)

        def _add_position_and_phase(
            self,
            token_embeddings: "torch.Tensor",
            phase_features: Optional["torch.Tensor"],
        ) -> "torch.Tensor":
            seq_len = token_embeddings.size(1)
            if seq_len > self.config.max_position_embeddings:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds configured maximum {self.config.max_position_embeddings}."
                )
            pos = self.positional_encoding[:seq_len]
            pos = pos.to(token_embeddings.device)
            enriched = token_embeddings + pos
            if phase_features is not None and self.config.use_phase_channel:
                if phase_features.shape != token_embeddings.shape:
                    raise ValueError(
                        "Phase features must match token embedding shape (batch, seq, d_model)."
                    )
                enriched = enriched + self.config.phase_scale * phase_features
            return self.dropout(enriched)

        def forward(
            self,
            token_ids: "torch.Tensor",
            *,
            attention_mask: Optional["torch.Tensor"] = None,
            phase_features: Optional["torch.Tensor"] = None,
            evidence_memory: Optional["torch.Tensor"] = None,
            evidence_mask: Optional["torch.Tensor"] = None,
            evidence_tokens: Optional["torch.Tensor"] = None,
            evidence_token_mask: Optional["torch.Tensor"] = None,
            evidence_features: Optional["torch.Tensor"] = None,
        ) -> OspaceTransformerOutput:
            """Run the Transformer and return admit logits + support margin.

            Args:
                token_ids: Long tensor of shape (batch, seq_len).
                attention_mask: Optional mask (batch, seq_len) where 1 indicates valid tokens.
                phase_features: Optional tensor matching the token embedding shape for phase/prime channels.
                evidence_memory: Optional tensor of shape (batch, mem_len, d_model) representing
                    truth-pack or manifold evidence. When provided, the last hidden state attends
                    over this memory via multi-head attention.
                evidence_mask: Optional mask (batch, mem_len) where 1 marks valid memory positions.
            """

            embeddings = self.token_embed(token_ids)
            enriched = self._add_position_and_phase(embeddings, phase_features)

            if attention_mask is not None:
                # Transformer expects True where we should mask out positions.
                src_key_padding_mask = attention_mask == 0
            else:
                src_key_padding_mask = None

            hidden = self.encoder(enriched, src_key_padding_mask=src_key_padding_mask)

            # Pool the final token representation; more advanced pooling can hook in later.
            pooled = hidden[:, -1, :]

            cross_attn_weights = None

            if evidence_tokens is not None:
                if evidence_token_mask is None:
                    evidence_token_mask = (evidence_tokens != 0).long()
                if evidence_features is not None and evidence_features.dim() != 3:
                    raise ValueError("evidence_features must be shaped (batch, items, features)")
                embeddings = self.token_embed(evidence_tokens)
                evidence_memory = self.evidence_encoder(
                    embeddings,
                    token_mask=evidence_token_mask,
                    features=evidence_features,
                )
                evidence_mask = evidence_token_mask.sum(dim=-1) > 0
                evidence_mask = evidence_mask.long()

            if self.cross_attention is not None and evidence_memory is not None:
                query = pooled.unsqueeze(1)
                key = value = evidence_memory
                if evidence_mask is not None:
                    key_padding_mask = evidence_mask == 0
                else:
                    key_padding_mask = None
                attended, attn_weights = self.cross_attention(
                    query,
                    key,
                    value,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                )
                cross_attn_weights = attn_weights  # (batch, 1, mem_len)
                # Concatenate pooled representation with attended evidence summary.
                representation = torch.cat([pooled, attended.squeeze(1)], dim=-1)
            else:
                representation = torch.cat([pooled, pooled], dim=-1)

            admit_logits, margin = self.reliability_head(representation)

            return OspaceTransformerOutput(
                admit_logits=admit_logits,
                support_margin=margin,
                span_representations=hidden,
                evidence_attention=cross_attn_weights,
            )

else:  # pragma: no cover - exercised when torch is not installed

    class OspaceTransformer:  # type: ignore[override]
        """Placeholder implementation when torch is unavailable."""

        def __init__(self, *_: object, **__: object) -> None:
            _require_torch()

        def forward(self, *args: object, **kwargs: object) -> None:  # noqa: D401
            _require_torch()


__all__ = [
    "OspaceTransformerConfig",
    "OspaceTransformerOutput",
    "OspaceTransformer",
    "torch_available",
]
