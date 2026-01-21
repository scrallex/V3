#!/usr/bin/env python3
"""Demo script for prompting the STM LM on math QA inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import torch
from transformers import AutoModelForCausalLM

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
TOP_ROOT = REPO_ROOT.parent
for candidate in (MODULE_DIR, REPO_ROOT, TOP_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from prototype_cache import ensure_prototypes
from src.manifold.codec import ManifoldEncoding, ManifoldSpan, decode_encoding, encode_text


def encode_prompt(text: str, vocab: List[str], pad_token_id: int, window_bytes: int, stride_bytes: int, precision: int):
    encoding = encode_text(
        text,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=precision,
        deduplicate=False,
    )
    signature_to_id = {sig: idx for idx, sig in enumerate(vocab)}
    token_ids = [signature_to_id[span.signature] for span in encoding.spans]
    prototypes = {span.signature: span.payload for span in encoding.spans if span.payload}
    spans = [
        ManifoldSpan(offset=index * stride_bytes, signature=span.signature)
        for index, span in enumerate(encoding.spans)
    ]
    manifold_encoding = ManifoldEncoding(
        spans=spans,
        prototypes=prototypes,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=precision,
        original_length=len(text.encode("utf-8")),
    )
    return token_ids, manifold_encoding


def decode_tokens(
    signatures: List[str],
    prototypes: Dict[str, bytes],
    *,
    window_bytes: int,
    stride_bytes: int,
) -> str:
    local = dict(prototypes)
    for sig in signatures:
        if sig and sig not in local:
            local[sig] = sig.encode("utf-8")
    spans = [
        ManifoldSpan(offset=index * stride_bytes, signature=sig)
        for index, sig in enumerate(signatures)
        if sig
    ]
    encoding = ManifoldEncoding(
        spans=spans,
        prototypes={sig: local[sig] for sig in signatures if sig in local},
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=3,
        original_length=0,
    )
    data = decode_encoding(encoding)
    return data.decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple math QA prompt/response demo for STM models.")
    parser.add_argument("--prompt", type=str, required=True, help="Math question text.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the trained STM LM.")
    parser.add_argument("--dataset-vocab", type=Path, required=True, help="vocab.json from dataset prep.")
    parser.add_argument("--text-root", type=Path, required=True, help="Raw corpus root for prototype lookup.")
    parser.add_argument("--cache", type=Path, help="Optional JSON cache for prototypes.")
    parser.add_argument("--max-documents", type=int, default=20000, help="Doc cap when filling prototype cache.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--window-bytes", type=int, default=512)
    parser.add_argument("--stride-bytes", type=int, default=384)
    parser.add_argument("--precision", type=int, default=3)
    args = parser.parse_args()

    vocab_data = json.loads(args.dataset_vocab.read_text(encoding="utf-8"))
    vocab = list(vocab_data.get("signatures", []))
    pad_token_id = len(vocab)

    prompt_token_ids, manifold_encoding = encode_prompt(
        args.prompt,
        vocab=vocab,
        pad_token_id=pad_token_id,
        window_bytes=args.window_bytes,
        stride_bytes=args.stride_bytes,
        precision=args.precision,
    )
    prompt_signatures = [span.signature for span in manifold_encoding.spans]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    input_tensor = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    generated = model.generate(
        input_tensor,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=pad_token_id,
        eos_token_id=model.config.eos_token_id,
    )[0].tolist()

    generated_signatures = [vocab[idx] for idx in generated]
    required_signatures = set(generated_signatures)
    prototypes = ensure_prototypes(
        required_signatures,
        cache_path=args.cache,
        text_root=args.text_root,
        window_bytes=args.window_bytes,
        stride_bytes=args.stride_bytes,
        precision=args.precision,
        max_documents=args.max_documents,
    )

    decoded_prompt = decode_tokens(
        prompt_signatures,
        prototypes=prototypes,
        window_bytes=args.window_bytes,
        stride_bytes=args.stride_bytes,
    )
    decoded_output = decode_tokens(
        generated_signatures,
        prototypes=prototypes,
        window_bytes=args.window_bytes,
        stride_bytes=args.stride_bytes,
    )

    print("=== Prompt (decoded) ===")
    print(decoded_prompt.strip())
    print("\n=== Model continuation ===")
    print(decoded_output.strip())


if __name__ == "__main__":
    main()
