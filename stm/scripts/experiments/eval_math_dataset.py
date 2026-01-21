#!/usr/bin/env python3
"""Evaluate the STM LM on a subset of the DeepMind math dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import torch
from datasets import load_dataset
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


def contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    size = len(needle)
    for start in range(len(haystack) - size + 1):
        if haystack[start : start + size] == needle:
            return True
    return False


def encode_problem(
    prompt: str,
    vocab: List[str],
    *,
    window_bytes: int,
    stride_bytes: int,
    precision: int,
) -> tuple[list[int], ManifoldEncoding]:
    signature_to_id = {sig: idx for idx, sig in enumerate(vocab)}
    encoding = encode_text(
        prompt,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=precision,
        deduplicate=False,
    )
    token_ids = [signature_to_id[span.signature] for span in encoding.spans if span.signature in signature_to_id]
    spans = [
        ManifoldSpan(offset=index * stride_bytes, signature=span.signature)
        for index, span in enumerate(encoding.spans)
    ]
    return token_ids, ManifoldEncoding(
        spans=spans,
        prototypes={span.signature: span.payload for span in encoding.spans if span.payload},
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=precision,
        original_length=len(prompt.encode("utf-8")),
    )


def decode_tokens(
    signatures: List[str],
    prototypes: Dict[str, bytes],
    *,
    window_bytes: int,
    stride_bytes: int,
) -> str:
    spans = [
        ManifoldSpan(offset=index * stride_bytes, signature=sig)
        for index, sig in enumerate(signatures)
        if sig
    ]
    encoding = ManifoldEncoding(
        spans=spans,
        prototypes=prototypes,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=3,
        original_length=0,
    )
    return decode_encoding(encoding).decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate STM LM on DeepMind math dataset.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-vocab", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="dim/competition_math", help="Hugging Face dataset ID.")
    parser.add_argument("--dataset-config", type=str, help="Optional dataset config/name.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to evaluate.")
    parser.add_argument("--question-field", type=str, default="problem", help="Column containing the question text.")
    parser.add_argument("--answer-field", type=str, default="solution", help="Column containing the reference answer.")
    parser.add_argument("--type-field", type=str, default="type", help="Column used for --subset filtering.")
    parser.add_argument("--subset", type=str, help="Filter value within --type-field (if available).")
    parser.add_argument("--max-problems", type=int, default=100)
    parser.add_argument("--text-root", type=Path, required=True, help="Corpus root for prototype recovery.")
    parser.add_argument("--cache", type=Path, help="Prototype cache JSON.")
    parser.add_argument("--window-bytes", type=int, default=512)
    parser.add_argument("--stride-bytes", type=int, default=384)
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-documents", type=int, help="Prototype scan cap.")
    parser.add_argument(
        "--match-mode",
        choices=("signatures", "text"),
        default=["signatures"],
        nargs="+",
        help="Select one or more scoring modes.",
    )
    args = parser.parse_args()
    match_modes = list(dict.fromkeys(args.match_mode))
    match_mode_set = set(match_modes)

    vocab = json.loads(args.dataset_vocab.read_text(encoding="utf-8")).get("signatures", [])
    pad_token_id = len(vocab)

    load_kwargs = {"split": args.split}
    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, **load_kwargs)
    else:
        ds = load_dataset(args.dataset, **load_kwargs)

    if args.subset:
        if args.type_field not in ds.column_names:
            raise ValueError(f"--subset provided but column '{args.type_field}' not found in dataset.")
        ds = ds.filter(lambda example: example[args.type_field] == args.subset)
    if args.max_problems:
        ds = ds.select(range(min(args.max_problems, len(ds))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    correct = {mode: 0 for mode in match_modes}
    prototypes_cache: Dict[str, bytes] = {}

    for sample in ds:
        question = sample[args.question_field].strip()
        answer = sample[args.answer_field].strip()
        token_ids, encoding = encode_problem(
            question,
            vocab=vocab,
            window_bytes=args.window_bytes,
            stride_bytes=args.stride_bytes,
            precision=args.precision,
        )
        signatures = [span.signature for span in encoding.spans]
        answer_signature_seq: list[str] = []

        if "text" in match_mode_set:
            required = set(signatures)
            prototypes_cache = ensure_prototypes(
                required,
                cache_path=args.cache,
                text_root=args.text_root,
                window_bytes=args.window_bytes,
                stride_bytes=args.stride_bytes,
                precision=args.precision,
                max_documents=args.max_documents,
            )
        if "signatures" in match_mode_set:
            answer_signature_seq = [
                span.signature
                for span in encode_text(
                    answer,
                    window_bytes=args.window_bytes,
                    stride_bytes=args.stride_bytes,
                    precision=args.precision,
                    deduplicate=False,
                ).spans
            ]

        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_tensor)
        output_ids = model.generate(
            input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0.0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            pad_token_id=pad_token_id,
            eos_token_id=model.config.eos_token_id,
        )[0].tolist()

        output_signatures = [vocab[idx] for idx in output_ids]
        if "text" in match_mode_set:
            prototypes_cache = ensure_prototypes(
                output_signatures,
                cache_path=args.cache,
                text_root=args.text_root,
                window_bytes=args.window_bytes,
                stride_bytes=args.stride_bytes,
                precision=args.precision,
                max_documents=args.max_documents,
            )

            decoded_answer = decode_tokens(
                output_signatures,
                prototypes=prototypes_cache,
                window_bytes=args.window_bytes,
                stride_bytes=args.stride_bytes,
            )
            if answer in decoded_answer:
                correct["text"] += 1
        if "signatures" in match_mode_set and contains_subsequence(output_signatures, answer_signature_seq):
            correct["signatures"] += 1

    summary = {
        "split": args.split,
        "subset": args.subset,
        "max_problems": args.max_problems,
        "total": len(ds),
        "results": {
            mode: {"correct": correct[mode], "accuracy": correct[mode] / max(len(ds), 1)}
            for mode in match_modes
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
