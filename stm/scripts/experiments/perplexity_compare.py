#!/usr/bin/env python3
"""Compare manifold LM perplexity against a GPT-2 baseline on raw text."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import (  # type: ignore  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_from_disk  # type: ignore  # noqa: E402

from scripts.training.manifold_lm_trainer import ManifoldDataCollator, load_vocab  # noqa: E402
from scripts.experiments.manifold_compression_eval import iter_text_documents  # noqa: E402


def select_eval_subset(dataset, max_samples: Optional[int], eval_fraction: float, seed: int):
    if max_samples:
        return dataset.select(range(min(max_samples, len(dataset))))
    eval_size = max(1, int(len(dataset) * eval_fraction))
    return dataset.shuffle(seed=seed).select(range(eval_size))


def evaluate_manifold(
    model_path: Path,
    dataset_path: Path,
    vocab_path: Path,
    eval_fraction: float,
    max_samples: Optional[int],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Dict[str, object]:
    dataset = load_from_disk(str(dataset_path))
    _, vocab_size = load_vocab(vocab_path)
    pad_token_id = vocab_size  # builder appends <pad> at the end
    collator = ManifoldDataCollator(pad_token_id=pad_token_id)
    subset = select_eval_subset(dataset, max_samples, eval_fraction, seed)
    loader = DataLoader(subset, batch_size=batch_size, collate_fn=collator)

    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            attention = batch["attention_mask"]
            lengths = attention.sum(dim=1)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            token_count = int(lengths.sum().item())
            total_tokens += token_count
            total_loss += outputs.loss.item() * token_count

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    dataset_token_total = int(sum(dataset["length"]))

    return {
        "samples_evaluated": len(subset),
        "tokens_evaluated": total_tokens,
        "average_loss": avg_loss,
        "perplexity": perplexity,
        "dataset_sequences": len(dataset),
        "dataset_tokens_total": dataset_token_total,
    }


def chunk_tokens(tokens: Iterable[int], block_size: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    token_list = list(tokens)
    if len(token_list) < 2:
        return
    for start in range(0, len(token_list) - 1, block_size):
        end = min(start + block_size, len(token_list) - 1)
        input_ids = token_list[start:end]
        labels = token_list[start + 1 : end + 1]
        if not labels:
            continue
        yield torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def evaluate_gpt2(
    model_name: str,
    text_path: Path,
    json_text_key: str,
    max_documents: Optional[int],
    block_size: int,
    device: torch.device,
) -> Dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_raw_tokens = 0
    processed_docs = 0
    with torch.no_grad():
        for doc_id, text in iter_text_documents(text_path, json_text_key=json_text_key):
            if max_documents is not None and processed_docs >= max_documents:
                break
            processed_docs += 1
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_raw_tokens += len(tokens)
            for input_ids, labels in chunk_tokens(tokens, block_size):
                input_ids = input_ids.unsqueeze(0).to(device)
                labels = labels.unsqueeze(0).to(device)
                outputs = model(input_ids, labels=labels)
                token_count = labels.size(1)
                total_tokens += token_count
                total_loss += outputs.loss.item() * token_count

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    return {
        "documents": processed_docs,
        "tokens_used_for_loss": total_tokens,
        "raw_tokens_total": total_raw_tokens,
        "block_size": block_size,
        "average_loss": avg_loss,
        "perplexity": perplexity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare manifold LM vs GPT-2 perplexity.")
    parser.add_argument("--manifold-model", type=Path, required=True, help="Path to manifold LM directory (HF format).")
    parser.add_argument("--manifold-dataset", type=Path, required=True, help="Path to Hugging Face dataset built from manifold signatures.")
    parser.add_argument("--manifold-vocab", type=Path, required=True, help="Path to vocab.json emitted by the dataset builder.")
    parser.add_argument("--manifold-eval-fraction", type=float, default=0.1, help="Fraction of manifold dataset to evaluate if --manifold-max-samples is not set.")
    parser.add_argument("--manifold-max-samples", type=int, help="Optional cap on number of manifold sequences to evaluate.")
    parser.add_argument("--manifold-batch-size", type=int, default=8)
    parser.add_argument("--gpt2-model", type=str, default="gpt2-medium", help="Hugging Face model id for the baseline.")
    parser.add_argument("--raw-text", type=Path, required=True, help="Raw text corpus (JSON/JSONL/txt) for GPT-2 evaluation.")
    parser.add_argument("--json-text-key", type=str, default="text", help="Field that contains text within JSON/JSONL corpora.")
    parser.add_argument("--gpt2-block-size", type=int, default=1024, help="Chunk size (in tokens) for GPT-2 loss evaluation.")
    parser.add_argument("--gpt2-max-documents", type=int, help="Optional cap on documents when evaluating GPT-2.")
    parser.add_argument("--output", type=Path, help="Optional JSON file to write results.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    manifold_stats = evaluate_manifold(
        model_path=args.manifold_model,
        dataset_path=args.manifold_dataset,
        vocab_path=args.manifold_vocab,
        eval_fraction=args.manifold_eval_fraction,
        max_samples=args.manifold_max_samples,
        batch_size=args.manifold_batch_size,
        device=device,
        seed=args.seed,
    )

    gpt2_stats = evaluate_gpt2(
        model_name=args.gpt2_model,
        text_path=args.raw_text,
        json_text_key=args.json_text_key,
        max_documents=args.gpt2_max_documents,
        block_size=args.gpt2_block_size,
        device=device,
    )

    compression_ratio = None
    if manifold_stats["tokens_evaluated"] and gpt2_stats["raw_tokens_total"]:
        compression_ratio = gpt2_stats["raw_tokens_total"] / manifold_stats["tokens_evaluated"]

    summary = {
        "manifold_eval": manifold_stats,
        "gpt2_eval": gpt2_stats,
        "compression_proxy": {
            "raw_tokens_over_manifold_tokens": compression_ratio,
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
