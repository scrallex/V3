#!/usr/bin/env python3
"""Train a GPT-style decoder on manifold signature sequences."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import Dataset, load_from_disk  # type: ignore  # noqa: E402
from transformers import (  # type: ignore  # noqa: E402
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint  # type: ignore  # noqa: E402


@dataclass
class ManifoldDataCollator:
    pad_token_id: int

    def __call__(self, features: List[dict]) -> dict:
        max_length = max(len(f["input_ids"]) for f in features)
        batch_input_ids: List[List[int]] = []
        batch_labels: List[List[int]] = []
        batch_attention: List[List[int]] = []
        for feature in features:
            ids = list(feature["input_ids"])
            labels = list(feature["labels"])
            length = len(ids)
            pad_len = max_length - length
            if pad_len:
                ids = ids + [self.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
            attention = [1] * length + [0] * pad_len
            batch_input_ids.append(ids)
            batch_labels.append(labels)
            batch_attention.append(attention)
        input_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
        label_tensor = torch.tensor(batch_labels, dtype=torch.long)
        attention_tensor = torch.tensor(batch_attention, dtype=torch.long)
        return {
            "input_ids": input_tensor,
            "labels": label_tensor,
            "attention_mask": attention_tensor,
        }


def load_metadata(dataset_path: Path) -> dict:
    metadata_path = dataset_path.parent / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def prepare_datasets(
    dataset_path: Path,
    eval_holdout: float,
    seed: int,
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
) -> Tuple[Dataset, Optional[Dataset]]:
    dataset = load_from_disk(str(dataset_path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset at {dataset_path}, found {type(dataset)}")
    shuffled = dataset.shuffle(seed=seed)
    if eval_holdout > 0.0:
        split = shuffled.train_test_split(test_size=eval_holdout, seed=seed)
        train_dataset = split["train"]
        eval_dataset: Optional[Dataset] = split["test"]
    else:
        train_dataset = shuffled
        eval_dataset = None
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if eval_dataset is not None and max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))
    return train_dataset, eval_dataset


def load_vocab(vocab_path: Path) -> Tuple[List[str], int]:
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    signatures = list(payload.get("signatures", []))
    return signatures, len(signatures)


def compute_training_overview(
    train_size: int,
    seq_len: int,
    batch_size: int,
    grad_accum: int,
    epochs: float,
) -> dict:
    tokens_per_step = seq_len * batch_size * grad_accum
    steps_per_epoch = math.ceil(train_size / (batch_size * grad_accum))
    total_steps = math.ceil(steps_per_epoch * epochs)
    total_tokens = tokens_per_step * total_steps
    return {
        "tokens_per_step": tokens_per_step,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "total_tokens_seen": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GPT-style manifold LM on a single 3080 Ti.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to hf_dataset produced by prepare_causal_dataset.py")
    parser.add_argument("--vocab-path", type=Path, required=True, help="Path to vocab.json emitted by dataset prep")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints + logs")
    parser.add_argument("--n-layer", type=int, default=12, help="Transformer decoder layers (12 ≈124M params)")
    parser.add_argument("--n-head", type=int, default=12, help="Attention heads")
    parser.add_argument("--n-embd", type=int, default=768, help="Model width / embedding dim")
    parser.add_argument("--n-inner", type=int, help="Optional MLP hidden size (defaults to 4 * n_embd)")
    parser.add_argument("--context-length", type=int, default=512, help="Maximum manifold tokens per sample")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--fp16", action="store_true", help="Enable torch.float16 training")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--eval-holdout", type=float, default=0.02, help="Fraction of samples reserved for eval")
    parser.add_argument("--max-train-samples", type=int, help="Optional limit on training samples (debug)")
    parser.add_argument("--max-eval-samples", type=int, help="Optional limit on eval samples (debug)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in output-dir if present")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--report-to", nargs="*", default=["tensorboard"])
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    vocab_path = args.vocab_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, vocab_size = load_vocab(vocab_path)
    pad_token_id = vocab_size  # append pad token
    vocab_size_with_pad = vocab_size + 1

    train_dataset, eval_dataset = prepare_datasets(
        dataset_path,
        eval_holdout=args.eval_holdout,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    collator = ManifoldDataCollator(pad_token_id=pad_token_id)

    config = GPT2Config(
        vocab_size=vocab_size_with_pad,
        n_positions=args.context_length,
        n_ctx=args.context_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_inner or (4 * args.n_embd),
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        activation_function="gelu_new",
        bos_token_id=pad_token_id,
        eos_token_id=pad_token_id,
        pad_token_id=pad_token_id,
    )
    model = GPT2LMHeadModel(config)

    def training_args_supports(param: str) -> bool:
        return param in inspect.signature(TrainingArguments.__init__).parameters

    training_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "fp16": bool(args.fp16 and not args.bf16),
        "bf16": bool(args.bf16),
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": args.report_to,
        "seed": args.seed,
    }

    if training_args_supports("save_strategy"):
        training_kwargs["save_strategy"] = "steps"
    if training_args_supports("evaluation_strategy"):
        training_kwargs["evaluation_strategy"] = "steps" if eval_dataset is not None else "no"
    elif eval_dataset is not None and training_args_supports("evaluate_during_training"):
        training_kwargs["evaluate_during_training"] = True
    training_kwargs["do_eval"] = bool(eval_dataset)

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = get_last_checkpoint(training_args.output_dir)
        if resume_checkpoint:
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            print("No checkpoint found; starting fresh.")

    overview = compute_training_overview(
        train_size=len(train_dataset),
        seq_len=args.context_length,
        batch_size=args.per_device_train_batch_size,
        grad_accum=args.gradient_accumulation_steps,
        epochs=args.num_train_epochs,
    )
    metadata = load_metadata(dataset_path)
    print(
        json.dumps(
            {
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
                "vocab_size": vocab_size_with_pad,
                "pad_token_id": pad_token_id,
                "model_config": {
                    "n_layer": args.n_layer,
                    "n_head": args.n_head,
                    "n_embd": args.n_embd,
                    "context_length": args.context_length,
                },
                "training_schedule": overview,
                "dataset_metadata": metadata,
            },
            indent=2,
        )
    )

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        perplexity = math.exp(metrics["eval_loss"]) if "eval_loss" in metrics else float("nan")
        metrics["perplexity"] = perplexity
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
