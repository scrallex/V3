#!/usr/bin/env python3
"""Fine-tune vision-language OCR models on the manifold benchmark manifests."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import inspect

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    TrOCRProcessor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.manifold_compression_eval import (  # noqa: E402
    TokenizerRuntime,
    compute_character_metrics,
    compute_token_metrics,
)

try:  # pragma: no cover - peft is optional until LoRA is requested
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - defer error until user asks for LoRA
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]


@dataclass
class DatasetSpec:
    label: str
    manifest: Path
    image_root: Path


@dataclass
class SampleRecord:
    dataset: str
    doc_id: str
    subset: Optional[str]
    language: Optional[str]
    image_path: Path
    text_path: Path
    text: str


class OCRManifestDataset(Dataset):
    def __init__(self, records: Sequence[SampleRecord]) -> None:
        self._records = list(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self._records[idx]
        with Image.open(record.image_path) as img:
            image = img.convert("RGB")
        return {
            "image": image,
            "text": record.text,
            "doc_id": record.doc_id,
            "dataset": record.dataset,
            "language": record.language or "",
            "subset": record.subset or "",
        }


class VisionSeqCollator:
    def __init__(
        self,
        processor,
        max_target_length: int,
        pad_to_max: bool,
        ignore_pad_token_for_loss: bool,
    ) -> None:
        self.processor = processor
        self.max_target_length = max_target_length
        self.pad_to_max = pad_to_max
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        tokenizer = getattr(processor, "tokenizer", None)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in features]
        texts = [str(item["text"]) for item in features]
        batch = self.processor(
            images=images,
            text=texts,
            padding="max_length" if self.pad_to_max else True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = batch.get("labels")
        if labels is None and "input_ids" in batch:
            labels = batch["input_ids"]
        if labels is None:
            raise RuntimeError("Processor did not return labels; ensure it supports image+text inputs.")
        if self.ignore_pad_token_for_loss and self.pad_token_id is not None:
            labels = labels.clone()
            labels[labels == self.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def parse_dataset_specs(args: Optional[Iterable[str]]) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    if not args:
        return specs
    for raw in args:
        if "=" not in raw or ":" not in raw:
            raise ValueError("Dataset spec must look like label=/path/to/manifest.jsonl:/path/to/images")
        label, remainder = raw.split("=", 1)
        manifest_str, image_root_str = remainder.split(":", 1)
        manifest = Path(manifest_str).expanduser().resolve()
        image_root = Path(image_root_str).expanduser().resolve()
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found for dataset '{label}': {manifest}")
        if not image_root.exists():
            raise FileNotFoundError(f"Image root not found for dataset '{label}': {image_root}")
        specs.append(DatasetSpec(label=label, manifest=manifest, image_root=image_root))
    return specs


def resolve_text_path(raw_path: str, text_root: Path, manifest_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    for base in (manifest_dir, text_root, REPO_ROOT):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (text_root / candidate).resolve()


def resolve_image_path(raw_path: Optional[str], spec: DatasetSpec) -> Path:
    if not raw_path:
        raise ValueError(f"Manifest record missing image/source path for dataset '{spec.label}'")
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    resolved = (spec.image_root / candidate).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Image not found for dataset '{spec.label}': {resolved}")
    return resolved


def load_manifest_records(
    spec: DatasetSpec,
    text_root: Path,
    include_languages: Optional[Sequence[str]],
    include_subsets: Optional[Sequence[str]],
    limit: Optional[int],
    seed: int,
) -> List[SampleRecord]:
    raw_records: List[Dict[str, object]] = []
    with spec.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            raw_records.append(raw)
    random.Random(seed).shuffle(raw_records)
    if limit is not None:
        raw_records = raw_records[:limit]

    samples: List[SampleRecord] = []
    for record in raw_records:
        language = str(record.get("language") or "").lower() or None
        subset = str(record.get("subset") or "") or None
        if include_languages and (language is None or language not in include_languages):
            continue
        if include_subsets and (subset is None or subset not in include_subsets):
            continue

        text_path = resolve_text_path(str(record["text_path"]), text_root, spec.manifest.parent)
        if not text_path.exists():
            raise FileNotFoundError(f"Text file missing for dataset '{spec.label}': {text_path}")
        image_key = record.get("image_path") or record.get("source_path")
        image_path = resolve_image_path(image_key if isinstance(image_key, str) else None, spec)
        text = text_path.read_text(encoding="utf-8").strip()
        doc_id = str(record.get("doc_id") or text_path.stem)
        samples.append(
            SampleRecord(
                dataset=spec.label,
                doc_id=doc_id,
                subset=subset,
                language=language,
                image_path=image_path,
                text_path=text_path,
                text=text,
            )
        )
    return samples


def split_train_val(
    records: List[SampleRecord],
    val_ratio: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    if not records:
        return [], []
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio)) if val_ratio > 0 else 0
    if val_size == 0:
        return shuffled, []
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    if not train_records:
        raise ValueError("Validation split consumed all samples; decrease --val-split or add more data.")
    return train_records, val_records


def aggregate_metrics(rows: List[Dict[str, float]], key_prefix: str) -> Dict[str, float]:
    original = sum(row[f"original_{key_prefix}"] for row in rows)
    reconstructed = sum(row[f"reconstructed_{key_prefix}"] for row in rows)
    edit_distance = sum(row["edit_distance"] for row in rows)
    denom = max(original, reconstructed, 1)
    accuracy = 1.0 - (edit_distance / denom)
    precision = 1.0 - (edit_distance / max(reconstructed, 1))
    recall = 1.0 - (edit_distance / max(original, 1))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    normalized = edit_distance / denom
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "normalized_distance": normalized,
    }


def build_metrics_fn(
    processor,
    metric_tokenizer: Optional[TokenizerRuntime],
) -> callable:
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    def compute_metrics_fn(eval_preds) -> Dict[str, float]:
        predictions = eval_preds.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pred_ids = np.argmax(predictions, axis=-1) if predictions.ndim == 3 else predictions
        label_ids = eval_preds.label_ids
        if pad_token_id is not None:
            label_ids = np.where(label_ids != -100, label_ids, pad_token_id)
        pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True) if tokenizer else []
        label_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True) if tokenizer else []

        char_metrics: List[Dict[str, float]] = []
        token_metrics: List[Dict[str, float]] = []
        for reference, prediction in zip(label_texts, pred_texts):
            char_metrics.append(compute_character_metrics(reference, prediction))
            if metric_tokenizer is not None:
                token_metrics.append(compute_token_metrics(reference, prediction, metric_tokenizer))

        summary: Dict[str, float] = {}
        if char_metrics:
            aggregated = aggregate_metrics(char_metrics, "characters")
            summary.update(
                {
                    "character_accuracy": aggregated["accuracy"],
                    "character_precision": aggregated["precision"],
                    "character_recall": aggregated["recall"],
                    "character_f1": aggregated["f1"],
                    "character_normalized_edit_distance": aggregated["normalized_distance"],
                }
            )
        if token_metrics:
            aggregated = aggregate_metrics(token_metrics, "tokens")
            summary.update(
                {
                    "token_accuracy": aggregated["accuracy"],
                    "token_precision": aggregated["precision"],
                    "token_recall": aggregated["recall"],
                    "token_f1": aggregated["f1"],
                }
            )
        return summary

    return compute_metrics_fn


def prepare_model_and_processor(
    model_id: str,
    processor_id: Optional[str],
    trust_remote_code: bool,
    torch_dtype: Optional[torch.dtype],
    gradient_checkpointing: bool,
) -> Tuple[AutoModelForVision2Seq, AutoProcessor, Optional[int]]:
    processor_name = processor_id or model_id
    try:
        processor = TrOCRProcessor.from_pretrained(processor_name, trust_remote_code=trust_remote_code)
    except Exception:
        processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=trust_remote_code)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(processor_id or model_id, trust_remote_code=trust_remote_code)
        except Exception as exc:
            raise RuntimeError(
                "Loaded processor does not expose a tokenizer and a fallback tokenizer could not be loaded."
            ) from exc
        setattr(processor, "tokenizer", tokenizer)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token_id is not None:
            tokenizer.pad_token = tokenizer.bos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.decoder_start_token_id is None and tokenizer.bos_token_id is not None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    decoder_config = getattr(model.config, "decoder", None)
    max_decoder_positions = None
    if decoder_config is not None:
        max_decoder_positions = getattr(decoder_config, "max_position_embeddings", None)
    return model, processor, max_decoder_positions


def maybe_wrap_with_lora(
    model: AutoModelForVision2Seq,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str],
    bias: str,
) -> AutoModelForVision2Seq:
    if lora_rank <= 0:
        return model
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("peft is required for LoRA training. Install it or disable --lora-rank.")
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type="SEQ_2_SEQ_LM",
        target_modules=list(target_modules),
    )
    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters()
    return lora_model


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or fine-tune an OCR VLM on benchmark manifests.")
    parser.add_argument("--train-dataset", action="append", required=True, help="Format: label=manifest.jsonl:image_root_dir")
    parser.add_argument("--val-dataset", action="append", help="Optional validation datasets (same format as --train-dataset)")
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction of training data to reserve for validation when no --val-dataset is provided")
    parser.add_argument("--text-root", type=Path, default=REPO_ROOT, help="Base directory for relative text_path entries")
    parser.add_argument("--include-language", action="append", help="Whitelist of language codes to keep (lowercase)")
    parser.add_argument("--include-subset", action="append", help="Whitelist of subset names to keep")
    parser.add_argument("--max-train-samples", type=int, help="Cap total training samples after merging datasets")
    parser.add_argument("--max-val-samples", type=int, help="Cap total validation samples after merging datasets")
    parser.add_argument("--model-id", type=str, default="microsoft/trocr-base-printed", help="Hugging Face model id to fine-tune")
    parser.add_argument("--processor-id", type=str, help="Optional processor id; defaults to --model-id")
    parser.add_argument("--metric-tokenizer", type=str, default="gpt2", help="Tokenizer used for token-level evaluation metrics")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote code for model/processor/tokenizer loading")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store checkpoints and logs")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument("--train-batch-size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Per-device eval batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--eval-accumulation-steps", type=int, default=1, help="GPU-friendly chunking for evaluation loop")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio for the LR scheduler")
    parser.add_argument("--scheduler", type=str, default="cosine", help="LR scheduler type (linear, cosine, constant, etc.)")
    parser.add_argument("--max-target-length", type=int, default=512, help="Target text max length for tokenization")
    parser.add_argument("--generation-max-length", type=int, default=768, help="Max length for generation during eval")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dataloader-workers", type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument("--logging-steps", type=int, default=25, help="Interval (in steps) for logging metrics")
    parser.add_argument("--save-steps", type=int, default=200, help="Interval (in steps) for checkpointing")
    parser.add_argument("--eval-steps", type=int, default=200, help="Interval (in steps) for evaluation when enabled")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Maximum checkpoints to keep")
    parser.add_argument("--report-to", type=str, default="tensorboard", help="Logging backends, e.g., tensorboard or none")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Stop training after N evals without improvement (0 disables)")
    parser.add_argument("--metric-for-best-model", type=str, default="character_accuracy", help="Metric name that drives best-model selection")
    parser.add_argument("--greater-is-better", action="store_true", help="Set if the tracked metric should increase; otherwise it is minimized")
    parser.add_argument("--resume-from-checkpoint", type=str, help="Path to resume training from")
    parser.add_argument("--seed", type=int, default=17, help="Global random seed")
    parser.add_argument("--lora-rank", type=int, default=0, help="Enable LoRA adapters with the given rank (0 disables)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,out_proj",
        help="Comma-separated module name fragments to receive LoRA adapters",
    )
    parser.add_argument("--lora-bias", type=str, default="none", help="Bias handling for LoRA (none, all, lora_only)")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_global_seed(args.seed)

    train_specs = parse_dataset_specs(args.train_dataset)
    val_specs = parse_dataset_specs(args.val_dataset)
    if not train_specs:
        raise ValueError("At least one --train-dataset entry is required.")

    text_root = args.text_root.expanduser().resolve()
    include_languages = [lang.lower() for lang in args.include_language] if args.include_language else None
    include_subsets = args.include_subset

    train_records: List[SampleRecord] = []
    for spec in train_specs:
        train_records.extend(
            load_manifest_records(
                spec,
                text_root=text_root,
                include_languages=include_languages,
                include_subsets=include_subsets,
                limit=None,
                seed=args.seed,
            )
        )
    if not train_records:
        raise ValueError("No training samples were loaded. Check dataset filters and manifests.")

    val_records: List[SampleRecord] = []
    if val_specs:
        for spec in val_specs:
            val_records.extend(
                load_manifest_records(
                    spec,
                    text_root=text_root,
                    include_languages=include_languages,
                    include_subsets=include_subsets,
                    limit=None,
                    seed=args.seed + 1,
                )
            )
    elif args.val_split > 0:
        train_records, val_records = split_train_val(train_records, args.val_split, args.seed)

    if args.max_train_samples and len(train_records) > args.max_train_samples:
        random.Random(args.seed).shuffle(train_records)
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples and len(val_records) > args.max_val_samples:
        random.Random(args.seed + 2).shuffle(val_records)
        val_records = val_records[: args.max_val_samples]

    print(f"[data] training samples: {len(train_records)}")
    print(f"[data] validation samples: {len(val_records)}")

    model, processor, decoder_max_positions = prepare_model_and_processor(
        model_id=args.model_id,
        processor_id=args.processor_id,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None),
        gradient_checkpointing=args.gradient_checkpointing,
    )

    target_length = args.max_target_length
    generation_length = args.generation_max_length
    if decoder_max_positions is not None:
        if target_length > decoder_max_positions:
            print(
                f"[warn] max_target_length {target_length} exceeds decoder limit {decoder_max_positions}; clamping.",
                flush=True,
            )
            target_length = decoder_max_positions
        if generation_length > decoder_max_positions:
            print(
                f"[warn] generation_max_length {generation_length} exceeds decoder limit {decoder_max_positions}; clamping.",
                flush=True,
            )
            generation_length = decoder_max_positions
    model = maybe_wrap_with_lora(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[module.strip() for module in args.lora_target_modules.split(",") if module.strip()],
        bias=args.lora_bias,
    )

    train_dataset = OCRManifestDataset(train_records)
    eval_dataset = OCRManifestDataset(val_records) if val_records else None
    collator = VisionSeqCollator(
        processor=processor,
        max_target_length=target_length,
        pad_to_max=True,
        ignore_pad_token_for_loss=True,
    )

    tokenizer_runtime = TokenizerRuntime(args.metric_tokenizer, args.trust_remote_code) if args.metric_tokenizer else None
    compute_metrics_fn = build_metrics_fn(processor, tokenizer_runtime) if eval_dataset else None

    evaluation_strategy = "no"
    eval_steps = None
    load_best_model = False
    if eval_dataset:
        evaluation_strategy = "steps" if args.eval_steps else "epoch"
        eval_steps = args.eval_steps if args.eval_steps else None
        load_best_model = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args_kwargs = {
        "do_eval": eval_dataset is not None,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "eval_steps": eval_steps,
        "logging_steps": args.logging_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.scheduler,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "num_train_epochs": args.epochs,
        "output_dir": str(args.output_dir),
        "fp16": args.fp16,
        "bf16": args.bf16,
        "dataloader_num_workers": args.dataloader_workers,
        "report_to": None if args.report_to.lower() == "none" else args.report_to,
        "save_total_limit": args.save_total_limit,
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": args.greater_is_better,
        "load_best_model_at_end": load_best_model,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
    }

    training_args_sig = inspect.signature(TrainingArguments.__init__).parameters

    training_args_sig = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_args_sig:
        training_args_kwargs["evaluation_strategy"] = evaluation_strategy
    elif "eval_strategy" in training_args_sig:
        training_args_kwargs["eval_strategy"] = evaluation_strategy
    else:
        training_args_kwargs["eval_strategy"] = evaluation_strategy

    if "predict_with_generate" in training_args_sig:
        training_args_kwargs["predict_with_generate"] = eval_dataset is not None
    if "generation_max_length" in training_args_sig:
        training_args_kwargs["generation_max_length"] = generation_length

    training_args = TrainingArguments(**training_args_kwargs)

    callbacks = []
    if args.early_stopping_patience > 0 and eval_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    main()
