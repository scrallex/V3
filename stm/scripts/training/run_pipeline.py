#!/usr/bin/env python3
"""Stageable helper for downloading corpora and launching OCR training."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_BIN = os.environ.get("PYTHON", sys.executable)

FOX_MANIFEST = REPO_ROOT / "data" / "benchmark_corpus" / "fox" / "metadata" / "text_manifest.jsonl"
FOX_RAW = REPO_ROOT / "data" / "benchmark_corpus" / "fox" / "raw"
OMNI_MANIFEST = REPO_ROOT / "data" / "benchmark_corpus" / "omnidocbench" / "metadata" / "text_manifest.jsonl"
OMNI_RAW = REPO_ROOT / "data" / "benchmark_corpus" / "omnidocbench" / "raw" / "OmniDocBench"


def _run(command: List[str]) -> None:
    print(f"[run] {' '.join(str(part) for part in command)}")
    subprocess.run(command, check=True)


def _strip_remainder(args: List[str]) -> List[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _split_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


def stage_download(stage_args: List[str]) -> None:
    command = [PYTHON_BIN, str(REPO_ROOT / "scripts" / "data" / "download_corpora.py"), *stage_args]
    _run(command)


def stage_train(stage_args: List[str]) -> None:
    run_name = os.environ.get("RUN_NAME", "fox_omni_lora")
    output_dir = Path(os.environ.get("OUTPUT_DIR", REPO_ROOT / "output" / "training_runs" / run_name)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    default_train = ",".join(
        [
            f"fox_en={FOX_MANIFEST}:{FOX_RAW}",
            f"fox_cn={FOX_MANIFEST}:{FOX_RAW}",
            f"omnidoc={OMNI_MANIFEST}:{OMNI_RAW}",
        ]
    )
    train_spec = os.environ.get("TRAIN_DATASETS", default_train)
    train_datasets = [entry for entry in (token.strip() for token in train_spec.split(",")) if entry]

    cmd: List[str] = [
        PYTHON_BIN,
        str(REPO_ROOT / "scripts" / "training" / "ocr_trainer.py"),
        "--output-dir",
        str(output_dir),
        "--epochs",
        os.environ.get("EPOCHS", "3"),
        "--train-batch-size",
        os.environ.get("TRAIN_BATCH_SIZE", "1"),
        "--eval-batch-size",
        os.environ.get("EVAL_BATCH_SIZE", "1"),
        "--gradient-accumulation",
        os.environ.get("GRAD_ACCUM", "8"),
        "--eval-accumulation-steps",
        os.environ.get("EVAL_ACCUM_STEPS", "1"),
        "--learning-rate",
        os.environ.get("LR", "1e-4"),
        "--weight-decay",
        os.environ.get("WEIGHT_DECAY", "0.01"),
        "--warmup-ratio",
        os.environ.get("WARMUP_RATIO", "0.05"),
        "--model-id",
        os.environ.get("MODEL_ID", "microsoft/trocr-base-printed"),
        "--max-target-length",
        os.environ.get("MAX_TARGET_LEN", "640"),
        "--generation-max-length",
        os.environ.get("GEN_MAX_LEN", "768"),
        "--logging-steps",
        os.environ.get("LOGGING_STEPS", "25"),
        "--save-steps",
        os.environ.get("SAVE_STEPS", "200"),
        "--eval-steps",
        os.environ.get("EVAL_STEPS", "200"),
        "--save-total-limit",
        os.environ.get("SAVE_LIMIT", "2"),
        "--report-to",
        os.environ.get("REPORT_TO", "tensorboard"),
        "--lora-rank",
        os.environ.get("LORA_RANK", "8"),
        "--lora-alpha",
        os.environ.get("LORA_ALPHA", "16"),
        "--lora-dropout",
        os.environ.get("LORA_DROPOUT", "0.05"),
        "--lora-target-modules",
        os.environ.get("LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,out_proj"),
        "--metric-tokenizer",
        os.environ.get("METRIC_TOKENIZER", "gpt2"),
    ]

    processor_id = os.environ.get("PROCESSOR_ID")
    if processor_id:
        cmd.extend(["--processor-id", processor_id])

    if os.environ.get("USE_GRADIENT_CHECKPOINTING", "1") != "0":
        cmd.append("--gradient-checkpointing")

    precision = os.environ.get("PRECISION", "fp16").lower().strip()
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "bf16":
        cmd.append("--bf16")

    languages = _split_list(os.environ.get("INCLUDE_LANGUAGES"))
    for language in languages:
        cmd.extend(["--include-language", language])

    subsets = _split_list(os.environ.get("INCLUDE_SUBSETS"))
    for subset in subsets:
        cmd.extend(["--include-subset", subset])

    for dataset in train_datasets:
        cmd.extend(["--train-dataset", dataset])

    val_datasets = _split_list(os.environ.get("VAL_DATASETS"))
    if val_datasets:
        for dataset in val_datasets:
            cmd.extend(["--val-dataset", dataset])
    else:
        cmd.extend(["--val-split", os.environ.get("VAL_SPLIT", "0.08")])

    resume_from = os.environ.get("RESUME_FROM")
    if resume_from:
        cmd.extend(["--resume-from-checkpoint", resume_from])

    extra_args = os.environ.get("EXTRA_ARGS")
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    cmd.extend(stage_args)

    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download corpora and launch training.")
    parser.add_argument(
        "stage",
        choices=("download", "prepare", "train", "all"),
        help="Which stage to run.",
    )
    parser.add_argument(
        "stage_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying stage (prefix with --).",
    )
    args = parser.parse_args()
    remainder = _strip_remainder(list(args.stage_args))

    if args.stage == "download":
        stage_download(remainder)
    elif args.stage == "prepare":
        stage_download(["--prepare-only", *remainder])
    elif args.stage == "train":
        stage_train(remainder)
    elif args.stage == "all":
        stage_download([])
        stage_train(remainder)
    else:
        parser.error(f"Unknown stage {args.stage}")


if __name__ == "__main__":
    main()
