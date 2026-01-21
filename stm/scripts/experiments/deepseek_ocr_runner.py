#!/usr/bin/env python3
"""Run DeepSeek-OCR on benchmark datasets using local manifests."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _ensure_cuda_lib_paths() -> None:
    """Append pip-installed CUDA libraries to LD_LIBRARY_PATH when needed."""

    def try_add(package: str) -> None:
        try:
            module = importlib.import_module(package)
        except ImportError:
            return
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return
        candidate = Path(module_file).resolve().parent / "lib"
        if candidate.exists():
            current = os.environ.get("LD_LIBRARY_PATH", "")
            paths = [p for p in current.split(os.pathsep) if p]
            if str(candidate) not in paths:
                paths.insert(0, str(candidate))
                os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(paths)

    for pkg in (
        "nvidia.cuda_runtime",
        "nvidia.cudnn",
        "nvidia.cublas",
        "nvidia.cusolver",
        "nvidia.cusparse",
        "nvidia.cufft",
        "nvidia.curand",
        "nvidia.cusparselt",
        "nvidia.nccl",
        "nvidia.nvtx",
        "nvidia.nvjitlink",
    ):
        try_add(pkg)


_ensure_cuda_lib_paths()

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer  # type: ignore

# DeepSeek's HF adapter expects FlashAttention kernels; provide a fallback that
# aliases to a vanilla attention kernel when flash-attn isn't available.
try:  # pragma: no cover - import shim
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - best effort compatibility
    from transformers.models.llama import modeling_llama as _llama_mod  # type: ignore

    class _FallbackFlashAttention(_llama_mod.LlamaAttention):  # type: ignore
        """Fallback that reuses the baseline LLaMA attention implementation."""

    _llama_mod.LlamaFlashAttention2 = _FallbackFlashAttention  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.manifold_compression_eval import (  # noqa: E402
    TokenizerRuntime,
    compute_character_metrics,
    compute_token_metrics,
)


@dataclass
class DatasetTask:
    label: str
    manifest: Path
    image_root: Path


def parse_dataset_args(dataset_args: Iterable[str]) -> List[DatasetTask]:
    tasks: List[DatasetTask] = []
    for raw in dataset_args:
        if "=" not in raw or ":" not in raw:
            raise ValueError("Dataset specification must be LABEL=MANIFEST:IMAGE_ROOT")
        label, remainder = raw.split("=", 1)
        manifest_str, image_root_str = remainder.split(":", 1)
        manifest_path = Path(manifest_str).expanduser().resolve()
        image_root = Path(image_root_str).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for dataset '{label}': {manifest_path}")
        if not image_root.exists():
            raise FileNotFoundError(f"Image root not found for dataset '{label}': {image_root}")
        tasks.append(DatasetTask(label=label, manifest=manifest_path, image_root=image_root))
    if not tasks:
        raise ValueError("At least one --dataset entry is required.")
    return tasks


def load_manifest(manifest_path: Path, limit: Optional[int] = None) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if line.strip():
                records.append(json.loads(line))
            if limit is not None and idx + 1 >= limit:
                break
    return records


class DeepSeekWrapper:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: torch.dtype,
        attn_impl: str,
        trust_remote_code: bool,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self._scratch_dir = Path(os.environ.get("DEESEEK_OCR_TMP", "/tmp")) / "deepseek_ocr_runner"
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            _attn_implementation=attn_impl,
            use_safetensors=True,
        )
        self.model = self.model.eval().to(self.device)

    def infer(
        self,
        prompt: str,
        image_path: Path,
        output_dir: Optional[Path],
        base_size: int,
        image_size: int,
        crop_mode: bool,
        test_compress: bool,
        save_results: bool,
        eval_mode: bool,
    ) -> str:
        image = Image.open(image_path).convert("RGB")
        image.close()
        result = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=str(output_dir or self._scratch_dir),
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            test_compress=test_compress,
            save_results=save_results,
            eval_mode=eval_mode,
        )
        if isinstance(result, dict) and "text" in result:
            return str(result["text"])
        if isinstance(result, str) and result.strip():
            return result
        raise RuntimeError("DeepSeek-OCR inference did not return textual output. Enable --eval-mode.")


def aggregate_metrics(records: List[Dict[str, object]]) -> Dict[str, object]:
    total_tokens = 0
    total_reconstructed_tokens = 0
    total_token_edit_distance = 0
    total_characters = 0
    total_reconstructed_characters = 0
    total_char_edit_distance = 0

    for record in records:
        token_metrics = record["token_metrics"]
        character_metrics = record["character_metrics"]
        total_tokens += token_metrics["original_tokens"]
        total_reconstructed_tokens += token_metrics["reconstructed_tokens"]
        total_token_edit_distance += token_metrics["edit_distance"]
        total_characters += character_metrics["original_characters"]
        total_reconstructed_characters += character_metrics["reconstructed_characters"]
        total_char_edit_distance += character_metrics["edit_distance"]

    token_accuracy = 1.0 - (
        total_token_edit_distance / max(total_tokens, total_reconstructed_tokens, 1)
    )
    token_precision = 1.0 - (total_token_edit_distance / max(total_reconstructed_tokens, 1))
    token_recall = 1.0 - (total_token_edit_distance / max(total_tokens, 1))
    token_f1 = (2 * token_precision * token_recall / (token_precision + token_recall)) if (
        token_precision + token_recall
    ) else 0.0

    normalized_char_edit_distance = (
        total_char_edit_distance / max(total_characters, total_reconstructed_characters, 1)
    )
    character_accuracy = 1.0 - normalized_char_edit_distance
    character_precision = 1.0 - (total_char_edit_distance / max(total_reconstructed_characters, 1))
    character_recall = 1.0 - (total_char_edit_distance / max(total_characters, 1))
    character_f1 = (2 * character_precision * character_recall / (character_precision + character_recall)) if (
        character_precision + character_recall
    ) else 0.0

    return {
        "records": len(records),
        "token_metrics": {
            "text_tokens_total": total_tokens,
            "reconstructed_tokens_total": total_reconstructed_tokens,
            "token_edit_distance_total": total_token_edit_distance,
            "token_accuracy": token_accuracy,
            "token_precision": token_precision,
            "token_recall": token_recall,
            "token_f1": token_f1,
        },
        "character_metrics": {
            "original_characters_total": total_characters,
            "reconstructed_characters_total": total_reconstructed_characters,
            "character_edit_distance_total": total_char_edit_distance,
            "character_accuracy": character_accuracy,
            "character_precision": character_precision,
            "character_recall": character_recall,
            "character_f1": character_f1,
            "normalized_edit_distance": normalized_char_edit_distance,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepSeek-OCR on benchmark manifests.")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset specification as LABEL=MANIFEST_PATH:IMAGE_ROOT. Repeat for multiple datasets.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "deepseek_runs")
    parser.add_argument("--model-name", type=str, default=str(REPO_ROOT / "external" / "DeepSeek-OCR" / "weights"))
    parser.add_argument("--prompt", type=str, default="<image>\nFree OCR.")
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--crop-mode", action="store_true", help="Enable crop_mode for model.infer")
    parser.add_argument("--no-test-compress", action="store_true", help="Disable model infer compression testing")
    parser.add_argument("--save-results", action="store_true", help="Ask model to save visual artifacts")
    parser.add_argument("--attn-impl", type=str, default="sdpa", help="Attention implementation (sdpa or flash_attention_2)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model weights",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote code when loading the model")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--tokenizer-trust-remote-code", action="store_true")
    parser.add_argument("--max-records", type=int, help="Optional cap on records per dataset")
    parser.add_argument(
        "--no-eval-mode",
        action="store_true",
        help="Disable eval_mode to mirror the default DeepSeek demo behavior (disables text return).",
    )
    args = parser.parse_args()

    datasets = parse_dataset_args(args.dataset or [])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    wrapper = DeepSeekWrapper(
        model_name=args.model_name,
        device=args.device,
        dtype=dtype,
        attn_impl=args.attn_impl,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer_runtime = TokenizerRuntime(args.tokenizer, args.tokenizer_trust_remote_code)

    summary_payload: Dict[str, object] = {
        "model_name": args.model_name,
        "prompt": args.prompt,
        "base_size": args.base_size,
        "image_size": args.image_size,
        "crop_mode": args.crop_mode,
        "attn_impl": args.attn_impl,
        "dtype": args.dtype,
        "device": args.device,
        "eval_mode": not args.no_eval_mode,
        "records": {},
    }

    for dataset in datasets:
        manifest_records = load_manifest(dataset.manifest, limit=args.max_records)
        dataset_results: List[Dict[str, object]] = []

        for record in tqdm(manifest_records, desc=f"DeepSeek {dataset.label}", unit="doc"):
            text_path = REPO_ROOT / record["text_path"]
            image_path = dataset.image_root / record["source_path"]
            original_text = text_path.read_text(encoding="utf-8")
            prediction = wrapper.infer(
                prompt=args.prompt,
                image_path=image_path,
                output_dir=args.output_dir if args.save_results else None,
                base_size=args.base_size,
                image_size=args.image_size,
                crop_mode=args.crop_mode,
                test_compress=not args.no_test_compress,
                save_results=args.save_results,
                eval_mode=not args.no_eval_mode,
            )
            token_metrics = compute_token_metrics(original_text, prediction, tokenizer_runtime)
            character_metrics = compute_character_metrics(original_text, prediction)
            dataset_results.append(
                {
                    "doc_id": record.get("doc_id"),
                    "text_path": record.get("text_path"),
                    "image_path": str(image_path.relative_to(REPO_ROOT)),
                    "subset": record.get("subset"),
                    "language": record.get("language"),
                    "token_metrics": token_metrics,
                    "character_metrics": character_metrics,
                    "prediction": prediction,
                }
            )

        dataset_summary = aggregate_metrics(dataset_results)
        summary_payload["records"][dataset.label] = dataset_summary

        output_jsonl = args.output_dir / f"{dataset.label}.jsonl"
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for record in dataset_results:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    (args.output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
