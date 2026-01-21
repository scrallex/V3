#!/usr/bin/env python3
"""
Run DeepSeek-OCR inference over a dataset manifest.

By default this script uses Hugging Face `pipeline("image-to-text")` with the given model id.
For larger workloads you can override the execution mode with `--external-command`, which
should be a printf-style template receiving `{image}`, `{doc_id}`, `{page_id}`, and `{prompt}`.
The command must write the decoded text to stdout.
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch is required for default pipeline mode
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_pipeline(model_id: str, device: str, dtype: str) -> "ImageToTextPipeline":
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError("PyTorch is required for pipeline mode") from TORCH_IMPORT_ERROR
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as exc:  # pragma: no cover - transformers required for default mode
        raise RuntimeError("transformers is required for pipeline mode") from exc

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype.lower())
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from float32,float16,bfloat16.")

    generator = pipeline(
        "image-to-text",
        model=model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    return generator


def run_external_command(template: str, image_path: Path, prompt: str, doc_id: str, page_id: int) -> str:
    command = template.format(image=image_path.as_posix(), prompt=prompt, doc_id=doc_id, page_id=page_id)
    result = subprocess.run(shlex.split(command), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def run_pipeline(
    generator,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
) -> str:
    generate_kwargs: Dict[str, object] = {"max_new_tokens": max_new_tokens}
    if temperature is not None:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["do_sample"] = temperature > 0.0
    else:
        generate_kwargs["do_sample"] = False
    if top_p is not None:
        generate_kwargs["top_p"] = top_p
    outputs = generator(image_path.as_posix(), prompt=prompt, **generate_kwargs)
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict) and "generated_text" in first:
            return str(first["generated_text"]).strip()
        return str(first).strip()
    if isinstance(outputs, dict) and "generated_text" in outputs:
        return str(outputs["generated_text"]).strip()
    return str(outputs).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR inference over a manifest")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest with image/text paths")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file with OCR outputs")
    parser.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-OCR", help="Hugging Face model id")
    parser.add_argument("--device", type=str, default="cpu", help="torch device spec (e.g., cuda:0 or cpu)")
    parser.add_argument("--dtype", type=str, default="float16", help="torch dtype for generation")
    parser.add_argument("--prompt", type=str, default="Please transcribe the document content.", help="Prompt string")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum tokens to generate per sample")
    parser.add_argument("--temperature", type=float, help="Sampling temperature; omit for greedy decoding")
    parser.add_argument("--top-p", type=float, help="Top-p nucleus sampling parameter")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--external-command",
        type=str,
        help="Optional external command template overriding pipeline execution",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    manifest_records = load_manifest(args.manifest.resolve())

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.external_command:
        generator = None
    else:
        generator = load_pipeline(args.model_id, args.device, args.dtype)

    with output_path.open("w", encoding="utf-8") as fh:
        for record in manifest_records:
            doc_id = str(record["doc_id"])
            page_id = int(record["page_id"])
            image_path = (Path.cwd() / Path(record["image_path"])).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            if generator is None:
                decoded_text = run_external_command(args.external_command, image_path, args.prompt, doc_id, page_id)
            else:
                decoded_text = run_pipeline(generator, image_path, args.prompt, args.max_new_tokens, args.temperature, args.top_p)

            output_record = {
                "doc_id": doc_id,
                "page_id": page_id,
                "text": decoded_text,
                "prompt": args.prompt,
                "model_id": args.model_id if generator is not None else "external-command",
                "image_path": str(image_path.relative_to(Path.cwd())),
            }
            fh.write(json.dumps(output_record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
