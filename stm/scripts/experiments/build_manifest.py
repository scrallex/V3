#!/usr/bin/env python3
"""
Build deterministic dataset manifests for the optical/structural compression study.

The script scans image and transcript directories, pairs files by stem, and emits a JSONL
manifest that downstream benchmarks can consume. It is intentionally minimal so the
manifests can be regenerated on any machine without additional dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_SPLIT = "test"


def parse_metadata(values: Iterable[str]) -> Dict[str, str]:
    """Parse repeated `key=value` metadata arguments."""
    metadata: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"metadata entry must be key=value, got '{item}'")
        key, value = item.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def collect_pairs(image_dir: Path, text_dir: Path, suffixes: Tuple[str, ...]) -> List[Tuple[Path, Path]]:
    """Return sorted pairs of (image_path, text_path) sharing the same stem."""
    images = {
        path.stem: path
        for path in image_dir.glob("**/*")
        if path.is_file() and path.suffix.lower() in suffixes
    }
    texts = {
        path.stem: path
        for path in text_dir.glob("**/*")
        if path.is_file() and path.suffix.lower() in {".txt", ".json", ".md"}
    }
    missing_images = sorted(set(texts) - set(images))
    missing_texts = sorted(set(images) - set(texts))
    if missing_images:
        raise FileNotFoundError(f"missing images for stems: {', '.join(missing_images[:5])}")
    if missing_texts:
        raise FileNotFoundError(f"missing transcripts for stems: {', '.join(missing_texts[:5])}")
    pairs = [(images[stem], texts[stem]) for stem in sorted(images)]
    return pairs


def make_record(
    doc_id: str,
    page_idx: int,
    image_path: Path,
    text_path: Path,
    split: str,
    metadata: Dict[str, str],
) -> Dict[str, object]:
    rel_image = image_path.as_posix()
    rel_text = text_path.as_posix()
    record = {
        "doc_id": doc_id,
        "page_id": page_idx,
        "image_path": rel_image,
        "text_path": rel_text,
        "split": split,
    }
    if metadata:
        record["metadata"] = metadata
    return record


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build optical/structural dataset manifest")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing page images")
    parser.add_argument("--text-dir", type=Path, required=True, help="Directory containing transcripts")
    parser.add_argument("--doc-id", type=str, required=True, help="Document identifier for all records")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="Data split label (default: test)")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL manifest")
    parser.add_argument(
        "--metadata",
        type=str,
        action="append",
        default=[],
        help="Optional metadata entries (repeatable key=value)",
    )
    parser.add_argument(
        "--image-suffixes",
        type=str,
        default=".png,.jpg,.jpeg",
        help="Comma-separated list of recognised image suffixes",
    )
    args = parser.parse_args(argv)

    image_dir: Path = args.image_dir.resolve()
    text_dir: Path = args.text_dir.resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(f"image directory not found: {image_dir}")
    if not text_dir.is_dir():
        raise NotADirectoryError(f"text directory not found: {text_dir}")

    suffixes = tuple(item.strip().lower() for item in args.image_suffixes.split(",") if item.strip())
    metadata = parse_metadata(args.metadata)

    pairs = collect_pairs(image_dir, text_dir, suffixes)
    if not pairs:
        raise RuntimeError("no matching image/text pairs found")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for page_idx, (image_path, text_path) in enumerate(pairs):
            record = make_record(args.doc_id, page_idx, image_path.relative_to(Path.cwd()), text_path.relative_to(Path.cwd()), args.split, metadata)
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")

    try:
        output_rel = output_path.relative_to(Path.cwd())
    except ValueError:
        output_rel = output_path

    print(json.dumps({"output": str(output_rel), "records": len(pairs), "doc_id": args.doc_id}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
