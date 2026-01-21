#!/usr/bin/env python3
"""Download-aligned preprocessing for Fox and OmniDocBench benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class TextRecord:
    dataset: str
    subset: str
    doc_id: str
    text_path: Path
    source_path: str
    language: str | None
    category: str | None
    tokens: int
    characters: int


def load_tokenizer(name: str):
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = 1_000_000
    return tokenizer


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def write_manifest(path: Path, records: Sequence[TextRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "dataset": record.dataset,
                "subset": record.subset,
                "doc_id": record.doc_id,
                "text_path": str(record.text_path.relative_to(REPO_ROOT)),
                "source_path": record.source_path,
                "language": record.language,
                "category": record.category,
                "tokens": record.tokens,
                "characters": record.characters,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def prepare_fox(fox_root: Path, output_root: Path, tokenizer) -> List[TextRecord]:
    subsets = {
        "en_page_ocr": fox_root / "en_page_ocr.json",
        "cn_page_ocr": fox_root / "cn_page_ocr.json",
    }
    subset_image_dirs = {
        "en_page_ocr": "en_pdf_png",
        "cn_page_ocr": "cn_pdf_png",
    }
    records: List[TextRecord] = []
    for subset, json_path in subsets.items():
        if not json_path.exists():
            continue
        subset_dir = output_root / "text" / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        entries = json.loads(json_path.read_text())
        for entry in tqdm(entries, desc=f"FOX {subset}", unit="page"):
            conversations = entry.get("conversations", [])
            if len(conversations) < 2:
                continue
            target_text = conversations[1].get("value", "")
            target_text = target_text.replace("\r\n", "\n")
            image_name = Path(entry.get("image", f"{subset}_page")).stem
            doc_id = f"fox__{subset}__{image_name}"
            text_path = subset_dir / f"{image_name}.txt"
            text_path.write_text(target_text, encoding="utf-8")
            image_dir = subset_image_dirs.get(subset)
            if image_dir:
                source_rel = Path("focus_benchmark_test") / image_dir / entry.get("image", "")
            else:
                source_rel = Path("focus_benchmark_test") / entry.get("image", "")

            records.append(
                TextRecord(
                    dataset="fox",
                    subset=subset,
                    doc_id=doc_id,
                    text_path=text_path,
                    source_path=source_rel.as_posix(),
                    language="english" if subset.startswith("en_") else "chinese",
                    category="page_ocr",
                    tokens=token_count(tokenizer, target_text),
                    characters=len(target_text),
                )
            )
    return records


def sanitize_text(lines: Iterable[str]) -> str:
    return "\n".join(filter(None, (line.strip("\n") for line in lines)))


def prepare_omnidocbench(omni_root: Path, output_root: Path, tokenizer) -> List[TextRecord]:
    manifest_path = omni_root / "OmniDocBench.json"
    if not manifest_path.exists():
        return []
    records: List[TextRecord] = []
    entries = json.loads(manifest_path.read_text())
    for entry in tqdm(entries, desc="OmniDocBench", unit="page"):
        page_info = entry.get("page_info", {})
        attr = page_info.get("page_attribute", {})
        image_rel = Path(page_info.get("image_path", ""))
        if not image_rel:
            continue
        layout_dets = entry.get("layout_dets", [])
        def order_key(payload: dict) -> int:
            value = payload.get("order")
            if isinstance(value, (int, float)):
                return int(value)
            return sys.maxsize

        ordered = sorted(
            (item for item in layout_dets if item and not item.get("ignore")),
            key=order_key,
        )
        text_lines = [item.get("text", "").strip() for item in ordered if item.get("text", "").strip()]
        text_blob = sanitize_text(text_lines)
        text_path = (output_root / "text" / image_rel).with_suffix(".txt")
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text_blob, encoding="utf-8")
        doc_id = f"omnidoc__{image_rel.with_suffix('').as_posix().replace('/', '__')}"
        source_rel = Path("images") / image_rel
        records.append(
            TextRecord(
                dataset="omnidocbench",
                subset=str(attr.get("data_source", "unknown")),
                doc_id=doc_id,
                text_path=text_path,
                source_path=source_rel.as_posix(),
                language=str(attr.get("language")),
                category=str(attr.get("layout")),
                tokens=token_count(tokenizer, text_blob),
                characters=len(text_blob),
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Fox and OmniDocBench benchmark texts.")
    parser.add_argument("--fox-root", type=Path, help="Path to focus_benchmark_test directory")
    parser.add_argument("--fox-output", type=Path, default=REPO_ROOT / "data" / "benchmark_corpus" / "fox")
    parser.add_argument("--omnidoc-root", type=Path, help="Path to OmniDocBench dataset root")
    parser.add_argument(
        "--omnidoc-output",
        type=Path,
        default=REPO_ROOT / "data" / "benchmark_corpus" / "omnidocbench",
    )
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name for token counts")
    parser.add_argument("--manifest-name", type=str, default="text_manifest.jsonl", help="Output manifest filename")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)

    all_records: List[TextRecord] = []

    if args.fox_root:
        fox_records = prepare_fox(args.fox_root, args.fox_output, tokenizer)
        write_manifest(args.fox_output / "metadata" / args.manifest_name, fox_records)
        all_records.extend(fox_records)

    if args.omnidoc_root:
        omni_records = prepare_omnidocbench(args.omnidoc_root, args.omnidoc_output, tokenizer)
        write_manifest(args.omnidoc_output / "metadata" / args.manifest_name, omni_records)
        all_records.extend(omni_records)

    if not all_records:
        print("No datasets processed; check paths.", file=sys.stderr)
        sys.exit(1)

    combined_manifest = REPO_ROOT / "data" / "benchmark_corpus" / "metadata"
    combined_manifest.mkdir(parents=True, exist_ok=True)
    write_manifest(combined_manifest / args.manifest_name, all_records)


if __name__ == "__main__":
    main()
