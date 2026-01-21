#!/usr/bin/env python3
"""Bulk-populate structural manifold prototype caches for text-mode evals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Set
import sys

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
TOP_ROOT = REPO_ROOT.parent
for candidate in (MODULE_DIR, REPO_ROOT, TOP_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from prototype_cache import load_prototype_cache, save_prototype_cache
from manifold_compression_eval import iter_text_documents
from src.manifold.codec import encode_text


def load_signature_set(vocab_path: Path | None, signature_files: List[Path]) -> Set[str]:
    signatures: Set[str] = set()
    if vocab_path:
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
        source = data["signatures"] if isinstance(data, dict) else data
        signatures.update(str(token) for token in source if token)
    for path in signature_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "signatures" in payload:
            payload = payload["signatures"]
        signatures.update(str(token) for token in payload if token)
    return signatures


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate prototype_cache.json for text-mode evaluations.")
    parser.add_argument("--cache", type=Path, required=True, help="Path to prototype_cache.json to create/update.")
    parser.add_argument("--text-root", type=Path, required=True, help="Directory or JSONL corpus used for prototype recovery.")
    parser.add_argument("--vocab", type=Path, help="JSON file with {'signatures': [...]} list.")
    parser.add_argument(
        "--signature-file",
        action="append",
        type=Path,
        default=[],
        help="Additional JSON files (list or {'signatures': [...]}) to seed the target set.",
    )
    parser.add_argument("--max-signatures", type=int, help="Optional limit on unique signatures to populate.")
    parser.add_argument("--window-bytes", type=int, default=512)
    parser.add_argument("--stride-bytes", type=int, default=384)
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--max-documents", type=int, help="Stop scanning after this many documents.")
    parser.add_argument("--log-every", type=int, default=500, help="Progress log interval.")
    args = parser.parse_args()

    target_signatures = load_signature_set(args.vocab, args.signature_file)
    if args.max_signatures:
        target_signatures = set(list(target_signatures)[: args.max_signatures])
    if not target_signatures:
        raise ValueError("No signatures provided via --vocab/--signature-file.")

    cache = load_prototype_cache(args.cache) if args.cache.exists() else {}
    missing = {sig for sig in target_signatures if sig not in cache}
    if not missing:
        print("Prototype cache already covers all target signatures.")
        return

    recovered = 0
    for doc_index, (_, text) in enumerate(iter_text_documents(args.text_root)):
        if args.max_documents is not None and doc_index >= args.max_documents:
            break
        encoding = encode_text(
            text,
            window_bytes=args.window_bytes,
            stride_bytes=args.stride_bytes,
            precision=args.precision,
            deduplicate=False,
        )
        for span in encoding.spans:
            if not span.signature or span.payload is None:
                continue
            if span.signature in missing:
                cache[span.signature] = span.payload
                missing.remove(span.signature)
                recovered += 1
                if recovered % args.log_every == 0:
                    print(f"[+] recovered {recovered} prototypes · remaining={len(missing)}")
        if not missing:
            break

    save_prototype_cache(args.cache, cache)
    print(
        json.dumps(
            {
                "total_requested": len(target_signatures),
                "already_cached": len(target_signatures) - len(missing) - recovered,
                "new_recovered": recovered,
                "missing_after_scan": len(missing),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
