#!/usr/bin/env python3
"""Decode manifold signature sequences back into UTF-8 text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List
import sys

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
TOP_ROOT = REPO_ROOT.parent
for candidate in (MODULE_DIR, REPO_ROOT, TOP_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from prototype_cache import ensure_prototypes
from src.manifold.codec import ManifoldEncoding, ManifoldSpan, decode_encoding


def parse_signature_json(path: Path) -> List[Dict[str, List[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Expected a list of samples in {path}")


def gather_signatures(records: Iterable[Dict[str, List[str]]]) -> List[str]:
    signatures: List[str] = []
    for record in records:
        for key in ("prompt_signatures", "generated_signatures"):
            signatures.extend(record.get(key, []))
    return signatures


def decode_sequence(
    signatures: List[str],
    *,
    prototypes: Dict[str, bytes],
    window_bytes: int,
    stride_bytes: int,
) -> str:
    local_prototypes = dict(prototypes)
    for sig in signatures:
        if sig and sig not in local_prototypes:
            local_prototypes[sig] = sig.encode("utf-8")
    spans = [
        ManifoldSpan(offset=index * stride_bytes, signature=sig)
        for index, sig in enumerate(signatures)
        if sig
    ]
    encoding = ManifoldEncoding(
        spans=spans,
        prototypes={sig: local_prototypes[sig] for sig in signatures if sig in local_prototypes},
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=3,
        original_length=0,
    )
    data = decode_encoding(encoding)
    return data.decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode manifold signature samples back into text.")
    parser.add_argument("--samples", type=Path, required=True, help="JSON file produced by the sampler.")
    parser.add_argument("--text-root", type=Path, required=True, help="Source corpus root used during encoding.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file with decoded text.")
    parser.add_argument("--window-bytes", type=int, default=512)
    parser.add_argument("--stride-bytes", type=int, default=384)
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--cache", type=Path, help="Optional prototype cache path (JSON).")
    parser.add_argument("--max-documents", type=int, help="Limit scans when filling the cache.")
    args = parser.parse_args()

    records = parse_signature_json(args.samples)
    required_signatures = gather_signatures(records)
    prototypes = ensure_prototypes(
        required_signatures,
        cache_path=args.cache,
        text_root=args.text_root,
        window_bytes=args.window_bytes,
        stride_bytes=args.stride_bytes,
        precision=args.precision,
        max_documents=args.max_documents,
    )

    decoded = []
    for record in records:
        prompt_text = decode_sequence(
            record.get("prompt_signatures", []),
            prototypes=prototypes,
            window_bytes=args.window_bytes,
            stride_bytes=args.stride_bytes,
        )
        generated_text = decode_sequence(
            record.get("generated_signatures", []),
            prototypes=prototypes,
            window_bytes=args.window_bytes,
            stride_bytes=args.stride_bytes,
        )
        decoded.append(
            {
                **record,
                "prompt_text": prompt_text,
                "generated_text": generated_text,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decoded, indent=2), encoding="utf-8")
    print(f"Wrote decoded samples to {args.output}")


if __name__ == "__main__":
    main()
