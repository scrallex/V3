#!/usr/bin/env python3
"""Evaluate manifold compression, token budgets, and reconstruction fidelity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def _maybe_extend_sys_path() -> None:
    candidates = [
        REPO_ROOT / "score" / "src",
        REPO_ROOT.parent / "score" / "src",
    ]
    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_maybe_extend_sys_path()

from sep_text_manifold import encode, native  # type: ignore  # noqa: E402


@dataclass
class WindowRecord:
    doc_id: str
    offset: int
    signature: str
    metrics: Dict[str, float]
    chunk: bytes


def _extract_text(record: object, text_key: str) -> str:
    if isinstance(record, dict):
        value = record.get(text_key)
    else:
        value = None
    if value is None:
        raise KeyError(f"Missing text field '{text_key}' in record: {record}")
    if not isinstance(value, str):
        value = str(value)
    return value


def _iter_text_directory(root: Path, text_key: str) -> Iterable[Tuple[str, str]]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".jsonl", ".ndjson", ".json"}:
            continue
        relative = path.relative_to(root).as_posix()
        base = relative[:-len(path.suffix)] if path.suffix else relative
        base_id = base.replace("/", "__")
        if suffix == ".txt":
            yield base_id, path.read_text(encoding="utf-8")
        elif suffix in {".jsonl", ".ndjson"}:
            yield from _iter_jsonl_file(path, text_key, doc_prefix=base_id)
        elif suffix == ".json":
            yield from _iter_json_file(path, text_key, doc_prefix=base_id)


def _iter_jsonl_file(path: Path, text_key: str, doc_prefix: str | None = None) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = _extract_text(record, text_key)
            prefix = doc_prefix or path.stem
            doc_id = f"{prefix}__{idx:07d}"
            yield doc_id, text


def _iter_json_file(path: Path, text_key: str, doc_prefix: str | None = None) -> Iterable[Tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prefix = doc_prefix or path.stem
    if isinstance(data, list):
        for idx, record in enumerate(data):
            text = _extract_text(record, text_key)
            yield f"{prefix}__{idx:07d}", text
        return
    if isinstance(data, dict):
        text = _extract_text(data, text_key)
        yield prefix, text
        return
    raise TypeError(f"Unsupported JSON structure in {path}")


def iter_text_documents(root: Path, json_text_key: str = "text") -> Iterable[Tuple[str, str]]:
    if root.is_dir():
        yield from _iter_text_directory(root, json_text_key)
        return
    suffix = root.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        yield from _iter_jsonl_file(root, json_text_key)
        return
    if suffix == ".json":
        yield from _iter_json_file(root, json_text_key)
        return
    # Fallback: treat as plain text file
    yield root.stem, root.read_text(encoding="utf-8")


def sliding_windows(data: bytes, window_bytes: int, stride_bytes: int) -> Iterable[Tuple[int, bytes]]:
    if not data:
        return
    if len(data) <= window_bytes:
        yield 0, data
        return
    for offset in range(0, len(data) - window_bytes + 1, stride_bytes):
        yield offset, data[offset : offset + window_bytes]
    tail_start = len(data) - window_bytes
    if tail_start % stride_bytes != 0:
        yield tail_start, data[tail_start:]


def bits_per_metric(precision: int) -> int:
    buckets = (10**precision) + 1
    return math.ceil(math.log2(buckets))


def signature_storage_bytes(precision: int) -> int:
    metric_bits = bits_per_metric(precision)
    metrics_total_bits = metric_bits * 4  # coherence, stability, entropy, hazard λ
    metrics_bytes = math.ceil(metrics_total_bits / 8)
    count_bytes = 4  # repetition count
    return metrics_bytes + count_bytes


def build_compressed_representation(
    text_root: Path,
    window_bytes: int,
    stride_bytes: int,
    precision: int,
    max_documents: Optional[int] = None,
    json_text_key: str = "text",
    document_offset: int = 0,
) -> Tuple[
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, List[WindowRecord]],
    Dict[str, str],
    Dict[str, int],
    Dict[str, Dict[str, bytes]],
]:
    compressed: Dict[str, Dict[str, Dict[str, float]]] = {}
    doc_windows: Dict[str, List[WindowRecord]] = defaultdict(list)
    doc_texts: Dict[str, str] = {}
    doc_sizes: Dict[str, int] = {}
    prototypes: Dict[str, Dict[str, bytes]] = defaultdict(dict)

    processed_docs = 0
    start_index = max(document_offset, 0)
    for doc_index, (doc_id, text) in enumerate(iter_text_documents(text_root, json_text_key=json_text_key)):
        if doc_index < start_index:
            continue
        if max_documents is not None and processed_docs >= max_documents:
            break
        processed_docs += 1
        text_bytes = text.encode("utf-8")
        doc_texts[doc_id] = text
        doc_sizes[doc_id] = len(text_bytes)
        doc_bucket = compressed.setdefault(doc_id, {})

        for offset, chunk in sliding_windows(text_bytes, window_bytes, stride_bytes):
            metrics = encode.encode_window(bytes(chunk))
            signature = encode.signature_from_metrics(
                metrics["coherence"],
                metrics["stability"],
                metrics["entropy"],
                precision=precision,
            )
            entry = doc_bucket.setdefault(signature, {"count": 0.0, "hazard_sum": 0.0})
            entry["count"] += 1.0
            entry["hazard_sum"] += float(metrics["lambda_hazard"])
            chunk_bytes = bytes(chunk)
            if signature not in prototypes[doc_id]:
                prototypes[doc_id][signature] = chunk_bytes
            doc_windows[doc_id].append(
                WindowRecord(
                    doc_id=doc_id,
                    offset=offset,
                    signature=signature,
                    metrics=metrics,
                    chunk=chunk_bytes,
                )
            )

    return compressed, doc_windows, doc_texts, doc_sizes, prototypes


def normalise_compressed(compressed: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    normalised: Dict[str, Dict[str, Dict[str, float]]] = {}
    for doc_id, signatures in compressed.items():
        doc_bucket: Dict[str, Dict[str, float]] = {}
        for signature, payload in signatures.items():
            count = payload["count"]
            hazard_sum = payload["hazard_sum"]
            hazard_avg = hazard_sum / count if count else 0.0
            doc_bucket[signature] = {"count": count, "hazard": hazard_avg}
        normalised[doc_id] = doc_bucket
    return normalised


def evaluate_verification(
    doc_signatures: Dict[str, set],
    doc_windows: Dict[str, List[WindowRecord]],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], int]:
    positives = sum(len(windows) for windows in doc_windows.values())
    true_positive = positives

    negatives = 0
    false_positive = 0
    per_doc_stats: Dict[str, Dict[str, float]] = {}

    for doc_id, signatures in doc_signatures.items():
        doc_pos = len(doc_windows.get(doc_id, []))
        doc_true_pos = doc_pos
        doc_neg = 0
        doc_false_pos = 0
        for other_id, other_windows in doc_windows.items():
            if other_id == doc_id:
                continue
            for record in other_windows:
                doc_neg += 1
                negatives += 1
                if record.signature in signatures:
                    doc_false_pos += 1
                    false_positive += 1
        precision = doc_true_pos / (doc_true_pos + doc_false_pos) if (doc_true_pos + doc_false_pos) else 1.0
        fpr = doc_false_pos / doc_neg if doc_neg else 0.0
        recall = 1.0 if doc_pos else 0.0
        per_doc_stats[doc_id] = {
            "positives": doc_pos,
            "true_positive": doc_true_pos,
            "negatives": doc_neg,
            "false_positive": doc_false_pos,
            "precision": precision,
            "false_positive_rate": fpr,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        }

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
    recall = true_positive / positives if positives else 0.0
    false_positive_rate = false_positive / negatives if negatives else 0.0

    overall = {
        "positives": positives,
        "true_positive": true_positive,
        "negatives": negatives,
        "false_positive": false_positive,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
    }
    return overall, per_doc_stats, negatives


def reconstruct_document(
    windows: Sequence[WindowRecord],
    prototypes: Dict[str, bytes],
    stride_bytes: int,
) -> bytes:
    if not windows:
        return b""
    sorted_windows = sorted(windows, key=lambda rec: rec.offset)
    result = bytearray()
    for record in sorted_windows:
        chunk = prototypes.get(record.signature, record.chunk)
        start = max(record.offset, 0)
        if len(result) < start:
            gap = start - len(result)
            result.extend(chunk[:gap])
        overlap = len(result) - start
        if overlap < 0:
            overlap = 0
        if overlap >= len(chunk):
            continue
        result.extend(chunk[overlap:])
    return bytes(result)


@lru_cache(maxsize=4)
def get_tokenizer(name: str, trust_remote_code: bool):
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = 1_000_000
    return tokenizer


class TokenizerRuntime:
    def __init__(self, name: str, trust_remote_code: bool):
        self.name = name
        self.trust_remote_code = trust_remote_code
        self._tokenizer = get_tokenizer(name, trust_remote_code)
        self._cache: Dict[str, Tuple[int, ...]] = {}

    def encode(self, text: str) -> Tuple[int, ...]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        tokens = tuple(self._tokenizer.encode(text, add_special_tokens=False))
        self._cache[text] = tokens
        return tokens

    def count(self, text: str) -> int:
        return len(self.encode(text))


def levenshtein_distance(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        curr = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            curr.append(
                min(
                    curr[-1] + 1,  # insertion
                    prev[j] + 1,    # deletion
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def compute_token_metrics(original: str, reconstructed: str, tokenizer: TokenizerRuntime) -> Dict[str, float]:
    original_tokens = tokenizer.encode(original)
    reconstructed_tokens = tokenizer.encode(reconstructed)
    distance = levenshtein_distance(original_tokens, reconstructed_tokens)
    original_len = len(original_tokens)
    reconstructed_len = len(reconstructed_tokens)
    denom = max(original_len, reconstructed_len, 1)
    accuracy = 1.0 - (distance / denom)
    precision = 1.0 - (distance / max(reconstructed_len, 1))
    recall = 1.0 - (distance / max(original_len, 1))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "original_tokens": original_len,
        "reconstructed_tokens": reconstructed_len,
        "edit_distance": distance,
        "token_accuracy": accuracy,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def compute_character_metrics(original: str, reconstructed: str) -> Dict[str, float]:
    original_chars = list(original)
    reconstructed_chars = list(reconstructed)
    distance = levenshtein_distance(original_chars, reconstructed_chars)
    original_len = len(original_chars)
    reconstructed_len = len(reconstructed_chars)
    denom = max(original_len, reconstructed_len, 1)
    accuracy = 1.0 - (distance / denom)
    precision = 1.0 - (distance / max(reconstructed_len, 1))
    recall = 1.0 - (distance / max(original_len, 1))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    normalized_distance = distance / denom
    return {
        "original_characters": original_len,
        "reconstructed_characters": reconstructed_len,
        "edit_distance": distance,
        "character_accuracy": accuracy,
        "character_precision": precision,
        "character_recall": recall,
        "character_f1": f1,
        "normalized_edit_distance": normalized_distance,
    }


def evaluate_manifold(
    text_root: Path,
    window_bytes: int,
    stride_bytes: int,
    precision: int,
    tokenizer_name: str = "gpt2",
    tokenizer_trust_remote_code: bool = False,
    max_documents: Optional[int] = None,
    use_native: bool = False,
    json_text_key: str = "text",
    document_offset: int = 0,
) -> Dict[str, object]:
    if use_native:
        native.set_use_native(True)

    (
        compressed_raw,
        doc_windows,
        doc_texts,
        doc_sizes,
        prototypes,
    ) = build_compressed_representation(
        text_root,
        window_bytes,
        stride_bytes,
        precision,
        max_documents=max_documents,
        json_text_key=json_text_key,
        document_offset=document_offset,
    )
    compressed = normalise_compressed(compressed_raw)
    doc_signatures = {doc_id: set(bucket.keys()) for doc_id, bucket in compressed.items()}
    tokenizer_runtime = TokenizerRuntime(tokenizer_name, tokenizer_trust_remote_code)

    storage_bytes_per_sig = signature_storage_bytes(precision)
    doc_compressed_size = {doc_id: len(signatures) * storage_bytes_per_sig for doc_id, signatures in doc_signatures.items()}
    compressed_size = sum(doc_compressed_size.values())
    original_size = sum(doc_sizes.values())
    compression_ratio = (original_size / compressed_size) if compressed_size else float("inf")

    signature_doc_counts = Counter()
    for doc_id, signatures in doc_signatures.items():
        for signature in signatures:
            signature_doc_counts[signature] += 1
    shared_signatures = sum(1 for count in signature_doc_counts.values() if count > 1)

    verification_metrics, per_doc_verification, negatives = evaluate_verification(doc_signatures, doc_windows)

    per_doc_summary = {}
    total_text_tokens = 0
    total_reconstructed_tokens = 0
    total_token_edit_distance = 0
    total_stream_tokens = 0
    total_unique_tokens = 0
    total_characters = 0
    total_reconstructed_characters = 0
    total_char_edit_distance = 0

    for doc_id, windows in doc_windows.items():
        original_text = doc_texts[doc_id]
        reconstructed_bytes = reconstruct_document(windows, prototypes[doc_id], stride_bytes)
        reconstructed_text = reconstructed_bytes.decode("utf-8", errors="replace")
        token_metrics = compute_token_metrics(original_text, reconstructed_text, tokenizer_runtime)
        character_metrics = compute_character_metrics(original_text, reconstructed_text)

        stream_tokens = len(windows)
        unique_tokens = len(doc_signatures.get(doc_id, set()))

        total_text_tokens += token_metrics["original_tokens"]
        total_reconstructed_tokens += token_metrics["reconstructed_tokens"]
        total_token_edit_distance += token_metrics["edit_distance"]
        total_stream_tokens += stream_tokens
        total_unique_tokens += unique_tokens
        total_characters += character_metrics["original_characters"]
        total_reconstructed_characters += character_metrics["reconstructed_characters"]
        total_char_edit_distance += character_metrics["edit_distance"]

        per_doc_summary[doc_id] = {
            "original_size_bytes": doc_sizes[doc_id],
            "compressed_size_bytes": doc_compressed_size.get(doc_id, 0),
            "compression_ratio": (
                doc_sizes[doc_id] / doc_compressed_size.get(doc_id, 1)
                if doc_compressed_size.get(doc_id, 0)
                else float("inf")
            ),
            "unique_signatures": unique_tokens,
            "stream_windows": stream_tokens,
            "token_metrics": token_metrics,
            "character_metrics": character_metrics,
            "normalized_edit_distance": character_metrics["normalized_edit_distance"],
            "token_compression_unique": (
                token_metrics["original_tokens"] / unique_tokens if unique_tokens else float("inf")
            ),
            "token_compression_stream": (
                token_metrics["original_tokens"] / stream_tokens if stream_tokens else float("inf")
            ),
            "verification": per_doc_verification.get(doc_id, {}),
        }

    token_compression_unique = (
        total_text_tokens / total_unique_tokens if total_unique_tokens else float("inf")
    )
    token_compression_stream = (
        total_text_tokens / total_stream_tokens if total_stream_tokens else float("inf")
    )
    token_accuracy = 1.0 - (
        total_token_edit_distance / max(total_text_tokens, total_reconstructed_tokens, 1)
    )
    token_precision = 1.0 - (total_token_edit_distance / max(total_reconstructed_tokens, 1))
    token_recall = 1.0 - (total_token_edit_distance / max(total_text_tokens, 1))
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

    try:
        text_root_rel = str(text_root.relative_to(REPO_ROOT))
    except ValueError:
        text_root_rel = str(text_root)

    summary: Dict[str, object] = {
        "text_root": text_root_rel,
        "json_text_key": json_text_key,
        "documents": len(doc_windows),
        "window_bytes": window_bytes,
        "stride_bytes": stride_bytes,
        "precision": precision,
        "tokenizer_name": tokenizer_name,
        "tokenizer_trust_remote_code": tokenizer_trust_remote_code,
        "signature_storage_bytes": storage_bytes_per_sig,
        "original_size_bytes": original_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": compression_ratio,
        "unique_signatures": sum(len(signatures) for signatures in doc_signatures.values()),
        "shared_signatures": shared_signatures,
        "verification": verification_metrics,
        "per_document": per_doc_summary,
        "records": sum(len(windows) for windows in doc_windows.values()),
        "negatives_evaluated": negatives,
        "token_metrics": {
            "text_tokens_total": total_text_tokens,
            "reconstructed_tokens_total": total_reconstructed_tokens,
            "token_edit_distance_total": total_token_edit_distance,
            "manifold_tokens_stream": total_stream_tokens,
            "manifold_tokens_unique": total_unique_tokens,
            "token_compression_stream": token_compression_stream,
            "token_compression_unique": token_compression_unique,
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
        "use_native": use_native,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate manifold compression fidelity")
    parser.add_argument("--text-root", type=Path, required=True, help="Root directory of UTF-8 text files")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON summary")
    parser.add_argument("--window-bytes", type=int, default=256, help="Sliding window size")
    parser.add_argument("--stride-bytes", type=int, default=192, help="Sliding window stride")
    parser.add_argument("--precision", type=int, default=2, help="Signature precision (decimal places)")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name or local path")
    parser.add_argument(
        "--tokenizer-trust-remote-code",
        action="store_true",
        help="Allow remote code when loading the tokenizer (required for some custom tokenizers)",
    )
    parser.add_argument(
        "--json-text-key",
        type=str,
        default="text",
        help="Field name to read when ingesting JSON/JSONL corpora",
    )
    parser.add_argument("--max-documents", type=int, help="Optional cap on number of documents to process")
    parser.add_argument("--document-offset", type=int, default=0, help="Skip the first N documents before processing")
    parser.add_argument("--use-native", action="store_true", help="Prefer the native manifold kernel if available")
    args = parser.parse_args()

    text_root = args.text_root.resolve()
    if not text_root.exists():
        raise FileNotFoundError(f"text root not found: {text_root}")

    summary = evaluate_manifold(
        text_root=text_root,
        window_bytes=args.window_bytes,
        stride_bytes=args.stride_bytes,
        precision=args.precision,
        tokenizer_name=args.tokenizer,
        tokenizer_trust_remote_code=args.tokenizer_trust_remote_code,
        max_documents=args.max_documents,
        use_native=args.use_native,
        json_text_key=args.json_text_key,
        document_offset=args.document_offset,
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
