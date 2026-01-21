"""Shared helpers for ROC research scripts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

WEEK_LABEL_RE = re.compile(r"(?P<start>\d{4}-\d{2}-\d{2})_to_(?P<end>\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class WeekWindow:
    label: str
    start: datetime
    end: datetime


def parse_week_label(path: Path) -> WeekWindow:
    """Extract the ISO week bounds from a ROC artefact filename."""

    match = WEEK_LABEL_RE.search(path.name)
    if not match:
        raise ValueError(f"Unable to parse week bounds from {path}")
    start = datetime.fromisoformat(match.group("start"))
    end = datetime.fromisoformat(match.group("end"))
    label = f"{match.group('start')}_to_{match.group('end')}"
    return WeekWindow(label=label, start=start, end=end)


def hazard_decile(value: float | None) -> int:
    if value is None:
        return -1
    return max(0, min(9, int(float(value) * 10.0)))


def repetition_bucket(value: int | float | None) -> str:
    if value is None:
        return "na"
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return "na"
    return "5p" if numeric >= 5 else str(max(0, numeric))


def strand_id(regime: str, hazard_dec: int, repetition: str) -> str:
    return f"{regime.lower()}_d{hazard_dec}_r{repetition}"


def iter_gate_files(directory: Path) -> Iterator[Path]:
    """Yield gate JSONL files in chronological order."""

    for path in sorted(directory.glob("gates_with_roc_*.jsonl")):
        yield path


def iter_summary_files(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.glob("roc_summary_*.json")):
        yield path


def load_gate_records(path: Path) -> List[Dict[str, object]]:
    """Load a ROC JSONL export into memory."""

    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            hazard_value = float(record.get("hazard") or 0.0)
            decile = hazard_decile(hazard_value)
            repetition = repetition_bucket(record.get("repetitions"))
            regime = str(record.get("regime") or "unknown")
            record["hazard_decile"] = decile
            record["repetition_bucket"] = repetition
            record["strand_id"] = strand_id(regime, decile, repetition)
            if "semantic_tags" not in record or not isinstance(record["semantic_tags"], list):
                record["semantic_tags"] = []
            rows.append(record)
    return rows


def load_summary(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_lookup(summary_dir: Path) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for path in iter_summary_files(summary_dir):
        window = parse_week_label(path)
        out[window.label] = load_summary(path)
    return out


def roc_value(record: Mapping[str, object], horizon: int) -> Optional[float]:
    roc_payload = record.get("roc_forward_pct")
    if isinstance(roc_payload, Mapping):
        value = roc_payload.get(str(horizon)) or roc_payload.get(horizon)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def semantic_primary(tags: Sequence[str]) -> str:
    priority = [
        "highly_stable",
        "strengthening_structure",
        "improving_stability",
        "low_hazard_environment",
        "high_rupture_event",
        "chaotic_price_action",
    ]
    tags_lower = [tag.lower() for tag in tags if isinstance(tag, str)]
    for key in priority:
        if key in tags_lower:
            return key
    return tags_lower[0] if tags_lower else "none"


def slope_bucket(value: float | None, *, pos: float = 0.01, neg: float = -0.01) -> str:
    if value is None:
        return "na"
    if value <= neg:
        return "neg"
    if value >= pos:
        return "pos"
    return "flat"
