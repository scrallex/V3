"""Helpers for deriving bundle hits from gate records."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml

from scripts.research import roc_utils

try:  # Semantic tagging is optional in some deployments
    from scripts.research.semantic_tagger import generate_semantic_tags as _generate_semantic_tags
except Exception:  # pragma: no cover - optional dependency
    _generate_semantic_tags = None


@dataclass
class BundleRule:
    bundle_id: str
    label: str
    action: str
    hold_minutes: int
    regime: Optional[str] = None
    hazard_deciles: Optional[Sequence[int]] = None
    hazard_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    min_repetitions: int = 0
    min_semantic_hits: Sequence[str] = field(default_factory=list)
    coherence_slope: Optional[str] = None
    domain_wall_slope: Optional[str] = None
    min_coherence: Optional[float] = None
    max_coherence: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "id": self.bundle_id,
            "label": self.label,
            "action": self.action,
            "hold_minutes": self.hold_minutes,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class BundleHit:
    rule: BundleRule
    score: float

    def to_dict(self) -> Dict[str, object]:
        payload = self.rule.to_dict()
        payload["score"] = self.score
        return payload


@dataclass
class BundleReadiness:
    rule: BundleRule
    conditions: Dict[str, bool]

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.rule.bundle_id,
            "label": self.rule.label,
            "action": self.rule.action,
            "ready": all(self.conditions.values()) if self.conditions else True,
            "conditions": dict(self.conditions),
        }


@dataclass
class BundleCatalog:
    rules: List[BundleRule]

    @classmethod
    def load(cls, path: Path) -> "BundleCatalog":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        bundles = []
        for entry in data.get("bundles", []) or []:
            bundle_id = str(entry.get("id") or "").strip().upper()
            if not bundle_id:
                continue
            action = str(entry.get("action") or "").strip().lower() or "promote"
            hold_minutes = int(entry.get("hold_minutes", 60) or 60)
            hazard_deciles = entry.get("hazard_deciles")
            hazard_range = entry.get("hazard_range")
            rule = BundleRule(
                bundle_id=bundle_id,
                label=str(entry.get("label") or bundle_id),
                action=action,
                hold_minutes=hold_minutes,
                regime=(str(entry.get("regime")) if entry.get("regime") else None),
                hazard_deciles=list(hazard_deciles) if isinstance(hazard_deciles, Sequence) else None,
                hazard_range=(
                    (float(hazard_range[0]) if hazard_range and hazard_range[0] is not None else None),
                    (float(hazard_range[1]) if hazard_range and hazard_range[1] is not None else None),
                )
                if hazard_range
                else None,
                min_repetitions=int(entry.get("min_repetitions", 0) or 0),
                min_semantic_hits=list(entry.get("semantic_tags") or []),
                coherence_slope=_clean_slope(entry.get("coherence_slope")),
                domain_wall_slope=_clean_slope(entry.get("domain_wall_slope")),
                min_coherence=_maybe_float(entry, "min_coherence"),
                max_coherence=_maybe_float(entry, "max_coherence"),
                metadata=dict(entry.get("metadata") or {}),
            )
            bundles.append(rule)
        bundles.sort(key=lambda item: item.bundle_id)
        return cls(rules=bundles)

    def evaluate_record(self, record: Mapping[str, object]) -> Tuple[List[BundleHit], List[str], Dict[str, BundleReadiness]]:
        hazard = _maybe_float(record, "hazard")
        repetition_value = record.get("repetitions")
        try:
            repetitions = int(repetition_value) if repetition_value is not None else None
        except Exception:
            repetitions = None
        semantics = _semantic_tags(record)
        coherence = _structure_value(record, "coherence")
        coherence_slope = roc_utils.slope_bucket(_structure_value(record, "coherence_tau_slope"))
        domain_slope = roc_utils.slope_bucket(_structure_value(record, "domain_wall_slope"))
        regime = _regime_label(record)
        hazard_dec = roc_utils.hazard_decile(hazard)
        hits: List[BundleHit] = []
        blocks: List[str] = []
        readiness: Dict[str, BundleReadiness] = {}
        for rule in self.rules:
            matched, conditions = _rule_matches(
                rule,
                hazard,
                hazard_dec,
                repetitions,
                semantics,
                coherence_slope,
                domain_slope,
                regime,
                coherence,
            )
            readiness[rule.bundle_id] = BundleReadiness(rule=rule, conditions=conditions)
            if not matched:
                continue
            hit = BundleHit(rule=rule, score=_score_record(record))
            hits.append(hit)
            if rule.action == "quarantine":
                blocks.append(rule.bundle_id)
        return hits, blocks, readiness


_BUNDLE_CATALOG_CACHE: Dict[Path, BundleCatalog] = {}


def _score_record(record: Mapping[str, object]) -> float:
    hazard = _maybe_float(record, "hazard") or 0.0
    coherence = _structure_value(record, "coherence") or 0.0
    stability = _structure_value(record, "stability") or 0.0
    return float(hazard) * 0.5 + float(coherence) * 0.3 + float(stability) * 0.2


def _clean_slope(raw: object) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"pos", "positive"}:
        return "pos"
    if value in {"neg", "negative"}:
        return "neg"
    if value in {"flat", "neutral"}:
        return "flat"
    return None


def _maybe_float(mapping: Mapping[str, object], key: str) -> Optional[float]:
    value = mapping.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _structure_value(record: Mapping[str, object], key: str) -> Optional[float]:
    structure = record.get("structure") if isinstance(record, Mapping) else None
    if isinstance(structure, Mapping) and key in structure:
        try:
            return float(structure[key])
        except (TypeError, ValueError):
            return None
    components = record.get("components") if isinstance(record, Mapping) else None
    if isinstance(components, Mapping) and key in components:
        try:
            return float(components[key])
        except (TypeError, ValueError):
            return None
    raw_value = record.get(key)
    if raw_value is not None:
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return None
    return None


def _semantic_tags(record: Mapping[str, object]) -> List[str]:
    tags = record.get("semantic_tags")
    normalised: List[str] = []
    if isinstance(tags, Sequence):
        normalised = [str(tag).strip().lower() for tag in tags if isinstance(tag, str)]
    seen = set(normalised)
    if _generate_semantic_tags:
        try:
            derived = _generate_semantic_tags(record)
        except TypeError:
            derived = _generate_semantic_tags(record, overrides=None)  # type: ignore[misc]
        except Exception:
            derived = []
        for tag in derived or []:
            if not isinstance(tag, str):
                continue
            cleaned = tag.strip().lower()
            if cleaned and cleaned not in seen:
                normalised.append(cleaned)
                seen.add(cleaned)
    stability = _structure_value(record, "stability")
    if stability is not None and stability >= 0.52 and "highly_stable" not in seen:
        normalised.append("highly_stable")
        seen.add("highly_stable")
    coherence_slope = _structure_value(record, "coherence_tau_slope")
    if coherence_slope is not None and coherence_slope > 0.005 and "strengthening_structure" not in seen:
        normalised.append("strengthening_structure")
    return normalised


def apply_semantic_tags(gate_payload: Mapping[str, object]) -> Dict[str, object]:
    """Return a copy of the payload with derived semantic tags applied."""

    payload = dict(gate_payload)
    tags = _semantic_tags(payload)
    if tags:
        payload["semantic_tags"] = tags
    else:
        payload.pop("semantic_tags", None)
    return payload


def get_bundle_hits(
    gate_payload: Mapping[str, object],
    *,
    catalog: Optional[BundleCatalog] = None,
    bundle_config: str | os.PathLike[str] | None = None,
) -> Tuple[List[Dict[str, object]], List[str], Dict[str, Dict[str, object]]]:
    """Return bundle hits, blocks, and readiness diagnostics for a gate payload."""

    active_catalog = catalog
    if active_catalog is None:
        config_path = Path(bundle_config or os.getenv("BUNDLE_STRATEGY_FILE", "config/bundle_strategy.yaml"))
        active_catalog = _catalog_from_path(config_path)
        if active_catalog is None:
            return [], [], {}
    try:
        hits, blocks, readiness = active_catalog.evaluate_record(gate_payload)
    except Exception:
        return [], [], {}
    readiness_payload = {key: ready.to_dict() for key, ready in readiness.items()}
    return [hit.to_dict() for hit in hits], list(blocks), readiness_payload


def _catalog_from_path(path: Path) -> Optional[BundleCatalog]:
    candidate = path.expanduser()
    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate
    cached = _BUNDLE_CATALOG_CACHE.get(resolved)
    if cached:
        return cached
    if not resolved.exists():
        return None
    try:
        catalog = BundleCatalog.load(resolved)
    except Exception:
        return None
    _BUNDLE_CATALOG_CACHE[resolved] = catalog
    return catalog


def _regime_label(record: Mapping[str, object]) -> Optional[str]:
    regime_payload = record.get("regime")
    if isinstance(regime_payload, Mapping):
        label = regime_payload.get("label")
        if isinstance(label, str):
            return label.strip().lower()
    if isinstance(regime_payload, str):
        return regime_payload.strip().lower()
    return None


def _rule_matches(
    rule: BundleRule,
    hazard: Optional[float],
    hazard_decile: int,
    repetitions: Optional[int],
    semantics: Sequence[str],
    coherence_slope: str,
    domain_slope: str,
    regime: Optional[str],
    coherence: Optional[float],
) -> Tuple[bool, Dict[str, bool]]:
    conditions: Dict[str, bool] = {}
    if rule.regime:
        conditions["regime"] = bool(regime == rule.regime.lower())
    if rule.hazard_deciles is not None:
        allowed = {int(v) for v in rule.hazard_deciles}
        conditions["hazard_decile"] = bool(hazard_decile in allowed)
    if rule.hazard_range:
        lo, hi = rule.hazard_range
        ok = True
        if lo is not None and (hazard is None or hazard < lo):
            ok = False
        if hi is not None and (hazard is None or hazard > hi):
            ok = False
        conditions["hazard_window"] = ok
    if rule.min_repetitions > 0:
        ok = repetitions is not None and repetitions >= rule.min_repetitions
        conditions["repetitions"] = ok
    if rule.min_semantic_hits:
        observed = {tag.lower() for tag in semantics}
        required = {tag.lower() for tag in rule.min_semantic_hits}
        conditions["semantic_tags"] = bool(required.intersection(observed))
    if rule.coherence_slope:
        conditions["coherence_slope"] = coherence_slope == rule.coherence_slope
    if rule.domain_wall_slope:
        conditions["domain_wall_slope"] = domain_slope == rule.domain_wall_slope
    if rule.min_coherence is not None:
        cond = coherence is not None and coherence >= rule.min_coherence
        conditions["coherence_min"] = cond
    if rule.max_coherence is not None:
        cond = coherence is not None and coherence <= rule.max_coherence
        conditions["coherence_max"] = cond
    matched = all(conditions.values()) if conditions else True
    return matched, conditions


def iter_gate_records(gates_path: Path) -> Iterable[Dict[str, object]]:
    directory = gates_path
    if directory.is_file():
        files = [directory]
    else:
        files = list(roc_utils.iter_gate_files(directory))
    for path in files:
        for record in roc_utils.load_gate_records(path):
            yield record


def write_activation_tape(records: Iterable[Mapping[str, object]], catalog: BundleCatalog, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            hits, blocks, readiness = catalog.evaluate_record(record)
            payload = {
                "instrument": record.get("instrument"),
                "ts_ms": record.get("ts_ms"),
                "hazard": record.get("hazard"),
                "repetitions": record.get("repetitions"),
                "regime": record.get("regime"),
                "coherence": _structure_value(record, "coherence"),
                "bundle_hits": [hit.to_dict() for hit in hits],
                "bundle_blocks": blocks,
                "roc_forward_pct": record.get("roc_forward_pct"),
                "semantic_tags": record.get("semantic_tags"),
                "bundle_readiness": {key: ready.to_dict() for key, ready in readiness.items()},
            }
            handle.write(json.dumps(payload))
            handle.write("\n")


def nearest_horizon(hold_minutes: int, available: Sequence[int]) -> Optional[int]:
    if not available:
        return None
    return min(available, key=lambda horizon: abs(int(horizon) - int(hold_minutes)))


def iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
