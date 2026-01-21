from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.semantic_tagger import SemanticThresholds, generate_semantic_tags


def test_high_stability_and_low_hazard_tags():
    payload = {
        "components": {"coherence": 0.85, "stability": 0.82, "entropy": 0.3},
        "hazard": 0.05,
        "coherence_delta": 0.02,
        "lambda_slope": -0.1,
    }
    tags = set(generate_semantic_tags(payload))
    assert "highly_stable" in tags
    assert "low_hazard_environment" in tags
    assert "strengthening_structure" in tags
    assert "improving_stability" in tags


def test_threshold_overrides_take_precedence():
    payload = {
        "components": {"coherence": 0.65, "stability": 0.65, "entropy": 0.2},
        "hazard": 0.2,
    }
    overrides = {"high_coherence": 0.6, "high_stability": 0.6, "low_hazard": 0.3}
    tags = set(generate_semantic_tags(payload, overrides=overrides))
    assert "highly_stable" in tags
    assert "low_hazard_environment" in tags


def test_entropy_fallback_from_coherence_distance():
    payload = {"components": {"coherence": 0.05}, "rupture": 0.5}
    thresholds = SemanticThresholds(high_entropy=0.9, high_rupture=0.4)
    tags = set(generate_semantic_tags(payload, thresholds=thresholds))
    assert "chaotic_price_action" in tags
    assert "high_rupture_event" in tags
