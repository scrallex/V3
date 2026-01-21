from __future__ import annotations

from scripts.trading.portfolio_manager import StrategyInstrument, gate_evaluation, structural_metric


def _profile_with_guards() -> StrategyInstrument:
    return StrategyInstrument(
        symbol="EUR_USD",
        hazard_max=None,
        min_repetitions=1,
        guards={
            "min_coherence": 0.2,
            "min_stability": 0.2,
            "max_entropy": 1.0,
            "max_coherence_tau_slope": -0.01,
            "max_domain_wall_slope": -0.01,
            "min_low_freq_share": 0.4,
            "max_reynolds_ratio": 1.0,
            "min_temporal_half_life": 1.5,
            "min_spatial_corr_length": 0.75,
            "min_pinned_alignment": 0.9,
        },
        session=None,
        semantic_filter=[],
    )


def test_gate_evaluation_accepts_when_structural_metrics_within_bounds():
    profile = _profile_with_guards()
    payload = {
        "admit": 1,
        "lambda": 0.05,
        "repetitions": 2,
        "components": {
            "coherence": 0.6,
            "stability": 0.55,
            "entropy": 0.5,
        },
        "structure": {
            "coherence_tau_slope": -0.02,
            "domain_wall_slope": -0.025,
            "spectral_lowf_share": 0.7,
            "reynolds_ratio": 0.85,
            "temporal_half_life": 1.8,
            "spatial_corr_length": 1.2,
            "pinned_alignment": 0.95,
        },
    }

    admitted, reasons = gate_evaluation(payload, profile)

    assert admitted is True
    assert reasons == []


def test_gate_evaluation_blocks_when_structural_metrics_violate_guards():
    profile = _profile_with_guards()
    payload = {
        "admit": 1,
        "lambda": 0.01,
        "repetitions": 3,
        "components": {
            "coherence": 0.7,
            "stability": 0.65,
            "entropy": 0.2,
        },
        "structure": {
            "coherence_tau_slope": -0.002,
            "domain_wall_slope": -0.003,
            "spectral_lowf_share": 0.25,
            "reynolds_ratio": 1.2,
            "temporal_half_life": 1.2,
            "spatial_corr_length": 0.7,
            "pinned_alignment": 0.82,
        },
    }

    admitted, reasons = gate_evaluation(payload, profile)

    assert admitted is False
    assert "coherence_tau_slope_above_max" in reasons
    assert "domain_wall_slope_above_max" in reasons
    assert "spectral_lowf_share_below_min" in reasons
    assert "reynolds_above_max" in reasons
    assert "temporal_half_life_below_min" in reasons
    assert "spatial_corr_length_below_min" in reasons
    assert "pinned_alignment_below_min" in reasons


def test_gate_evaluation_handles_missing_new_metrics():
    profile = _profile_with_guards()
    payload = {
        "admit": 1,
        "lambda": 0.05,
        "repetitions": 2,
        "components": {
            "coherence": 0.6,
            "stability": 0.55,
            "entropy": 0.5,
        },
        "structure": {
            "coherence_tau_slope": -0.02,
            "domain_wall_slope": -0.025,
            "spectral_lowf_share": 0.7,
        },
        "metrics": {
            "reynolds_ratio": 0.9,
            "temporal_half_life": 1.6,
        },
    }

    admitted, reasons = gate_evaluation(payload, profile)

    assert admitted is False
    assert "spatial_corr_length_invalid" in reasons
    assert "pinned_alignment_invalid" in reasons


def test_structural_metric_fallbacks():
    payload = {
        "components": {"coherence": 0.6},
        "structure": {"coherence_tau_slope": -0.02},
        "metrics": {"domain_wall_slope": -0.01},
    }

    assert structural_metric(payload, "coherence_tau_slope") == -0.02
    assert structural_metric(payload, "domain_wall_slope") == -0.01
    assert structural_metric(payload, "spectral_lowf_share") is None
