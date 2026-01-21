# Semantic Regime Research Playbook

## Overview

The semantic regime pipeline extends the SEP backtesting stack with tags that
summarise the “rhythm” of every QFH gate. By attaching human-readable labels to
candle-level metrics we can:

- identify when the echo strategy thrives (e.g. highly coherent, low-hazard minutes)
- suppress allocations in regimes that historically bleed capital (e.g. chaotic, rupture-heavy tape)
- communicate findings to the broader quant group without exposing raw manifold tensors

This playbook documents the research workflow introduced in October 2025.

## Semantic Tagging

`scripts/research/semantic_tagger.py` exports `generate_semantic_tags` and the
`SemanticThresholds` dataclass. The default rules translate QFH metrics into the
following tags:

| Tag | Rule |
| --- | --- |
| `highly_stable` | coherence ≥ 0.7 **and** stability ≥ 0.7 |
| `strengthening_structure` | coherence delta > 0 |
| `high_rupture_event` | rupture ≥ 0.4 |
| `chaotic_price_action` | entropy ≥ 0.9 (or, if entropy missing, coherence far from 1) |
| `low_hazard_environment` | hazard or λ ≤ 0.1 |
| `improving_stability` | λ slope < 0 |

Thresholds are configurable via `SemanticThresholds` or a simple override dict.
The tagger gracefully handles Valkey gate payloads and synthetic fallback gates.

## Regime-Aware Backtesting

`scripts/run_unified_backtest.py` now accepts `--semantic-regime-filter` and
writes the tags for every minute into the JSON output (`decisions[].semantic_tags`).
Key CLI options:

- `--config` – JSON/YAML file describing window, instruments and default filters.
- `--semantic-regime-filter` – comma-separated tags applied to all instruments.
- `--semantic-threshold` – overrides such as `high_coherence=0.75`.
- `--output` – output path (defaults to `output/backtests/unified_latest.json`).

The script reads the new `configs/research/semantic_pilot.json` file by default
when invoked via `make unified-backtest`.

## Discovery Loop

Use `make discover-semantic-regimes` to brute-force profitable combinations. The
rule executes `scripts/research/semantic_regime_discovery.py`, which in turn
runs `make unified-backtest` for each tag/instrument pair, captures the JSON
output, and assembles a CSV report at `output/semantic_regime_summary.csv`.

After each sweep, `scripts/research/semantic_regime_report.py` helps rank the
top-performing regimes:

```bash
PYTHONPATH=/sep .venv/bin/python scripts/research/semantic_regime_report.py \
  output/semantic_regime_summary.csv --sort sharpe --top 3
```

Environment variables for the Makefile target:

- `CONFIG` – path to the backtest config (defaults to `configs/research/semantic_pilot.json`).
- `DISCOVER_ARGS` – optional flags forwarded to the discovery script (e.g.
  `DISCOVER_ARGS='--regime improving_stability --quiet'`).

The CSV includes PnL, Sharpe, trade count, and tag utilisation metrics so you
can immediately flag the regimes worth piloting.

## From Discovery to Pilot

1. Run the discovery loop and inspect `output/semantic_regime_summary.csv` for
   regimes with stable Sharpe and acceptable drawdowns.
2. Update `configs/research/semantic_pilot.json` with the per-instrument tags you
   want to enforce.
3. Execute `make unified-backtest ARGS='--config configs/research/semantic_pilot.json'`
   to validate the combined portfolio.
4. Promote the winning configuration into the production profile once you have a
   statistically significant sample and live confirmation.

## Relationship with the SCORE Toolkit

The embedded SCORE repository (`/sep/score`) still hosts the broader semantic
framework and explains how the quantum metrics projection is wired into other
modalities. The following resources remain useful when extending the regime
library:

- `score/docs/integration_with_sep.md` – mapping between SEP C++ primitives and
  the SCORE adapters.
- `score/scripts/semantic_bridge_demo.py` – examples of translating metric
deltas into categorical features for non-trading datasets.

Keep the tag definitions aligned across repositories to ensure the reporting
layer can merge trading results with SCORE experiments.
