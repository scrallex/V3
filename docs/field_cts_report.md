# Deterministic Convolution on Binary Manifolds: Contractive vs Expansive Dynamics

## Abstract
We quantify information dynamics in deterministic binary manifolds under local update rules. Majority-radius (contractive) and Rule-30 (expansive) maps are evaluated via six observables: perturbation growth, domain-wall density, lagged coherence, temporal mutual information, compressibility, and spectral slope. Across diverse seeds and grid sizes on GPU, contractive dynamics collapse perturbations, shift spectral power to low frequencies, reduce domain walls, and maintain higher temporal mutual information; expansive dynamics do the opposite. Lag acts as an information lens: coherence decreases with lag for contractive rules but remains pinned for expansive ones. Early-phase behaviour under a tie-drop policy (averaging the first eight phases) shows monotone decreases in rupture and coherence with neighbourhood radius. The results furnish fast, reproducible diagnostics for distinguishing structure-forming versus entropy-spreading regimes and principled “weight/wait” knobs for binary-field systems.

## Methods Snapshot
- **Kernel policy:** majority uses tie-drop (`keep_self_on_tie = false`); Cellular Automata rules use the standard masks. Boundary policy is clamped unless otherwise stated.
- **Averaging window:** unless noted, metrics shown for Fig. 2 and Fig. 3 use the mean of the first eight phases to emphasise pre-settling behaviour.
- **Seeds & grids:** checkerboard, stripes, discs, Gray, LFSR, and Bernoulli `p∈{0.1,0.3,0.5,0.7,0.9}` on 256² and 512² lattices.
- **Statistics:** medians across seeds with 95 % bootstrap CI bands (500 resamples); Spearman ρ reported for monotonic trends. GPU runs executed via `python apps/field_cts/scripts/run_sweeps.py`.

## Results (Fig. 1–Fig. 6)
- **Perturbation growth (Fig. 1):** majority collapses perturbations to ≈5×10⁻⁴ by phase 128; Rule-30 amplifies to ≈3–4 % Hamming distance. Confidence bands sit well apart, yielding a Lyapunov-sign split.
- **Radius sweep (Fig. 2):** under tie-drop, early-phase rupture-per-bit falls monotonically 0.499 → 0.443 (r=0 → 4) and coherence falls 1.000 → ≈0.90. Larger neighbourhoods enforce heavier early smoothing before convergence.
- **Lag sweep (Fig. 3):** majority coherence decays with τ (r=1: 0.994→0.976; r=3: 0.965→0.871); Rule-30 remains near 0.5. Temporal MI stays high for contractive rules (~0.73–0.95) and ≈0.03 for expansive ones. Lag is an information lens.
- **Compressibility (Fig. 4):** majority fields remain highly compressible (gzip ratio ≈0.27) after settling; Rule-30 rises toward ≈1.0 (incompressible). Treat as intuition; MI/spectra carry the claim.
- **Domain walls (Fig. 5):** majority r=1 reduces domain-wall density a few basis points over 128 phases (e.g., 0.244→0.240); expansive rules do not exhibit sustained decay.
- **Spectral slope (Fig. 6):** majority pushes radial power to low frequencies between phase 1 and phase 128; Rule-30 stays broadband with high-frequency mass intact.

## Weight & Wait Controls
- **Weight (radius r):** larger neighbourhoods front-load smoothing pressure. With tie-drop and early-phase averaging the response is strictly monotone—rupture-per-bit drops 0.499→0.443 (r=0→4) and coherence falls 1.000→≈0.90. Bigger r “spends” edits early and leaves fewer boundaries to resolve later.
- **Wait (lag τ):** looking farther apart in time is an information lens. Majority coherence declines with τ (r=1: 0.994→0.976; r=3: 0.965→0.871) while expansive rules are already decorrelated and barely move. The coherence(τ) slope is therefore a quick detector of drift in convergent manifolds.

## Gate Integration
- **Structural metrics now exported:** `coherence_tau_slope` (τ∈{1,4} slope), `domain_wall_slope` (first-half vs second-half decay), and `spectral_lowf_share` (low-frequency power share). These flow directly into `gate:last:{instrument}` alongside coherence/stability/entropy.
- **Scoring stencil:** for a window `W`,  
  `score(W) = α * (-d/dτ coherence_τ|_{τ∈{1,4}}) + β * (-domain_wall_slope) + γ * (low_frequency_power_share) + δ * (temporal_MI)`  
  with α…δ defaulting to 1.0. Higher scores favour structure-forming, contractive behaviour; low scores indicate entropy-spreading regimes.
- **Default panels:**  
  1. *Conservative* – r=3, τ∈{1,4}; require strongly negative slopes before promoting trades.  
  2. *Responsive* – r=1 with τ∈{1,4}; quicker to acknowledge forming structure while still penalising positive slopes.
- **Guardrail policy:** live strategy profiles enforce `max_coherence_tau_slope`, `max_domain_wall_slope`, and `min_low_freq_share` thresholds (see `config/echo_strategy.yaml`). Negative slopes plus rising low-frequency share keep the gate open; positive slopes or whitening spectra now block allocations.

## Figure Captions
1. **Perturbation growth:** contractive dynamics annihilate tiny differences; expansive ones amplify them.
2. **Radius sweep (early-phase mean):** increasing radius monotonically reduces rupture and coherence—the system spends edits early, then settles.
3. **Lag sweep (early-phase mean):** lag reveals contractive drift (coherence ↓ with τ); expansive rules remain decorrelated.
4. **Compressibility trajectory:** majority becomes/stays highly compressible; Rule-30 approaches incompressible.
5. **Domain-wall energy:** boundaries decay under majority; not under expansive rules.
6. **Spectral slope:** majority shifts power to low frequencies; Rule-30 keeps broad high-frequency content.

## Boundary Policy Ablation
A targeted ablation with periodic wrapping (256², Bernoulli p=0.5, 4 replicates) confirms the qualitative signs:
- **Radius monotonicity** (rupture and coherence) holds for both clamped and periodic boundaries (monotone decreases with r).
- **Lag decay** for majority (r=1,3) remains monotone with τ under both boundary conditions; expansive rules stay flat at low coherence.
The JSON artefact is emitted at `apps/field_cts/output/analysis/boundary_ablation.json`.

## Threats to Validity
- **Complexity proxy:** gzip is only a proxy; MI and spectra mitigate but do not replace formal complexity measures.
- **Finite-size artefacts:** 256² and 512² grids show identical directional behaviour; larger lattices remain future work.
- **Policy sensitivity:** tie-drop is essential for exposing early-phase gradients—documented here. Boundary policy was ablated (clamped vs periodic) with no sign flips, but absolute levels do shift.

## Reproduction Assets
- `python apps/field_cts/scripts/run_sweeps.py`
- `python apps/field_cts/scripts/plot_figures.py`
- Outputs under `apps/field_cts/output/` (metrics, states, figures, seeds, params.json).
- Expectations codified in `apps/field_cts/EXPECTED.md`.
- Release record: `docs/releases/v0.1-field-cts.md`.
