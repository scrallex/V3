# Manifold Compression for Long-Context Agents

## Scope and Intent
- **Objective:** Prove that SEP's manifold signatures alone can act as a high-ratio compression medium for long textual contexts while preserving verification power.
- **Deliverable:** A reproducible, citable paper and code bundle that quantifies compression, verification accuracy, and operational envelopes using only structural signatures—no optical or vision components.
- **Hypotheses to Validate:**
  1. Structural signatures (coherence, stability, entropy buckets) compress long contexts by ≥10× relative to raw UTF-8 bytes when stored as deduplicated repetitions.
  2. The compressed manifold can verify membership of unseen spans (window-level recall ≥95%) with negligible false positives.
  3. Hazard λ and repetition counts enable adaptive confidence bands similar to optical confidence thresholds but with lower memory footprints.

## Paper Outline
1. **Abstract** – Highlight compression ratio, verification fidelity, and operational relevance.
2. **Introduction**
   - Motivation: LLM agents hit context ceilings; optical tricks are not the only path.
   - Recap of SEP manifold philosophy: rhythm + hazard + repetition as a compressed memory.
   - Contributions: compression protocol, verification benchmarks, reproducible tooling.
3. **Related Work**
   - Token-level compression techniques (e.g., LongT5, FlashAttention).
   - Optical compression (DeepSeek-OCR) as an external reference point.
   - Structural reliability metrics (manifold hazard / stability guards, STM guardrails)【score/docs/whitepaper/QFH_Manifold_Foundation.tex:16】.
4. **Methodology**
   - Window construction: byte windows, stride, signature precision.
   - Compression scheme: per-document signature deduplication, hazard and repetition summaries.
   - Verification protocol: membership checks, hazard thresholds, delta analysis.
5. **Experimental Setup**
   - Datasets: `docs/whitepaper/sample_data`, SEP design docs, optional trading logs.
   - Metrics: compression ratio, recall/precision/F1, hazard-aligned ROC.
   - Tooling: `scripts/experiments/manifold_compression_eval.py`, `make manifold-compression`, ablation scripts.
6. **Results**
   - Compression tables vs precision/stride.
   - Verification curves, confusion matrices, hazard correlation.
   - Ablations: signature precision, window size, stride, hazard caps.
7. **Discussion**
   - Operational integration (Valkey manifolds, PortfolioManager gating).
   - Limitations: semantics lost without text payload, multilingual considerations.
   - Comparison with optical approaches (memory, throughput, deployment complexity).
8. **Reproducibility Checklist**
   - Repository state / commit hash.
   - Commands (`make manifold-compression`, optional ablations).
   - Data manifests, checksum logs, expected runtimes.
9. **Conclusion** – Structural manifolds deliver high compression with verifiable fidelity for long-context agents.

## Execution Plan
1. Expand sample corpora by adding SEP docs and trading excerpts into `docs/whitepaper/sample_data` with regeneration scripts (`scripts/experiments/build_manifest.py`).
2. Run baseline evaluation via `make manifold-compression` to produce `output/manifold_compression_summary.json`.
3. Launch ablation sweeps (`make manifold-compression-sweep` → `output/manifold_compression_sweep.jsonl`) covering precision ∈ {1,2,3}, stride variations, and hazard caps to populate the Results section.
4. Draft notebooks/plots that convert JSON summaries into compression curves and ROC charts.
5. Write paper sections following the outline, embedding reproducible tables/figures.
6. Document the operational playbook for consuming compressed manifolds inside PortfolioManager/STM, including hazard gating thresholds and Valkey storage guidance.

## Optical Baseline Alignment (DeepSeek vs. Manifold)
1. **Subset selection:** Use the same manifest order for both methods. Limit evaluation to the first `N` samples (e.g., `N=150`) so runtimes stay manageable while preserving diversity.
2. **DeepSeek-OCR run (images → text signatures):**
   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/experiments/deepseek_ocr_runner.py \
     --dataset fox=data/benchmark_corpus/fox/metadata/text_manifest.jsonl:data/benchmark_corpus/fox/raw \
     --dataset omnidoc=data/benchmark_corpus/omnidocbench/metadata/text_manifest.jsonl:data/benchmark_corpus/omnidocbench/raw/OmniDocBench \
     --prompt "<image>\nFree OCR." \
     --model-name external/DeepSeek-OCR/weights \
     --trust-remote-code --dtype bfloat16 --device cuda \
     --attn-impl eager --max-records 150
   ```
   - Outputs aggregate metrics at `output/deepseek_runs/summary.json` and per-doc logs in `output/deepseek_runs/{fox,omnidoc}.jsonl`.
   - `--max-records` enforces the shared 150-doc slice; drop it for the full benchmark.
3. **Manifold structural run (text → manifold signatures) on the exact same slice:**
   ```bash
   python scripts/experiments/benchmark_eval.py \
     --dataset fox=data/benchmark_corpus/fox/text/en_page_ocr \
     --dataset fox_cn=data/benchmark_corpus/fox/text/cn_page_ocr \
     --dataset omnidoc=data/benchmark_corpus/omnidocbench/text \
     --window-bytes 512 --stride-bytes 384 --precision 3 \
     --tokenizer external/DeepSeek-OCR/weights --tokenizer-trust-remote-code \
     --max-documents 150 \
     --output-dir output/benchmark_runs/subset_150
   ```
   - `--max-documents` caps each dataset to the same number of documents the optical run saw.
   - The resulting CSV/JSON summaries live under `output/benchmark_runs/subset_150/`.
4. **Compare metrics:** Align `compression_ratio`, `token_accuracy`, and hazard-gating precision using:
   - Optical baseline: `output/deepseek_runs/summary.json`.
   - Manifold baseline: `output/benchmark_runs/subset_150/summary.csv`.
5. **Narrative hooks for the whitepaper:**
   - Optical compression hits ~5× at 90%+ accuracy on Fox; our manifold hits ~45× at ~91% token accuracy (fox_en summary, `output/benchmark_runs/full/fox_en.json`).
   - On OmniDoc, manifold maintains 10×+ unique-token compression without vision tokens but sees accuracy drop—use DeepSeek’s subset numbers to contextualize failure modes.
   - Hazard gating (false-positive rate in `verification.false_positive_rate`) provides a differentiator absent in the optical pipeline.
