# Structural Manifold Compression

**Text-only manifold signatures that compress Fox EN/CN and OmniDocBench by 42× on bytes / 85–90× on tokens while preserving ≥ 94.9 % token accuracy, ≤ 5.1 % normalized edit distance, and 80–97 % verification precision with < 0.09 % false-positive rate.** Runs complete in < 1 hour on a single RTX 3080 Ti. Full methodology and results live in [`docs/manifold_vs_optical/report.pdf`](docs/manifold_vs_optical/report.pdf).

---

## 1. Overview & Contributions

- **Sliding-window manifold signatures:** 512 B windows, 384 B stride, quantized coherence/stability/entropy/hazard packed into a 9 B payload (+ repetition count).
- **Perfect-recall hazard gating:** Cross-document verifier reuses the hazard prior to audit collisions; precision hits 91.2 % (Fox EN), 97.2 % (Fox CN), 80.9 % (OmniDoc) with FPR < 0.09 %.
- **End-to-end reproducibility:** `scripts/experiments/benchmark_eval.py` regenerates all CSV/JSON metrics cited in the report; `make report` rebuilds the PDF.
- **Optical baseline harness:** `scripts/experiments/deepseek_ocr_runner.py` replays DeepSeek-OCR on the same manifest for apples-to-apples comparisons.

If you only want the narrative, figures, and tables, read the PDF:  
📄 [`docs/manifold_vs_optical/report.pdf`](docs/manifold_vs_optical/report.pdf)

---

## 2. Benchmark Snapshot (Full Run @ RTX 3080 Ti)

| Dataset | Docs | Byte × | Token × | Token Acc. | Char Acc. | Verif. Precision | Verif. FPR |
|---------|-----:|-------:|--------:|-----------:|----------:|-----------------:|-----------:|
| Fox EN  | 112 | 42.03 | 85.48 | 95.35 % | 95.62 % | 91.21 % | 0.087 % |
| Fox CN  | 100 | 42.01 | 88.08 | 94.94 % | 95.04 % | 97.19 % | 0.029 % |
| OmniDoc | 1 349 | 41.59 | 89.49 | 94.90 % | 94.94 % | 80.85 % | 0.017 % |

Source: [`output/benchmark_runs/full_benchmark/summary.csv`](output/benchmark_runs/full_benchmark/summary.csv)

---

## Quick Start (Structured Text Demo)

```bash
cd structural-manifold-compression
source .venv/bin/activate
python scripts/experiments/benchmark_eval.py \
  --dataset briefs=examples/structured_demo/news_sample.jsonl \
  --json-text-key text \
  --window-bytes 512 --stride-bytes 384 --precision 3 \
  --use-native \
  --output-dir output/benchmark_runs/news_demo
```

- Ships with a tiny JSONL corpus (`examples/structured_demo/news_sample.jsonl`) that mimics PDF/news briefs. Even this toy run demonstrates 30×+ byte compression, 50×+ token compression, and negligible verification false positives.
- Swap `briefs=...` for any structured corpus (JSONL/JSON/txt). Pass multiple `--dataset label=path` entries to compare sources (Fox EN vs. CN, R&D vs. Ops, etc.).
- `output/benchmark_runs/news_demo/briefs.json` records byte/token ratios, per-document reconstructions, and verification stats. Attach this JSON to issues, blog posts, or audit reports.
- For perplexity vs. GPT‑2 baselines, use `scripts/experiments/perplexity_compare.py` with your manifold checkpoint and matching raw text (see §“Benchmark vs. GPT‑2”).
- Prefer zero-setup? A Hugging Face Space (in progress) will expose the same workflow via Gradio so reviewers can upload JSONL dumps, run compression, and inspect reconstructions/verification from the browser.

---

## Examples & Notebooks

- `examples/structured_demo/news_sample.jsonl` — three briefing-style docs that mimic PDF/corpora use cases.
- `examples/structured_demo/README.md` — shell-friendly walkthrough that benchmarks the sample corpus and explains how to swap in your own JSONL.
- `examples/structured_demo/mini_pipeline.ipynb` — notebook version of the same commands so you can annotate or share results without leaving Jupyter.

Use these assets for smoke tests, demos, or as a template when filing reproduction issues (attach the JSON + metrics emitted to `output/benchmark_runs/news_demo/`).

---

## 3. Repository Layout

```
data/benchmark_corpus/      # Fox / OmniDoc text dumps (symlink; not committed)
docs/
  manifold_vs_optical/
    report.tex              # LaTeX source
    report.pdf              # Ready-to-share manuscript
scripts/
  experiments/
    benchmark_eval.py       # Structural manifold benchmark
    deepseek_ocr_runner.py  # Optical baseline (DeepSeek-OCR)
    plot_manifold_sweep.py  # Curves for the report
  data/
    download_corpora.py     # Fox + OmniDocBench downloader + manifest builder
  training/
    ocr_trainer.py          # LoRA-ready OCR finetuning harness
    run_pipeline.py         # Stageable download/train orchestrator
src/                        # Encoder + manifold helpers
output/                     # Generated summaries/plots
Makefile                    # install | native | full-run | report
docs/03_training_playbook.md# Deployment + training roadmap
```

---

## 4. Setup

```bash
git clone https://github.com/SepDynamics/structural-manifold-compression.git
cd structural-manifold-compression
python3 -m venv .venv && source .venv/bin/activate
make install            # installs Python deps
make native             # optional, builds CUDA kernel if nvcc is present
```

### Dataset & Weights

1. **Fox benchmark** (English + Chinese) text manifests → place under `data/benchmark_corpus/fox/text/{en_page_ocr,cn_page_ocr}`.
2. **OmniDocBench** page-level text → `data/benchmark_corpus/omnidocbench/text`.
3. Keep datasets outside Git; symlink them in if needed: `ln -s /data/share benchmark_corpus/data`.
4. Place the DeepSeek-OCR weights under `external/DeepSeek-OCR/weights` (symlink `external` if you reuse a global models directory).

To pull the corpora automatically (Fox + OmniDoc) and regenerate manifests, run:

```bash
./scripts/training/run_pipeline.py download
# or with filters, e.g.:
# ./scripts/training/run_pipeline.py download --datasets fox
```

The helper wraps `scripts/data/download_corpora.py`, which uses Hugging Face snapshots. Set `HF_TOKEN=...` when downloading from private mirrors.

### Custom Corpora & Manifold LM Prep

Bring your own UTF-8 corpus (plain `.txt`, `.jsonl`, or `.json`) and run the manifold encoder end-to-end:

```bash
# 1) Benchmark compression / fidelity on arbitrary text
python scripts/experiments/benchmark_eval.py \
  --dataset wikitext=data/raw_text/wikitext_train.jsonl \
  --json-text-key text \
  --window-bytes 512 --stride-bytes 384 --precision 3 \
  --output-dir output/benchmark_runs/wikitext_custom

# Run the same benchmark inside a CUDA-ready Docker image (works even when local nvcc breaks):
CUDA_IMAGE=nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04 \
DATASET=wikitext=data/raw_text/wikitext_train.jsonl \
OUTDIR=output/benchmark_runs/wikitext_custom_gpu \
scripts/experiments/run_benchmark_docker.sh

# 2) Build (and resume) the HF dataset used for causal training
python scripts/data/prepare_causal_dataset.py \
  --text-root data/raw_text/wikitext_train.jsonl \
  --output-dir output/wikitext_manifold \
  --window-bytes 512 --stride-bytes 384 --precision 3 \
  --sequence-length 512 --min-sequence-length 8 \
  --use-native --export-signatures --concat-documents --reset-output

# Re-run the same command (without --reset-output) any time to resume from
# the last processed document. Progress lives under output/wikitext_manifold/.
```

Artifacts:
- `output/benchmark_runs/wikitext_custom/*.json`: compression + fidelity metrics.
- `output/wikitext_manifold/samples.jsonl`: append-only store of every manifold sequence (safe to resume).
- `output/wikitext_manifold/processed_docs.txt`: set of completed documents (reruns skip them automatically).
- `output/wikitext_manifold/hf_dataset`: Hugging Face dataset (`input_ids`, `labels`) for GPT-style training.
- `output/wikitext_manifold/vocab.json`: signature → token-id mapping (index corresponds to ID).
- `output/wikitext_manifold/metadata.json`: run stats (doc count, sample count, effective vocab, etc.).
- `output/wikitext_manifold/signatures/<doc_id>.json` (optional): per-document signature streams for inspection.

Interrupt the builder at any time—rerunning the command continues where it left off, honors `--concat-documents` (so short texts still form long sequences), and refreshes the `hf_dataset` shard.

### Train a Manifold LM on a 3080 Ti

Once the dataset exists, launch the scratch GPT-style run (≈250 M params, tuned for a single 12 GB card):

```bash
# one-liner (honours MANIFOLD_PY / CUDA overrides)
make train-manifold-gpt

# or manual invocation
RUN_DIR=output/training_runs/wikitext_manifold_gpt
python scripts/training/manifold_lm_trainer.py \
  --dataset-path output/wikitext_manifold/hf_dataset \
  --vocab-path output/wikitext_manifold/vocab.json \
  --output-dir "${RUN_DIR}" \
  --n-layer 16 --n-head 16 --n-embd 1024 \
  --context-length 512 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --num-train-epochs 3 \
  --warmup-steps 500 \
  --eval-holdout 0.02 \
  --gradient-checkpointing \
  --fp16 \
  --resume
```

Why these knobs:
- 16×16×1024 decoder lands in the 230–250 M parameter band but still fits the 3080 Ti when combined with FP16 + gradient checkpointing.
- Batch size 2 × grad-accum 16 ⇒ effective 32 sequences/step (~16k manifold tokens). With ~200k samples, you get ≈6.2k steps/epoch → ~18.6k updates for 3 epochs (≈2–3 h wall-clock on the 3080 Ti).
- `--resume` inspects `${RUN_DIR}` for `checkpoint-*` folders and picks up automatically after interruptions (Ctrl+C, reboot, etc.).

Monitoring & verification:
- `tensorboard --logdir ${RUN_DIR}` to follow loss curves mid-run.
- Each checkpoint stores HF-compatible weights under `${RUN_DIR}/checkpoint-XXXX`.
- After training, the script prints `eval_loss` + `perplexity` on the held-out split (already reconstructed for manifold tokens).

### Benchmark vs. GPT-2 (Perplexity & Reconstruction)

1. **Long-context fidelity / compression** (structural reconstruction metrics):
   ```bash
   python scripts/experiments/benchmark_eval.py \
     --dataset wikitext=data/raw_text/wikitext_train.jsonl \
     --json-text-key text \
     --window-bytes 512 --stride-bytes 384 --precision 3 \
     --output-dir output/benchmark_runs/wikitext_custom \
     --use-native
   ```
   Inspect `output/benchmark_runs/wikitext_custom/wikitext.json` for token accuracy, compression ratio, and verification stats—the numbers quantify how 512-signature contexts preserve 20k+ raw tokens.

2. **A/B perplexity (manifold LM vs. GPT-2 on raw text)**:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/experiments/perplexity_compare.py \
     --manifold-model output/training_runs/wikitext_manifold_gpt \
     --manifold-dataset output/wikitext_manifold/hf_dataset \
     --manifold-vocab output/wikitext_manifold/vocab.json \
     --manifold-eval-fraction 0.25 \
     --manifold-batch-size 8 \
     --gpt2-model gpt2-medium \
     --raw-text data/raw_text/wikitext_train.jsonl \
     --json-text-key text \
     --gpt2-block-size 1024 \
     --gpt2-max-documents 500 \
     --output output/benchmark_runs/wikitext_perplexity.json
   ```
   The script reports both perplexities plus a compression proxy (`raw_tokens / manifold_tokens`) so you can cite the effective gain in sequence length. Adjust `--gpt2-max-documents` or switch to another baseline (e.g. `gpt2-large`) as needed.

---

## 5. Reproduce the Structural Benchmark

```bash
python scripts/experiments/benchmark_eval.py \
  --dataset fox=data/benchmark_corpus/fox/text/en_page_ocr \
  --dataset fox_cn=data/benchmark_corpus/fox/text/cn_page_ocr \
  --dataset omnidoc=data/benchmark_corpus/omnidocbench/text \
  --window-bytes 512 --stride-bytes 384 --precision 3 \
  --tokenizer external/DeepSeek-OCR/weights --tokenizer-trust-remote-code \
  --output-dir output/benchmark_runs/full_benchmark
```

Outputs:
- CSV: `output/benchmark_runs/full_benchmark/summary.csv` (table above)
- JSON: dataset- and per-document stats (`fox.json`, `fox_cn.json`, `omnidoc.json`)

### Optional: Optical Baseline (Subset)

```bash
python scripts/experiments/deepseek_ocr_runner.py \
  --dataset fox=data/benchmark_corpus/fox/metadata/text_manifest.jsonl:data/benchmark_corpus/fox/raw \
  --dataset omnidoc=data/benchmark_corpus/omnidocbench/metadata/text_manifest.jsonl:data/benchmark_corpus/omnidocbench/raw/OmniDocBench \
  --prompt "<image>\nFree OCR." \
  --model-name external/DeepSeek-OCR/weights \
  --trust-remote-code --dtype bfloat16 --device cuda --attn-impl eager \
  --max-records 150 \
  --output output/deepseek_runs
```

---

## 6. Rebuild the Report

```bash
make report   # runs pdflatex twice, emits docs/manifold_vs_optical/report.pdf
```

The PDF includes methodology, metric definitions, full benchmark tables, DeepSeek comparison, limitations, and step-by-step reproducibility instructions.

---

## 7. Make Targets

| Target          | Description |
|-----------------|-------------|
| `make install`  | Install Python dependencies into `.venv`. |
| `make native`   | Build the optional CUDA kernel (`scripts/utils/native_kernel.cu`). |
| `make full-run` | Shortcut for the structural benchmark command above. |
| `make report`   | Compile the LaTeX report into `docs/manifold_vs_optical/report.pdf`. |
| `make docker`   | Build a `manifold-compression:latest` image (requires datasets mounted at runtime). |

---

## Release Channels & Hugging Face Space

- **GitHub (this repo)** – canonical scripts, docs, and benchmarks. Open issues with the JSON/CSV outputs from `benchmark_eval.py` so we can reproduce quickly.
- **Hugging Face Model** – [`scrallex/structural-manifold-compression`](https://huggingface.co/scrallex/structural-manifold-compression) hosts the 8 h manifold LM checkpoint plus the latest benchmark JSON (Wikitext slice, perplexity comparison, etc.). The model card mirrors the Quick Start commands and tables above.
- **Upcoming HF Space** – a Gradio app that:
  1. Lets you upload JSONL/txt exports (e.g., PDF OCR, news briefs) and runs `benchmark_eval.py` with adjustable window/stride/precision.
  2. Displays byte/token compression, token accuracy, verification precision/FPR, and reconstructed snippets.
  3. Provides a verification tab to compare two documents/hazard profiles.
  4. Optionally compares manifold LM perplexity vs. GPT‑2 on the uploaded text.

Once the Space is live it will be embedded in both this README and the HF model card so newcomers can test the workflow without a local GPU.

---

## 8. License & Citation

This project is released under the [MIT License](LICENSE). If you build on these scripts, please cite the 2025 structural manifold release:

```bibtex
@misc{nagy2025manifold,
  author       = {Alexander Nagy},
  title        = {Structural Manifold Compression: A Text-Only Alternative to Optical Context Encoding},
  year         = {2025},
  howpublished = {\url{https://github.com/SepDynamics/structural-manifold-compression}}
}
```

## 9. OCR Fine‑Tuning & Deployment

- Read `docs/03_training_playbook.md` for the end-to-end plan (production integration → language expansion → hardware scaling → robustness).
- Spin up finetuning runs with `scripts/training/ocr_trainer.py`, which consumes the existing Fox/OmniDoc manifests and supports mixed precision, gradient checkpointing, LoRA, and token/character F1 tracking.
- Example (single RTX 3080 Ti, English Fox pages):

  ```bash
  python scripts/training/ocr_trainer.py \
    --train-dataset fox_en=data/benchmark_corpus/fox/metadata/text_manifest.jsonl:data/benchmark_corpus/fox/raw \
    --include-language english \
    --val-split 0.08 \
    --model-id microsoft/trocr-base-printed \
    --output-dir output/training_runs/trocr_fox_en \
    --epochs 3 \
    --train-batch-size 1 \
    --gradient-accumulation 8 \
    --learning-rate 1e-4 \
    --gradient-checkpointing \
    --fp16 \
    --lora-rank 8
  ```

- TensorBoard logs, checkpoints, and eval summaries will land in `output/training_runs/<run-name>`; feed the resulting adapters back into the benchmarking scripts to compare against DeepSeek-OCR and the manifold baselines.

- ### One-Command Pipeline + Resume

- `./scripts/training/run_pipeline.py all` → downloads (if needed), prepares manifests, and launches training with the default Fox+OmniDoc mix.
- `./scripts/training/run_pipeline.py train` → starts (or resumes) training only. Pass `RESUME_FROM=output/training_runs/<run>/checkpoint-XXXX` to pick up after an interruption.
- Tweak env vars instead of editing the script:
  - `RUN_NAME=my_run` (changes output directory).
  - `TRAIN_DATASETS="fox_en=...:...,omnidoc=..."` (choose subsets).
  - `INCLUDE_LANGUAGES=english,chinese` (language filters) or leave unset for all text.
  - `PRECISION=bf16`, `LORA_RANK=4`, `VAL_DATASETS=...` etc.
- Any extra CLI flags appended after the stage name are forwarded straight to `ocr_trainer.py`.

Questions or reproducibility issues? File an issue or ping **@alexandernagy**. Every figure and table is derived directly from the scripts and datasets above. Happy verifying!
