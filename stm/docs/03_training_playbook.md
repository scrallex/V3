# OCR Deployment & Training Playbook

This playbook converts the R&D roadmap (integration → expansion → scaling → robustness) into concrete actions inside this repository so you can launch supervised finetuning runs on a single RTX 3080 Ti and iterate toward production demonstrations.

## Phase Roadmap

1. **Production Integration (now)**
   - Ingest OCR metrics via the manifold service (`scripts/experiments/benchmark_eval.py`) and expose them through automation jobs or `/api/...` health probes.
   - Use the new training harness (`scripts/training/ocr_trainer.py`) to produce adapters that minimize normalized edit distance while holding 40×–45× compression. Store checkpoints under `output/training_runs/<date>`.
   - Wire the resulting adapters into portfolio or invoice processors by feeding compressed spans from `gate:last:{instrument}` into the OCR model before the portfolio manager consumes them.
2. **Task & Language Expansion**
   - Extend manifests under `data/benchmark_corpus/{fox,omnidocbench}` with handwritten, chart-heavy, or code-heavy subsets (`scripts/experiments/build_manifest.py` already handles JSONL generation).
   - Add additional JSONL manifests for public corpora (ICDAR, PubLayNet, DocVQA) and point `--train-dataset` to them. Use `--include-language zh`, `--include-subset handwritten`, etc., to create dedicated adapters.
   - Schedule reasoning-style probes (needle-in-haystack, JSON/LaTeX targets) by emitting structured text labels from the same manifest rows.
3. **Scaling & Efficiency**
   - Run adapters in modes (`tiny`, `small`, `dense`) by saving separate configs with different `--max-target-length`, LoRA ranks, and resolution-specific manifests.
   - Toggle 4-bit/8-bit loading externally (via `bitsandbytes`) and rely on small LoRA ranks (`--lora-rank 8`) plus gradient checkpointing for 12 GB VRAM budgets.
   - Export trained checkpoints to Hugging Face Spaces or an internal wheel; combine with `scripts/experiments/run_optical_inference.py` to benchmark token/F1 vs. compression tradeoffs automatically.
4. **Ethics & Robustness**
   - Balance datasets with noisy scans and synthetic distortions (TRDG, random blur/noise augmentation) before each run.
   - Track per-subset FPR/FNR by parsing the trainer’s eval JSON logs and alerting when metrics drift past thresholds noted in `docs/optical_structural_compression.md`.
   - Publish “OCR efficiency scores” by comparing DeepSeek-OCR baselines with your adapters using the shared manifests and `output/deepseek_runs/*.jsonl`.

## Environment & Data Prereqs

1. `cd structural-manifold-compression`
2. `python3 -m venv .venv && source .venv/bin/activate`
3. `make install` (installs torch, transformers, peft, accelerate, tensorboard, etc.)
4. Pull corpora with `./scripts/training/run_pipeline.py download` (wraps `scripts/data/download_corpora.py` + manifest generation). Pass `--datasets fox` or `--datasets omnidoc` to limit the scope; set `HF_TOKEN` if the mirrors require authentication.
5. (Optional) Mount additional datasets (ICDAR, PubLayNet, DocVQA) under `data/benchmark_corpus/<name>` following the same manifest schema.

### Stageable Commands

| Stage | Command | Notes |
|-------|---------|-------|
| Download + prepare | `./scripts/training/run_pipeline.py download` | Fetches Fox + OmniDoc, unpacks, rebuilds manifests. Use `prepare` to regenerate manifests only. |
| Training (fresh) | `RUN_NAME=fox_stage1 ./scripts/training/run_pipeline.py train` | Uses defaults (Fox EN + CN + OmniDoc). Override env vars (e.g., `TRAIN_DATASETS`, `INCLUDE_LANGUAGES`, `PRECISION`) or append extra CLI flags after `train`. |
| Training (resume) | `RESUME_FROM=output/training_runs/fox_stage1/checkpoint-1200 ./scripts/training/run_pipeline.py train` | Start TensorBoard (`tensorboard --logdir output/training_runs`) before leaving the machine; CTRL+C stops safely, re-run with `RESUME_FROM` to continue. |
| Full pipeline | `./scripts/training/run_pipeline.py all` | Runs download (if needed) then training in one shot. |

## Training Harness (`scripts/training/ocr_trainer.py`)

The trainer consumes one or more manifest/image-root pairs, builds PyTorch datasets on the fly, and exposes all knobs needed for 3080 Ti-class finetuning (mixed precision, gradient checkpointing, LoRA).

### CLI Cheatsheet

```
python scripts/training/ocr_trainer.py \
  --train-dataset fox_en=data/benchmark_corpus/fox/metadata/text_manifest.jsonl:data/benchmark_corpus/fox/raw \
  --include-language english \
  --val-split 0.08 \
  --model-id microsoft/trocr-large-printed \
  --output-dir output/training_runs/trocr_fox_en \
  --epochs 3 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --gradient-checkpointing \
  --fp16 \
  --lora-rank 8 \
  --lora-target-modules q_proj,k_proj,v_proj,out_proj \
  --metric-tokenizer external/DeepSeek-OCR/weights \
  --max-target-length 640 \
  --generation-max-length 768 \
  --logging-steps 20 \
  --save-steps 200 \
  --eval-steps 200
```

Key behaviors:

- Manifests are shuffled per run; set `--seed` for reproducibility.
- `--train-dataset` / `--val-dataset` accept multiple entries; if no validation manifest is passed, `--val-split` carves examples from the training pool.
- `--include-language` / `--include-subset` filter rows (handwritten, legal, receipts, etc.).
- LoRA is optional; leaving `--lora-rank 0` performs full fine-tuning.
- Metrics are aggregated with the same token/character editors used in `scripts/experiments/manifold_compression_eval.py`, so token accuracy/precision map directly to the report.
- Outputs (checkpoints, `trainer_state.json`, TensorBoard logs) land in `--output-dir`.

### Recommended 3080 Ti Run Card

| Setting | Value |
|---------|-------|
| Model | `microsoft/trocr-base-printed` or `qwen/Qwen2-VL-2B-Instruct` (if VRAM allows) |
| Precision | `--fp16` |
| Batch | `--train-batch-size 1`, `--gradient-accumulation 8` (effective batch 8) |
| LoRA | `--lora-rank 8`, dropout 0.05 |
| Checkpointing | `--gradient-checkpointing` + `--max-target-length 640` |
| Runtime | 100 k samples × 3 epochs ≈ 36–48 h |
| Monitoring | `tensorboard --logdir output/training_runs/<run>` |

## Epoch Checklist

1. **Dry run** on 500–1 000 samples (`--max-train-samples 1000`) to verify data + VRAM headroom.
2. **Full training** with TensorBoard + `output/training_runs/<date>/trainer_state.json`. Capture git commit, dataset manifest hashes, and CLI flags in a `run.yaml`.
3. **Evaluation**: `trainer.eval_results.json` already mirrors character/token metrics. For deeper dives, run `python scripts/experiments/benchmark_eval.py ...` on the model outputs to compare against structural manifolds.
4. **Compression sanity checks**: replay `scripts/experiments/run_optical_inference.py` using the new checkpoints to ensure compression × accuracy stays within Fox/OmniDoc targets.
5. **Promotion**: copy the best checkpoint (based on `--metric-for-best-model`, default character accuracy) to `external/checkpoints/<name>` for downstream inference demos.

## Demo & Deployment Hooks

- **Web demo**: wrap `scripts/training/ocr_trainer.py` checkpoints with a Streamlit/Gradio UI (upload PDF → run `scripts/experiments/run_optical_inference.py` → visualize compression + F1). Host on Hugging Face Spaces.
- **CLI/API**: expose a `scripts/tools/ocr_eval.py` that accepts arbitrary manifests and reports Fox/OmniDoc-style metrics to integrate with enterprise workflows (invoice/contract parsing).
- **Leaderboard**: publish aggregated `summary.json` from both manifold and optical runs under `docs/leaderboard/README.md`, inviting community submissions via pull requests.
- **Ops & Ethics**: use the per-language/subset metrics to flag regressions on noisy scans; log bias reports and failure modes in `docs/whitepaper/limitations.md`.

Follow this playbook to iterate quickly: keep manifests versioned, record every command, and feed each trained adapter back through the structural manifold benchmarks so the compression-vs-accuracy story stays tight across prototyping → production.
