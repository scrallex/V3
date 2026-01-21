# Structured Demo (News Briefs)

This folder contains a tiny JSONL corpus (`news_sample.jsonl`) that mimics the short, structured paragraphs we use when validating manifold compression on PDF-derived or briefing-style documents. Each record keeps an `id` plus the raw `text` field so it can be passed directly to `benchmark_eval.py`.

## Quick run

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

Inspect `output/benchmark_runs/news_demo/briefs.json` to see byte/token compression, verification precision, and reconstructed snippets. Use the same `briefs.json` file as input to a reconstruction script or the upcoming Hugging Face Space.

## Notebook

`mini_pipeline.ipynb` walks through the same commands in a linear, copy/paste friendly format so you can attach the outputs to issues, blog posts, or tutorials.
