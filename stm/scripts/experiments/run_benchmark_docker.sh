#!/usr/bin/env bash
set -euo pipefail

IMAGE="${CUDA_IMAGE:-nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04}"
DATASET="${DATASET:-wikitext=data/raw_text/wikitext_train.jsonl}"
JSON_KEY="${JSON_KEY:-text}"
WINDOW="${WINDOW:-512}"
STRIDE="${STRIDE:-384}"
PRECISION="${PRECISION:-3}"
OUTDIR="${OUTDIR:-output/benchmark_runs/wikitext_custom}"
PYBIN="${PYBIN:-python3}"
SCRIPT="${SCRIPT:-${PYBIN} scripts/experiments/benchmark_eval.py}"
HOST_SCORE="${HOST_SCORE:-$(cd .. && pwd)/score}"
MOUNT_SCORE=""
SCORE_INSTALL_CMD="true"
if [ -d "${HOST_SCORE}" ]; then
  MOUNT_SCORE="-v ${HOST_SCORE}:/score"
  SCORE_INSTALL_CMD="pip3 install /score >/dev/null"
fi
HOST_DATA="${HOST_DATA:-$(cd .. && pwd)/data}"
MOUNT_DATA=""
if [ -d "${HOST_DATA}" ]; then
  MOUNT_DATA="-v ${HOST_DATA}:/data"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[docker] docker CLI not found. Install Docker before running this script." >&2
  exit 1
fi

CMD=$(cat <<EOF
set -euo pipefail
apt-get update >/dev/null && apt-get install -y --no-install-recommends build-essential python3 python3-pip python3-dev nlohmann-json3-dev libtbb-dev >/dev/null
pip3 install --no-cache-dir -r requirements.txt >/dev/null
${SCORE_INSTALL_CMD}
CCBIN=g++ make native
CUDA_VISIBLE_DEVICES=0 ${SCRIPT} \
  --dataset ${DATASET} \
  --json-text-key ${JSON_KEY} \
  --window-bytes ${WINDOW} \
  --stride-bytes ${STRIDE} \
  --precision ${PRECISION} \
  --output-dir ${OUTDIR} \
  --use-native
EOF
)

docker run --rm --gpus all -v "${PWD}":/workspace ${MOUNT_SCORE} ${MOUNT_DATA} -w /workspace "${IMAGE}" bash -lc "${CMD}"
