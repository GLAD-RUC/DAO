#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Usage:
#   bash scripts/run/structure_generation.sh <DATA> <MODEL_DIR> [ENERGY_MODEL_CKPT]
#
# Example:
#   bash tools/structure_generation.sh mp_20 /path/to/finetune_mp20 /path/to/DAO_P.ckpt

DATA="${1:-mp_20}"
MODEL_PATH="${2:-}"
ENERGY_MODEL_PATH="${3:-}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required (the finetune output directory)."
  echo "Example: bash tools/structure_generation.sh mp_20 /path/to/finetune_mp20 /path/to/DAO_P.ckpt"
  exit 2
fi

NUM_EVALS="${NUM_EVALS:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
BASE_GPU="${BASE_GPU:-0}"
TOTAL_BATCHES="${TOTAL_BATCHES:-}"

extra=()
if [[ -n "${ENERGY_MODEL_PATH}" ]]; then
  extra+=(--energy-model-path "${ENERGY_MODEL_PATH}" --energy-guidance)
fi
if [[ -n "${TOTAL_BATCHES}" ]]; then
  extra+=(--total-batches "${TOTAL_BATCHES}")
fi

python -m dao csp generate \
  --dataset "${DATA}" \
  --model-path "${MODEL_PATH}" \
  --num-evals "${NUM_EVALS}" \
  --num-gpus "${NUM_GPUS}" \
  --base-gpu "${BASE_GPU}" \
  "${extra[@]}"
