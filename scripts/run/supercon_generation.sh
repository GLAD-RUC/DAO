#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Usage:
#   bash scripts/run/supercon_generation.sh <supercon_real|supercon_rest> <MODEL_DIR> [ENERGY_MODEL_CKPT]

DATA="${1:-supercon_real}"
MODEL_PATH="${2:-}"
ENERGY_MODEL_PATH="${3:-}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required (the DAO-G finetune output dir on Supercon3D)."
  exit 2
fi

NUM_EVALS="${NUM_EVALS:-1}"
GPU="${GPU:-0}"
LABEL="${LABEL:-${NUM_EVALS}_all}"
ENERGY_GUIDANCE="${ENERGY_GUIDANCE:-1}"
SAMPLE_AUG="${SAMPLE_AUG:-}"

extra=(--gpu "${GPU}")
if [[ -n "${ENERGY_MODEL_PATH}" ]]; then
  extra+=(--energy-model-path "${ENERGY_MODEL_PATH}")
fi
if [[ "${ENERGY_GUIDANCE}" == "1" ]]; then
  extra+=(--energy-guidance)
fi
if [[ -n "${SAMPLE_AUG}" ]]; then
  extra+=(--aug "${SAMPLE_AUG}")
fi

python -m dao supercon generate \
  --dataset "${DATA}" \
  --model-path "${MODEL_PATH}" \
  --num-evals "${NUM_EVALS}" \
  --label "${LABEL}" \
  "${extra[@]}"
