#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Usage:
#   bash tools/dataset_relaxation.sh <MODEL_DIR> <CFG_DIR>
#
# Optional env vars:
#   GPU=0
#   RELAX_MODE=lbfgs
#   STEPS=5
#   STEP_SIZE=1
#   START=-1
#   END=-1
#   RETURN_STABLE=0|1
#   RETURN_UNRELAX=0|1

MODEL_DIR="${1:-}"
CFG_DIR="${2:-}"
GPU="${GPU:-0}"

if [[ -z "${MODEL_DIR}" || -z "${CFG_DIR}" ]]; then
  echo "MODEL_DIR and CFG_DIR are required." >&2
  echo "Example: bash tools/dataset_relaxation.sh /path/to/model_dir /path/to/cfg_dir" >&2
  exit 2
fi

RELAX_MODE="${RELAX_MODE:-lbfgs}"
STEPS="${STEPS:-5}"
STEP_SIZE="${STEP_SIZE:-1.0}"
START="${START:- -1}"
END="${END:- -1}"
RETURN_STABLE="${RETURN_STABLE:-0}"
RETURN_UNRELAX="${RETURN_UNRELAX:-0}"

label="_relax"
if (( RETURN_STABLE && RETURN_UNRELAX )); then
  label="_full"
elif (( RETURN_STABLE )); then
  label="_stable_relax"
elif (( RETURN_UNRELAX )); then
  label="_relax_unrelax"
fi

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/run/relax_dataset.py \
  --model_dir="${MODEL_DIR}" --cfg_dir="${CFG_DIR}" --label="${label}" \
  --start="${START}" --end="${END}" \
  --step_size="${STEP_SIZE}" --num_steps="${STEPS}" --threshold=0.5 \
  --relax_mode="${RELAX_MODE}" \
  --return_stable="${RETURN_STABLE}" --return_unrelax="${RETURN_UNRELAX}"
