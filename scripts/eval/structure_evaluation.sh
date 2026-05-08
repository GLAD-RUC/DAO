#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Usage:
#   bash scripts/eval/structure_evaluation.sh <DATA> <MODEL_DIR>
#
# Optional env vars:
#   NUM_EVALS=1|20
#   LABEL=...

DATA="${1:-mp_20}"
MODEL_PATH="${2:-}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required (the directory containing generated results)."
  echo "Example: bash tools/structure_evaluation.sh mp_20 /path/to/finetune_mp20"
  exit 2
fi

NUM_EVALS="${NUM_EVALS:-1}"
LABEL="${LABEL:-${NUM_EVALS}_all}"

extra=()
if [[ "${NUM_EVALS}" != "1" ]]; then
  extra+=(--multi-eval)
fi

python -m dao csp evaluate \
  --dataset "${DATA}" \
  --root-path "${MODEL_PATH}" \
  --num-evals "${NUM_EVALS}" \
  --label "${LABEL}" \
  "${extra[@]}"
