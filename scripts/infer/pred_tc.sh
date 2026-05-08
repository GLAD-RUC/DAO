#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Usage:
#   bash scripts/infer/pred_tc.sh <ORI_PT_PATH> <MODEL_CKPT_PATTERN>
#
# Example:
#   bash tools/pred_tc.sh data/super_conductors/real_world/output_ori.pt \
#     "/path/to/finetuned_DAO_P_{fold}.ckpt"

ORI_PATH="${1:-}"
PATTERN="${2:-}"

if [[ -z "${ORI_PATH}" || -z "${PATTERN}" ]]; then
  echo "ORI_PATH and MODEL_CKPT_PATTERN are required." >&2
  exit 2
fi

for fold in 0 1 2 3 4; do
  model_path="${PATTERN//\{fold\}/${fold}}"
  echo "fold=${fold} model=${model_path}"
  python -m dao prop predict-dataset --ori-path "${ORI_PATH}" --model-path "${model_path}" --prop logtc
done
