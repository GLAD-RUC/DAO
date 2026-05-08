#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Usage:
#   bash scripts/run/finetune_csp.sh <DATA> <PRETRAIN_CKPT>
#
# Optional env vars:
#   EPOCHS=1000
#   LR=2e-5
#   DECAY=1e-5
#   GPUS=1

DATA="${1:-mp_20}"
CKPT="${2:-}"

if [[ -z "${CKPT}" ]]; then
  echo "PRETRAIN_CKPT is required (e.g. DAO_G_Stage2.ckpt)."
  echo "Example: bash tools/finetune_csp.sh mp_20 /path/to/DAO_G_Stage2.ckpt"
  exit 2
fi

EPOCHS="${EPOCHS:-1000}"
LR="${LR:-0.00002}"
DECAY="${DECAY:-0.00001}"
GPUS="${GPUS:-1}"

python -m dao csp finetune \
  --dataset "${DATA}" \
  --pretrain-ckpt "${CKPT}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${DECAY}" \
  --gpus "${GPUS}"
