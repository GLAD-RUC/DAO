#!/usr/bin/env bash
# Set up the `dao` conda environment.
#
#   conda:  Python 3.8.18 + PyTorch 1.10.0 + cudatoolkit 11.3
#   pip:    everything else (pinned in pyproject.toml)
#
# Usage:
#   bash setup.sh
#   conda activate dao

set -euo pipefail

ENV_NAME="${ENV_NAME:-dao}"

echo "==> Creating conda env: $ENV_NAME"
conda create -y -n "$ENV_NAME" python=3.8.18

echo "==> Installing PyTorch + CUDA runtime via conda"
conda install -y -n "$ENV_NAME" \
    pytorch==1.10.0 cudatoolkit=11.3 \
    -c pytorch -c conda-forge

echo "==> Installing Python dependencies from pyproject.toml"
conda run -n "$ENV_NAME" --no-capture-output python -m pip install -U pip
conda run -n "$ENV_NAME" --no-capture-output python -m pip install -e .

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
