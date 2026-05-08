"""Bootstrap helpers for running scripts from anywhere.

We avoid relying on working directory or ad-hoc sys.path edits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    # DAO/scripts/_bootstrap.py -> DAO
    return Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> Path:
    root = repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def set_default_env() -> None:
    root = ensure_repo_on_path()
    os.environ.setdefault("PROJECT_ROOT", str(root))
    os.environ.setdefault("HYDRA_JOBS", str(root / "outputs" / "hydra"))
    os.environ.setdefault("WANDB_DIR", str(root / "outputs" / "wandb"))

