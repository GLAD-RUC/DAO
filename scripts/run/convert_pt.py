"""Convert eval_diff_*.pt generation outputs to CIF or POSCAR files.

Usage:
    python -m scripts.run.convert_pt eval_diff_1_all.pt --format cif
    python -m scripts.run.convert_pt eval_diff_1_all.pt --format poscar
    python -m scripts.run.convert_pt eval_diff_1_all.pt --format cif --eval-idx 0
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from scripts._bootstrap import set_default_env

set_default_env()

from scripts.eval.eval_utils import get_crystals_list


def _write_one(
    structure: Structure,
    out_path: Path,
    fmt: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "poscar":
        structure.to(filename=str(out_path))
    else:
        structure.to(filename=str(out_path))


def convert_pt(
    pt_path: str,
    *,
    out_dir: str,
    fmt: str,
    eval_idx: int,
    labels: List[str],
) -> None:
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(pt_path, map_location="cpu")

    frac_coords = data["frac_coords"]  # (num_evals, total_atoms, 3)
    atom_types = data["atom_types"]
    lengths = data["lengths"]
    angles = data["angles"]
    num_atoms = data["num_atoms"]

    n_evals = frac_coords.size(0)
    if eval_idx < 0:
        eval_range = range(n_evals)
    else:
        if eval_idx >= n_evals:
            raise ValueError(f"--eval-idx={eval_idx} but file has {n_evals} evals")
        eval_range = [eval_idx]

    out_root = Path(out_dir)
    ext = "cif" if fmt == "cif" else "vasp"

    for ei in eval_range:
        crystals = get_crystals_list(
            frac_coords[ei], atom_types[ei], lengths[ei], angles[ei], num_atoms[ei]
        )
        for i, crys in enumerate(crystals):
            label = labels[i] if i < len(labels) else f"struct_{i}"
            fname = f"{label}_eval{ei}.{ext}" if n_evals > 1 else f"{label}.{ext}"
            try:
                structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(crys["lengths"].tolist() + crys["angles"].tolist())
                    ),
                    species=[int(z) for z in crys["atom_types"].tolist()],
                    coords=crys["frac_coords"],
                    coords_are_cartesian=False,
                )
                _write_one(structure, out_root / fname, fmt)
            except Exception as e:
                print(f"[warn] {fname}: {e}")

    print(f"Wrote {len(list(out_root.glob(f'*.{ext}')))} {ext} files to {out_root}")


def _infer_labels(data: dict, pt_path: str) -> List[str]:
    """Try to recover per-crystal labels from the pt dict."""
    # generate_from_formula stores this
    if "formulas" in data:
        return list(data["formulas"])

    # For benchmark generation, use material_id from input_data_batch if present
    batch = data.get("input_data_batch")
    if batch is not None and hasattr(batch, "material_id"):
        return [str(mid) for mid in batch.material_id]

    # Fallback: use file stem + index
    n = data["num_atoms"].size(-1) if data["num_atoms"].dim() > 1 else len(data["num_atoms"])
    stem = Path(pt_path).stem.replace("eval_diff_", "")
    return [f"{stem}_{i}" for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt_file", help="Path to eval_diff_*.pt file")
    parser.add_argument(
        "--format",
        choices=["cif", "poscar"],
        default="cif",
        help="Output format (default: cif). POSCAR files use .vasp extension.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: <pt_file_dir>/<pt_stem>_structures/)",
    )
    parser.add_argument(
        "--eval-idx",
        type=int,
        default=-1,
        help="Which eval index to convert (-1 = all, default: all)",
    )
    args = parser.parse_args()

    pt_path = Path(args.pt_file).expanduser().resolve()
    if not pt_path.exists():
        raise SystemExit(f"File not found: {pt_path}")

    out_dir = args.out_dir or str(pt_path.parent / (pt_path.stem + "_structures"))

    try:
        data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(str(pt_path), map_location="cpu")
    labels = _infer_labels(data, str(pt_path))

    convert_pt(
        str(pt_path),
        out_dir=out_dir,
        fmt=args.format,
        eval_idx=args.eval_idx,
        labels=labels,
    )


if __name__ == "__main__":
    main()
