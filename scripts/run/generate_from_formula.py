"""Generate crystal structures from user-provided chemical formulas.

Two input modes are supported (the core generation code is shared):
  * Single formula:  --formula "Li2FeO4"
  * Batch file:      --input_file formulas.txt   (one formula per line;
                      optional second column for num_atoms override)

Unlike `scripts/run/generate.py`, this script does NOT require a benchmark
dataset (mp_20 / mpts_52 / ...): batches are built directly from formulas.
"""

import argparse
import os
import os.path as osp
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from torch_geometric.data import Batch, Data, DataLoader

from scripts._bootstrap import set_default_env

set_default_env()

from dao.common.data_utils import chemical_symbols
from dao.pl_modules.PTModels import CrystGenerativePretrainModel
from scripts.eval.eval_utils import (
    get_crystals_list,
    lattices_to_params_shape,
    load_model,
    recommand_step_lr,
)


_SYMBOL_TO_Z = {s: i for i, s in enumerate(chemical_symbols)}


def parse_formula(formula: str, num_atoms: Optional[int] = None) -> Tuple[str, np.ndarray]:
    """Expand a chemical formula into a flat atomic-number array.

    If `num_atoms` is given, the formula is scaled so the total atom count
    equals `num_atoms` (must be divisible by the reduced formula unit count).
    Otherwise the raw formula (rounded to int counts) is used directly.
    """
    comp = Composition(formula)
    reduced, factor = comp.get_reduced_composition_and_factor()

    if num_atoms is not None:
        unit_size = int(sum(reduced.values()))
        if num_atoms % unit_size != 0:
            raise ValueError(
                f"num_atoms={num_atoms} is not a multiple of reduced formula size "
                f"{unit_size} for {formula}"
            )
        scale = num_atoms // unit_size
        counts = {el: int(reduced[el]) * scale for el in reduced}
    else:
        counts = {el: int(round(comp[el])) for el in comp}
        if sum(counts.values()) == 0:
            raise ValueError(f"Empty composition after parsing {formula!r}")

    atom_types: List[int] = []
    for el, n in counts.items():
        z = _SYMBOL_TO_Z.get(el.symbol)
        if z is None:
            raise ValueError(f"Unknown element {el.symbol!r} in {formula!r}")
        atom_types.extend([z] * n)

    return comp.reduced_formula, np.asarray(atom_types, dtype=np.int64)


def _load_inputs(args: argparse.Namespace) -> List[Tuple[str, Optional[int]]]:
    """Read (formula, num_atoms_override) entries from CLI or input file."""
    entries: List[Tuple[str, Optional[int]]] = []

    if args.formula:
        entries.append((args.formula, args.num_atoms if args.num_atoms > 0 else None))

    if args.input_file:
        path = Path(args.input_file).expanduser().resolve()
        for ln in path.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p for p in s.replace(",", " ").split() if p]
            if len(parts) == 1:
                entries.append((parts[0], None))
            elif len(parts) == 2:
                entries.append((parts[0], int(parts[1])))
            else:
                raise ValueError(
                    f"Invalid line in {path}: {ln!r}. "
                    "Expected `<formula>` or `<formula> <num_atoms>`."
                )

    if not entries:
        raise SystemExit("No input: provide --formula and/or --input_file.")
    return entries


def build_dataloader(
    entries: List[Tuple[str, Optional[int]]], batch_size: int
) -> Tuple[DataLoader, List[str]]:
    """Build a torch_geometric DataLoader with one graph per formula entry."""
    data_list: List[Data] = []
    labels: List[str] = []
    for formula, n_override in entries:
        label, atom_types = parse_formula(formula, n_override)
        num_atoms = int(atom_types.shape[0])
        data = Data(
            atom_types=torch.from_numpy(atom_types),
            num_atoms=torch.tensor([num_atoms], dtype=torch.long),
            num_nodes=num_atoms,
            # Dummy tensors: sample() overwrites coords/lattices from noise.
            frac_coords=torch.zeros(num_atoms, 3),
            lengths=torch.ones(1, 3),
            angles=torch.full((1, 3), 90.0),
        )
        data_list.append(data)
        labels.append(label)

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    return loader, labels


def sample_batches(
    loader: DataLoader,
    model,
    *,
    num_evals: int,
    energy_model,
    step_lr: float,
    energy_guidance: bool,
    aug: float,
):
    frac_coords_all, num_atoms_all, atom_types_all, lattices_all = [], [], [], []
    input_data_list = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch = batch.cuda()

        per_eval_frac, per_eval_num, per_eval_type, per_eval_latt = [], [], [], []
        for eval_idx in range(num_evals):
            print(f"batch {idx + 1}/{len(loader)} sample {eval_idx + 1}/{num_evals}")
            outputs, _ = model.sample(
                batch,
                energy_model=energy_model,
                step_lr=step_lr,
                energy_guidance=energy_guidance,
                aug=aug,
            )
            per_eval_frac.append(outputs["frac_coords"].detach().cpu())
            per_eval_num.append(outputs["num_atoms"].detach().cpu())
            per_eval_type.append(outputs["atom_types"].detach().cpu())
            per_eval_latt.append(outputs["lattices"].detach().cpu())

        frac_coords_all.append(torch.stack(per_eval_frac, dim=0))
        num_atoms_all.append(torch.stack(per_eval_num, dim=0))
        atom_types_all.append(torch.stack(per_eval_type, dim=0))
        lattices_all.append(torch.stack(per_eval_latt, dim=0))
        input_data_list.extend(batch.to_data_list())

    frac_coords = torch.cat(frac_coords_all, dim=1)
    num_atoms = torch.cat(num_atoms_all, dim=1)
    atom_types = torch.cat(atom_types_all, dim=1)
    lattices = torch.cat(lattices_all, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)
    return frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch


def save_cifs(
    out_dir: Path,
    labels: List[str],
    frac_coords: torch.Tensor,
    atom_types: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    num_atoms: torch.Tensor,
) -> None:
    """Write one CIF per (formula, eval) pair under `out_dir/cifs/`."""
    out_dir = Path(out_dir)
    cif_dir = out_dir / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)

    n_eval = frac_coords.size(0)
    for eval_idx in range(n_eval):
        crystals = get_crystals_list(
            frac_coords[eval_idx],
            atom_types[eval_idx],
            lengths[eval_idx],
            angles[eval_idx],
            num_atoms[eval_idx],
        )
        for i, crys in enumerate(crystals):
            label = labels[i] if i < len(labels) else f"sample_{i}"
            try:
                structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(crys["lengths"].tolist() + crys["angles"].tolist())
                    ),
                    species=[int(z) for z in crys["atom_types"].tolist()],
                    coords=crys["frac_coords"],
                    coords_are_cartesian=False,
                )
                cif_path = cif_dir / f"{label}_eval{eval_idx}_{i}.cif"
                structure.to(filename=str(cif_path))
            except Exception as e:
                print(f"[warn] failed to write CIF for {label} (eval={eval_idx}, idx={i}): {e}")


def main(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    model, _, _ = load_model(model_path, load_data=False, from_scratch=False)
    if torch.cuda.is_available():
        model.to("cuda")

    energy_model = None
    if args.energy_model_path:
        print("Loading energy model ...")
        energy_model = CrystGenerativePretrainModel.load_from_checkpoint(args.energy_model_path)
        if torch.cuda.is_available():
            energy_model.to("cuda")

    entries = _load_inputs(args)
    print(f"Parsed {len(entries)} formula entries")
    loader, labels = build_dataloader(entries, batch_size=args.batch_size)

    step_lr = (
        args.step_lr
        if args.step_lr >= 0
        else recommand_step_lr["csp" if args.num_evals == 1 else "csp_multi"]["mp_20"]
    )

    start_time = time.time()
    frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch = sample_batches(
        loader,
        model,
        num_evals=args.num_evals,
        energy_model=energy_model,
        step_lr=step_lr,
        energy_guidance=args.energy_guidance,
        aug=args.aug,
    )

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else model_path
    out_dir.mkdir(parents=True, exist_ok=True)

    label = args.label or f"formula_{args.num_evals}"
    out_pt = out_dir / f"eval_diff_{label}.pt"
    torch.save(
        {
            "eval_setting": args,
            "formulas": labels,
            "input_data_batch": input_data_batch,
            "frac_coords": frac_coords,
            "num_atoms": num_atoms,
            "atom_types": atom_types,
            "lattices": lattices,
            "lengths": lengths,
            "angles": angles,
            "time": time.time() - start_time,
        },
        out_pt,
    )
    print(f"Saved tensors to {out_pt}")

    if args.write_cifs:
        save_cifs(out_dir, labels, frac_coords, atom_types, lengths, angles, num_atoms)
        print(f"Wrote CIF files under {out_dir / 'cifs'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="DAO-G finetune output dir")
    parser.add_argument("--energy_model_path", default="", help="DAO-P checkpoint for guidance")
    parser.add_argument("--energy_guidance", action="store_true")

    parser.add_argument("--formula", default="", help="Single formula, e.g. 'Li2FeO4'")
    parser.add_argument(
        "--input_file",
        default="",
        help="Text file with one formula per line; optional second column sets num_atoms.",
    )
    parser.add_argument(
        "--num_atoms",
        type=int,
        default=0,
        help="Optional override for --formula: scale to this many atoms (must be a multiple of the reduced formula unit).",
    )

    parser.add_argument("--num_evals", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--step_lr", type=float, default=-1)
    parser.add_argument("--aug", type=float, default=1.0)

    parser.add_argument("--output_dir", default="", help="Where to write outputs (default: model_path)")
    parser.add_argument("--label", default="", help="Suffix for eval_diff_<label>.pt (default: formula_<num_evals>)")
    parser.add_argument("--write_cifs", action="store_true", help="Also write per-sample CIF files")

    args = parser.parse_args()
    if not args.formula and not args.input_file:
        parser.error("provide --formula and/or --input_file")
    main(args)
