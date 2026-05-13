"""Predict properties directly from CIF files.

This is a convenience wrapper that:
  1. Converts CIF files to a cached .pt dataset (via scripts.data.prepare_custom_data),
  2. Runs DAO-P inference (via scripts.infer.inference_prop).

Usage:
    python -m scripts.infer.inference_prop_from_cif \
        --cif-dir my_cifs/ \
        --model-path ckpts/dao_p/last.ckpt \
        --pred-energy
"""

import argparse
import os
import tempfile
from pathlib import Path

from scripts._bootstrap import set_default_env

set_default_env()

from scripts.data.prepare_custom_data import _iter_cif_files, _write_csv
from scripts.data.csv_to_cache import convert_one_csv
from scripts.infer.inference_prop import run_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cif-dir",
        required=True,
        help="Directory containing CIF files (e.g. my_cifs/).",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="DAO-P checkpoint file (e.g. ckpts/dao_p/last.ckpt).",
    )
    parser.add_argument(
        "--prop",
        default="ehull",
        help="Property name (default: ehull). Ignored when --pred-energy is set.",
    )
    parser.add_argument(
        "--pred-energy",
        action="store_true",
        help="Predict energy using pretrained DAO-P (energy head).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Inference batch size (default: 100).",
    )
    parser.add_argument(
        "--out-npy",
        default="",
        help="Where to save predictions (.npy). Default: <cif-dir>/pred_<prop>_of_custom.npy",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=30,
        help="Workers for CIF -> graph preprocessing (default: 30).",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep intermediate CSV and .pt cache files (default: delete after inference).",
    )
    args = parser.parse_args()

    cif_dir = Path(args.cif_dir).expanduser().resolve()
    if not cif_dir.is_dir():
        raise SystemExit(f"Not a directory: {cif_dir}")

    cif_paths = _iter_cif_files(cif_dir)
    print(f"Found {len(cif_paths)} CIF file(s) in {cif_dir}")

    # Use a temp dir so intermediate files are cleaned up automatically.
    work_dir = cif_dir if args.keep_cache else None
    tmp = tempfile.TemporaryDirectory(dir=work_dir, prefix="_cif_infer_") if not args.keep_cache else None
    base = Path(tmp.name) if tmp else cif_dir

    out_csv = base / "custom.csv"
    out_pt = base / "custom_ori.pt"

    # Step 1: CIF -> CSV -> cached .pt
    prop_name = "ehull" if args.pred_energy else args.prop
    _write_csv(cif_paths=cif_paths, out_csv=out_csv, prop_name=prop_name, prop_values=None)
    print(f"Wrote CSV to {out_csv}")

    convert_one_csv(
        str(out_csv),
        str(out_pt),
        niggli=True,
        primitive=False,
        graph_method="crystalnn",
        prop=prop_name,
        num_workers=args.num_workers,
    )
    print(f"Wrote cache to {out_pt}")

    # Step 2: Run inference
    out_npy = args.out_npy or str(cif_dir / f"pred_{prop_name}_of_custom.npy")

    run_inference(
        mode="dataset",
        ori_path=str(out_pt),
        model_path=args.model_path,
        prop=prop_name,
        pred_energy=args.pred_energy,
        batch_size=args.batch_size,
        out_npy=out_npy,
    )

    # Cleanup
    if tmp is not None:
        tmp.cleanup()
        print("Cleaned up intermediate files")


if __name__ == "__main__":
    main()
