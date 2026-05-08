import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts._bootstrap import set_default_env

set_default_env()

from scripts.data.csv_to_cache import convert_one_csv


def _iter_cif_files(cif_dir: Path) -> List[Path]:
    cifs = sorted([p for p in cif_dir.iterdir() if p.is_file() and p.suffix.lower() == ".cif"])
    if not cifs:
        raise FileNotFoundError(f"No .cif files found under: {cif_dir}")
    return cifs


def _load_prop_mapping(prop_txt: Path) -> Tuple[str, Dict[str, float]]:
    """Load mapping from lines like: `xxx.cif 1.23`.

    Keys are CIF filenames (including `.cif`). The property name is inferred
    from the txt filename stem.
    """
    prop_name = prop_txt.stem
    mapping: Dict[str, float] = {}

    for ln in prop_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue

        parts = [p for p in s.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid line in {prop_txt}: {ln!r}. Expected exactly 2 fields: `<filename.cif> <value>`."
            )

        fname, val_s = parts
        if not fname.lower().endswith(".cif"):
            raise ValueError(
                f"Invalid CIF filename in {prop_txt}: {fname!r}. Expected something like `xxx.cif`."
            )

        try:
            val = float(val_s)
        except Exception as e:
            raise ValueError(f"Invalid numeric value in {prop_txt}: {val_s!r}") from e

        mapping[fname] = val

    if not mapping:
        raise ValueError(f"No valid lines found in {prop_txt}")

    return prop_name, mapping


def _write_csv(
    *,
    cif_paths: List[Path],
    out_csv: Path,
    prop_name: str,
    prop_values: Optional[Dict[str, float]],
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["material_id", "cif", prop_name])

        for p in cif_paths:
            material_id = p.stem
            cif_str = p.read_text(encoding="utf-8", errors="ignore")
            if prop_values is None:
                val = 0.0
            else:
                val = prop_values[p.name]
            writer.writerow([material_id, cif_str, val])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a custom dataset from a CIF directory by generating a CSV and then "
            "calling scripts.data.csv_to_cache to create a *_ori.pt cache."
        )
    )

    parser.add_argument(
        "--cif_dir",
        type=str,
        required=True,
        help="Directory containing CIF files named as `material_id.cif`.",
    )
    parser.add_argument(
        "--prop_txt",
        type=str,
        default=None,
        help=(
            "Optional mapping txt. Each line must be: `<filename.cif> <value>`. "
            "The property column name is inferred from the txt filename stem."
        ),
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Path to the generated CSV. Default: <cif_dir>/custom.csv",
    )
    parser.add_argument(
        "--out_pt",
        type=str,
        default=None,
        help="Path to the generated cache .pt. Default: <out_csv_stem>_ori.pt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing out_csv/out_pt if they exist.",
    )

    # Keep preprocessing args consistent with `csv_to_cache.py`.
    parser.add_argument("--niggli", default=True, type=bool)
    parser.add_argument("--primitive", default=False, type=bool)
    parser.add_argument("--graph-method", default="crystalnn", type=str)
    parser.add_argument("--num_workers", default=30, type=int)

    args = parser.parse_args()

    cif_dir = Path(args.cif_dir).expanduser().resolve()
    cif_paths = _iter_cif_files(cif_dir)

    out_csv = (
        Path(args.out_csv).expanduser().resolve()
        if args.out_csv
        else (cif_dir / "custom.csv")
    )
    out_pt = (
        Path(args.out_pt).expanduser().resolve()
        if args.out_pt
        else Path(str(out_csv.with_suffix("")) + "_ori.pt")
    )

    if (out_csv.exists() or out_pt.exists()) and (not args.overwrite):
        raise FileExistsError(
            f"Output exists: csv={out_csv.exists()} pt={out_pt.exists()}. Use --overwrite to overwrite."
        )

    if args.prop_txt:
        prop_txt = Path(args.prop_txt).expanduser().resolve()
        prop_name, mapping = _load_prop_mapping(prop_txt)

        cif_names = {p.name for p in cif_paths}
        map_names = set(mapping.keys())
        missing = sorted(cif_names - map_names)
        extra = sorted(map_names - cif_names)
        if missing or extra:
            msg = ["prop_txt does not match cif_dir:"]
            if missing:
                msg.append(f"- missing {len(missing)} entries, e.g. {missing[:5]}")
            if extra:
                msg.append(f"- extra {len(extra)} entries, e.g. {extra[:5]}")
            raise ValueError("\n".join(msg))
        prop_values = mapping
    else:
        prop_name = "ehull"
        prop_values = None

    _write_csv(
        cif_paths=cif_paths,
        out_csv=out_csv,
        prop_name=prop_name,
        prop_values=prop_values,
    )

    convert_one_csv(
        str(out_csv),
        str(out_pt),
        niggli=args.niggli,
        primitive=args.primitive,
        graph_method=args.graph_method,
        prop=prop_name,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
