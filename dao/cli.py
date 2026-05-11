import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _repo_root() -> Path:
    # DAO/dao/cli.py -> DAO
    return Path(__file__).resolve().parents[1]


def _default_env(root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PROJECT_ROOT", str(root))
    env.setdefault("HYDRA_JOBS", str(root / "outputs" / "hydra"))
    env.setdefault("WANDB_DIR", str(root / "outputs" / "wandb"))
    return env


def _run(cmd: List[str], *, env: Dict[str, str], cwd: Path, check: bool = True) -> int:
    proc = subprocess.run(cmd, env=env, cwd=str(cwd))
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def _script_rel_to_module(script_rel: str, *, cwd: Path) -> Optional[str]:
    """Convert a repo-relative python file path to a module path.

    Example:
      scripts/run/generate.py -> scripts.run.generate
      dao/finetune.py -> dao.finetune

    Returns None if the path cannot be treated as an importable package module.
    """

    p = Path(script_rel)
    if p.is_absolute() or p.suffix != ".py":
        return None

    parts = p.with_suffix("").parts
    if len(parts) == 0:
        return None

    # Ensure every package directory has an __init__.py
    for i in range(len(parts) - 1):
        init_py = cwd.joinpath(*parts[: i + 1], "__init__.py")
        if not init_py.exists():
            return None

    return ".".join(parts)


def _run_python(script_rel: str, args: List[str], *, env: Dict[str, str], cwd: Path, check: bool = True) -> int:
    module = _script_rel_to_module(script_rel, cwd=cwd)
    if module is not None:
        # Prefer module execution so imports like `from scripts._bootstrap ...` work naturally.
        return _run([sys.executable, "-m", module, *args], env=env, cwd=cwd, check=check)

    script_path = cwd / script_rel
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")
    return _run([sys.executable, str(script_path), *args], env=env, cwd=cwd, check=check)


def _split_batches(total: int, parts: int) -> Iterable[Tuple[int, int, int]]:
    if parts <= 0:
        raise ValueError("parts must be > 0")
    base = total // parts
    rem = total % parts
    start = 0
    for i in range(parts):
        count = base + (1 if i < rem else 0)
        end = start + count
        yield i, start, end
        start = end


def _popen_python(script_rel: str, args: List[str], *, env: Dict[str, str], cwd: Path) -> subprocess.Popen:
    module = _script_rel_to_module(script_rel, cwd=cwd)
    if module is not None:
        return subprocess.Popen([sys.executable, "-m", module, *args], env=env, cwd=str(cwd))

    script_path = cwd / script_rel
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")
    return subprocess.Popen([sys.executable, str(script_path), *args], env=env, cwd=str(cwd))


def _combine_eval_diff_shards(output_dir: Path, *, num_evals: int, num_shards: int, suffix: str = "") -> Path:
    """Combine multi-GPU `eval_diff_{num_evals}{suffix}_{rank}.pt` shards.

    Logic is adapted from `/mnt/bn/atomistic-dev/users/liming/tmp/combine_eval_results.ipynb`.
    Returns the merged file path.
    """

    from copy import deepcopy
    import torch

    if num_shards <= 1:
        raise ValueError("num_shards must be > 1")

    # shard files: eval_diff_{num_evals}{suffix}_{rank}.pt
    shard_paths = [
        output_dir / f"eval_diff_{num_evals}{suffix}_{rank}.pt" for rank in range(num_shards)
    ]
    ## skip missing files
    shard_paths = [str(p) for p in shard_paths if p.exists()]
    if not shard_paths:
        raise FileNotFoundError("No shard files found.")

    demo = torch.load(shard_paths[0], map_location="cpu")
    res = deepcopy(demo)
    keys_to_cat = ["frac_coords", "atom_types", "num_atoms", "lengths", "angles"]

    for idx in range(1, len(shard_paths)):
        temp = torch.load(shard_paths[idx], map_location="cpu")
        for key in keys_to_cat:
            if key not in temp or key not in res:
                raise KeyError(f"Missing key={key} in shard={shard_paths[idx]}")
            res[key] = torch.cat((res[key], temp[key]), dim=1)

    # Best-effort: make eval_setting label consistent with merged file.
    merged_label = f"{num_evals}{suffix}_all"
    try:
        if isinstance(res.get("eval_setting", None), object) and hasattr(res["eval_setting"], "label"):
            res["eval_setting"].label = merged_label
    except Exception:
        pass

    merged_path = output_dir / f"eval_diff_{num_evals}{suffix}_all.pt"
    torch.save(res, merged_path)
    return merged_path


def _cleanup_files(paths: List[Path]) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except TypeError:
            # Python < 3.8 fallback
            if p.exists():
                p.unlink()


def cmd_doctor(_: argparse.Namespace) -> int:
    root = _repo_root()
    env = _default_env(root)
    print("PROJECT_ROOT=", env.get("PROJECT_ROOT"))
    print("HYDRA_JOBS=", env.get("HYDRA_JOBS"))
    print("WANDB_DIR=", env.get("WANDB_DIR"))
    for p in [root / "conf", root / "dao", root / "scripts"]:
        print(f"{'OK' if p.exists() else 'MISSING'}  {p}")
    for p in [root / "data", root / "ckpts"]:
        print(f"{'OK' if p.exists() else 'NOTE'}  {p}")
    return 0


def cmd_csp_finetune(args: argparse.Namespace) -> int:
    root = _repo_root()
    env = _default_env(root)

    overrides: List[str] = [
        f"data={args.dataset}",
        "model=finetune",
        f"expname=finetune_{args.dataset}",
        f"model.pretrain_repr={args.pretrain_ckpt}",
        f"data.train_max_epochs={args.epochs}",
        f"optim.lr_scheduler_cos.T_0={args.epochs}",
        f"optim.optimizer.lr={args.lr}",
        f"optim.optimizer.weight_decay={args.weight_decay}",
        f"train.finetune_mode=gen",
        f"train.pl_trainer.gpus={args.gpus}",
    ]
    overrides.extend(args.overrides or [])
    return _run_python("dao/finetune.py", overrides, env=env, cwd=root)


def cmd_csp_generate(args: argparse.Namespace) -> int:
    root = _repo_root()
    base_env = _default_env(root)

    procs: List[subprocess.Popen] = []
    for rank in range(args.num_gpus):
        gpu_id = args.base_gpu + rank
        label = f"{args.num_evals}_all" if args.num_gpus == 1 else f"{args.num_evals}_{rank}"

        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        py_args = [
            "--dataset",
            args.dataset,
            "--num_evals",
            str(args.num_evals),
            "--label",
            label,
            "--model_path",
            args.model_path,
            "--rank",
            str(rank),
            "--world_size",
            str(args.num_gpus),
            "--batch_size",
            str(args.batch_size),
        ]
        if args.energy_guidance:
            py_args.append("--energy_guidance")
        if args.energy_model_path:
            py_args += ["--energy_model_path", args.energy_model_path]

        procs.append(_popen_python("scripts/run/generate.py", py_args, env=env, cwd=root))

    rc = 0
    for p in procs:
        r = p.wait()
        rc = rc or r
    if rc != 0:
        raise SystemExit(rc)

    # Auto-combine multi-GPU shards into `*_all.pt`, then delete shard files.
    if args.num_gpus > 1:
        out_dir = Path(args.model_path).expanduser().resolve()
        merged = _combine_eval_diff_shards(out_dir, num_evals=args.num_evals, num_shards=args.num_gpus)
        shard_paths = [out_dir / f"eval_diff_{args.num_evals}_{rank}.pt" for rank in range(args.num_gpus)]
        _cleanup_files(shard_paths)
        print(f"Merged shards -> {merged}")
    return 0


def cmd_csp_evaluate(args: argparse.Namespace) -> int:
    root = _repo_root()
    env = _default_env(root)
    label = args.label
    if not label:
        label = f"{args.num_evals}_all"

    py_args = [
        "--root_path",
        args.root_path,
        "--tasks",
        "csp",
        "--label",
        label,
        "--gt_file",
        str(root / "data" / args.dataset / "test.csv"),
    ]
    if args.multi_eval:
        py_args.append("--multi_eval")
    return _run_python("scripts/eval/evaluate_gen.py", py_args, env=env, cwd=root)


def cmd_supercon_generate(args: argparse.Namespace) -> int:
    root = _repo_root()
    env = _default_env(root)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    label = args.label or f"{args.num_evals}_all"
    py_args = [
        "--dataset",
        args.dataset,
        "--num_evals",
        str(args.num_evals),
        "--label",
        label,
        "--model_path",
        args.model_path,
    ]
    if args.energy_guidance:
        py_args.append("--energy_guidance")
    if args.energy_model_path:
        py_args += ["--energy_model_path", args.energy_model_path]
    if args.aug is not None:
        py_args += ["--aug", str(args.aug)]

    return _run_python("scripts/run/generate_supercon.py", py_args, env=env, cwd=root)


def cmd_prop_predict(args: argparse.Namespace) -> int:
    root = _repo_root()
    env = _default_env(root)

    if args.mode == "generated" and not args.eval_path:
        raise SystemExit("--eval-path is required when --mode=generated")

    py_args = [
        "--mode",
        args.mode,
        "--ori_path",
        args.ori_path,
        "--model_path",
        args.model_path,
        "--prop",
        args.prop,
    ]

    if args.mode == "generated":
        py_args += ["--eval_path", args.eval_path, "--sample_size", str(args.sample_size)]

    if args.pred_energy:
        py_args.append("--pred_energy")
    if getattr(args, "out_npy", ""):
        py_args += ["--out_npy", args.out_npy]
    return _run_python("scripts/infer/inference_prop.py", py_args, env=env, cwd=root)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dao",
        description="DAO CLI entrypoint (training / sampling / evaluation).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_doctor = sub.add_parser("doctor", help="Check repo layout and default environment variables")
    p_doctor.set_defaults(func=cmd_doctor)

    # CSP
    p_csp = sub.add_parser("csp", help="Crystal Structure Prediction commands")
    csp_sub = p_csp.add_subparsers(dest="csp_cmd", required=True)

    p_ft = csp_sub.add_parser("finetune", help="Finetune DAO-G for CSP")
    p_ft.add_argument("--dataset", required=True, help="e.g. mp_20 / mpts_52 / supercon3d_gen")
    p_ft.add_argument("--pretrain-ckpt", required=True, help="Path to DAO-G pretrained checkpoint")
    p_ft.add_argument("--epochs", type=int, default=1000)
    p_ft.add_argument("--lr", type=float, default=2e-5)
    p_ft.add_argument("--weight-decay", type=float, default=1e-5)
    p_ft.add_argument("--gpus", type=int, default=1)
    p_ft.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Extra Hydra overrides (passed through to dao/finetune.py)",
    )
    p_ft.set_defaults(func=cmd_csp_finetune)

    p_gen = csp_sub.add_parser("generate", help="Generate structures (supports multi-GPU sharding)")
    p_gen.add_argument("--dataset", required=True, help="e.g. mp_20 / mpts_52")
    p_gen.add_argument("--model-path", required=True, help="Finetune output directory (contains ckpt and scalers)")
    p_gen.add_argument("--energy-model-path", default="", help="DAO-P checkpoint path (for energy guidance)")
    p_gen.add_argument("--energy-guidance", action="store_true")
    p_gen.add_argument("--num-evals", type=int, default=1)
    p_gen.add_argument("--num-gpus", type=int, default=1)
    p_gen.add_argument("--base-gpu", type=int, default=0)
    p_gen.add_argument("--batch-size", type=int, default=-1, help="Test batch size; -1 means use hparams default")
    p_gen.set_defaults(func=cmd_csp_generate)

    p_eval = csp_sub.add_parser("evaluate", help="Evaluate generated results (MR / RMSD)")
    p_eval.add_argument("--dataset", required=True, help="e.g. mp_20 / mpts_52")
    p_eval.add_argument("--root-path", required=True, help="Directory containing generated results (usually finetune output dir)")
    p_eval.add_argument("--num-evals", type=int, default=1)
    p_eval.add_argument("--label", default="")
    p_eval.add_argument("--multi-eval", action="store_true")
    p_eval.set_defaults(func=cmd_csp_evaluate)

    # Supercon
    p_sc = sub.add_parser("supercon", help="Superconductivity-related commands")
    sc_sub = p_sc.add_subparsers(dest="sc_cmd", required=True)

    p_sc_gen = sc_sub.add_parser("generate", help="Generate structures for supercon_real / supercon_rest")
    p_sc_gen.add_argument("--dataset", required=True, choices=["supercon_real", "supercon_rest"])
    p_sc_gen.add_argument("--model-path", required=True, help="Finetune output dir (DAO-G on Supercon3D)")
    p_sc_gen.add_argument("--energy-model-path", default="", help="DAO-P checkpoint path")
    p_sc_gen.add_argument("--energy-guidance", action="store_true")
    p_sc_gen.add_argument("--num-evals", type=int, default=1)
    p_sc_gen.add_argument("--label", default="")
    p_sc_gen.add_argument("--gpu", type=int, default=None)
    p_sc_gen.add_argument("--aug", type=float, default=None)
    p_sc_gen.set_defaults(func=cmd_supercon_generate)

    # Property prediction helpers
    p_prop = sub.add_parser("prop", help="Property prediction helpers")
    prop_sub = p_prop.add_subparsers(dest="prop_cmd", required=True)

    p_pp = prop_sub.add_parser("predict", help="Predict a property for dataset/generated structures")
    p_pp.add_argument("--mode", required=True, choices=["dataset", "generated"])
    p_pp.add_argument("--ori-path", required=True)
    p_pp.add_argument("--eval-path", default="", help="Required when --mode=generated")
    p_pp.add_argument("--model-path", required=True)
    p_pp.add_argument("--prop", default="e_above_hull")
    p_pp.add_argument("--sample-size", type=int, default=1, help="Only for --mode=generated")
    p_pp.add_argument("--pred-energy", action="store_true")
    p_pp.add_argument("--out-npy", default="", help="Optional output .npy path")
    p_pp.set_defaults(func=cmd_prop_predict)

    ns = parser.parse_args(argv)
    return int(ns.func(ns))
