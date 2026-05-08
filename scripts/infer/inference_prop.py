import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from scripts._bootstrap import set_default_env

set_default_env()

from dao.pl_data.dataset import MyDataset, SimpleDataset
from dao.pl_data.datamodule import worker_init_fn
from dao.pl_modules.PTModels import CrystGenerativePretrainModel
from dao.pl_modules.FTModels import CrystPredictiveFinetuneModel


class PropInferenceDataset(torch.utils.data.Dataset):
    """
        Unify two types of inference data sources: dataset and generated.

        eval_path is None: directly make predictions on ori_path (e.g., *_ori.pt)

        eval_path is not None: make predictions on generated results eval_diff*.pt (using ori_path to provide graph/ground truth)
    """

    def __init__(
        self,
        *,
        ori_path: str,
        prop: str,
        eval_path: Optional[str] = None,
        sample_size: int = 1,
    ):
        super().__init__()
        self.eval_path = eval_path
        if eval_path is None:
            self._ds = SimpleDataset(ori_path, prop=prop)
        else:
            self._ds = MyDataset(ori_path, eval_path, prop=prop, sample_size=sample_size)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]

    @property
    def scaler(self):
        return getattr(self._ds, "scaler", None)

    @scaler.setter
    def scaler(self, v):
        setattr(self._ds, "scaler", v)


def _default_out_npy(*, mode: str, prop: str, ori_path: str, eval_path: Optional[str]) -> Path:
    if mode == "generated":
        assert eval_path is not None
        base = Path(eval_path)
        return base.parent / f"pred_{prop}_of_{base.stem}.npy"
    base = Path(ori_path)
    return base.parent / f"pred_{prop}_of_{base.stem}.npy"


def run_inference(
    *,
    mode: str,
    ori_path: str,
    model_path: str,
    prop: str,
    pred_energy: bool,
    eval_path: Optional[str] = None,
    sample_size: int = 1,
    batch_size: int = 100,
    out_npy: Optional[str] = None,
) -> float:
    dataset = PropInferenceDataset(ori_path=ori_path, prop=prop, eval_path=eval_path, sample_size=sample_size)

    if pred_energy:
        model = CrystGenerativePretrainModel.load_from_checkpoint(model_path)
    else:
        model = CrystPredictiveFinetuneModel.load_from_checkpoint(model_path)

    scaler_path = Path(model_path).expanduser().resolve().parent / "prop_scaler.pt"
    scaler = torch.load(str(scaler_path))

    model.eval()
    model.scaler = scaler
    dataset.scaler = scaler

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )

    if torch.cuda.is_available():
        model.to("cuda")

    pred_props = []
    gt_props = []
    for _, batch in enumerate(tqdm(dataloader)):
        if torch.cuda.is_available():
            batch.cuda()

        if pred_energy:
            pred_prop = model.pred_energy(batch)
        else:
            pred_prop = model.pred_prop(batch)

        pred_props.extend(pred_prop.detach().cpu().tolist())
        gt_props.extend(batch.y.detach().cpu().tolist())

    inverse_pred = scaler.inverse_transform(np.array(pred_props).reshape(-1, 1)).flatten().numpy()
    inverse_gt = scaler.inverse_transform(np.array(gt_props).reshape(-1, 1)).flatten().numpy()
    mae = float(np.mean(np.abs(inverse_pred - inverse_gt)))

    # save npy (real unit)
    pred_arr = inverse_pred
    if mode == "generated" and sample_size != 1:
        pred_arr = pred_arr.reshape(sample_size, -1)

    out_path = Path(out_npy) if out_npy else _default_out_npy(mode=mode, prop=prop, ori_path=ori_path, eval_path=eval_path)
    out_path = out_path.expanduser().resolve()
    np.save(str(out_path), np.array(pred_arr))

    print(f"MAE: {mae}")
    print(f"Saved predictions to: {out_path}")
    return mae


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser("Unified property inference")
    parser.add_argument("--mode", choices=["dataset", "generated"], required=True)
    parser.add_argument("--ori_path", required=True, help="Path to original cached dataset (e.g. *_ori.pt)")
    parser.add_argument("--eval_path", default="", help="Path to generated results (eval_diff*.pt), required for mode=generated")
    parser.add_argument("--model_path", required=True, help="Path to DAO-P checkpoint")
    parser.add_argument("--pred_energy", action="store_true", help="Use pretrained DAO-P (energy head)")
    parser.add_argument("--prop", default="e_above_hull")
    parser.add_argument("--sample_size", type=int, default=1, help="Only for mode=generated")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--out_npy", default="", help="Where to save predicted properties (.npy)")
    args = parser.parse_args(argv)

    if args.mode == "generated" and not args.eval_path:
        raise SystemExit("--eval_path is required when --mode=generated")

    run_inference(
        mode=args.mode,
        ori_path=args.ori_path,
        eval_path=args.eval_path if args.mode == "generated" else None,
        model_path=args.model_path,
        prop=args.prop,
        pred_energy=args.pred_energy,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        out_npy=args.out_npy or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
