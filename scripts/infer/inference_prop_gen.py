import numpy as np
import pandas as pd
import torch
import os.path as osp

from scripts._bootstrap import set_default_env
set_default_env()
from dao.pl_data.dataset import MyDataset, SimpleDataset
from torch_geometric.data import DataLoader
from dao.pl_modules.PTModels import CrystGenerativePretrainModel
from dao.pl_modules.FTModels import CrystPredictiveFinetuneModel
from dao.pl_data.datamodule import worker_init_fn
from tqdm import tqdm
import argparse


def run(args):
    dataset = MyDataset(args.ori_path, args.eval_path, prop=args.prop, sample_size=args.sample_size)

    if args.pred_energy:
        ### load pretrained DAO-P model
        model = CrystGenerativePretrainModel.load_from_checkpoint(args.model_path)
    else:
        ### load finetuned DAO-P model
        model = CrystPredictiveFinetuneModel.load_from_checkpoint(args.model_path)

    scaler = torch.load(osp.join(osp.dirname(args.model_path), 'prop_scaler.pt'))
    model.eval()

    model.scaler = scaler
    dataset.scaler = scaler

    dataloder = DataLoader(
                    dataset,
                    shuffle=False,
                    batch_size=100,
                    num_workers=0,
                    worker_init_fn=worker_init_fn,
                )

    if torch.cuda.is_available():
        model.to('cuda')

    all_pred_props = []
    for idx, batch in enumerate(tqdm(dataloder)):
        if torch.cuda.is_available():
                batch.cuda()
        
        if args.pred_energy:
            pred_prop = model.pred_energy(batch)
        else:
            pred_prop = model.pred_prop(batch)
        
        pred_prop = scaler.inverse_transform(pred_prop)
        all_pred_props.extend(pred_prop.cpu().tolist())

    all_pred_props = np.array(all_pred_props)
    if args.sample_size != 1:
        all_pred_props = all_pred_props.reshape(args.sample_size, -1)
    eval_name = osp.splitext(osp.basename(args.eval_path))[0]
    np.save(f'{osp.dirname(args.eval_path)}/pred_{args.prop}_of_{eval_name}.npy', np.array(all_pred_props))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('predict props for structures.')
    parser.add_argument('--ori_path', type=str, required=True, help='path to the original dataset')
    parser.add_argument('--eval_path', type=str, required=True, help='path to the generated results, e.g. eval_xxx.pt')
    parser.add_argument('--model_path', type=str, required=True, help='path to the predictive model')
    parser.add_argument('--pred_energy', action='store_true', help='whether to predict energy')
    parser.add_argument('--sample_size', type=int, default=1, help='number of samples for each structure')
    parser.add_argument('--prop', type=str, default='e_above_hull', help='property to predict')
    args = parser.parse_args()
    run(args)
