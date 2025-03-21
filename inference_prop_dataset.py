import numpy as np
import pandas as pd
import torch
import sys
import os.path as osp
sys.path.append('..')

from diffcsp.pl_data.dataset import MyDataset, SimpleDataset
from torch_geometric.data import DataLoader
from diffcsp.pl_modules.PTModels import CrystGenerativePretrainModel
from diffcsp.pl_modules.FTModels import CrystGenerativeFinetuneModel
from diffcsp.pl_data.datamodule import worker_init_fn
from tqdm import tqdm
import argparse


def run(args):      
    dataset = SimpleDataset(args.ori_path, prop=args.prop)

    if args.pred_energy:
        ### load pretrained DAO-P model
        model = CrystGenerativePretrainModel.load_from_checkpoint(args.model_path)
    else:
        ### load finetuned DAO-P model
        model = CrystPredictiveFinetuneModel.load_from_checkpoint(args.model_path)
    scaler = torch.load(osp.dirname(args.model_path) + '/prop_scaler.pt')
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

    pred_props = []
    gt_props = []
    for idx, batch in enumerate(tqdm(dataloder)):
        if torch.cuda.is_available():
                batch.cuda()
                
        if args.pred_energy:
            pred_prop = model.pred_energy(batch)
        else:
            pred_prop = model.pred_prop(batch)
        pred_props.extend(pred_prop.cpu().tolist())
        
        gt_props.extend(batch.y.cpu().tolist())
    
    
    inverse_pred_props = scaler.inverse_transform(np.array(pred_props).reshape(-1, 1)).flatten().numpy()
    inverse_gt_props = scaler.inverse_transform(np.array(gt_props).reshape(-1, 1)).flatten().numpy()
    
    mae = np.mean(np.abs(inverse_pred_props - inverse_gt_props))
    print(f'MAE: {mae}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate the prop of dataset structures.')
    parser.add_argument('--ori_path', type=str, required=True, help='path to the original dataset')
    parser.add_argument('--model_path', type=str, required=True, help='path to the predictive model')
    parser.add_argument('--pred_energy', action='store_true', help='whether to predict energy')
    parser.add_argument('--prop', type=str, default='e_above_hull', help='property to predict')
    args = parser.parse_args()
    run(args)