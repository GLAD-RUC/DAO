import argparse
from datetime import datetime
from pathlib import Path
from typing import List

from tqdm import tqdm

from scripts._bootstrap import set_default_env
set_default_env()
import hydra
import numpy as np
import torch
import os
import os.path as osp

from dao.common.utils import load_state_dict_from_checkpoint
from dao.common.data_utils import lattices_to_params_shape
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from dao.pl_modules.PTModels import CrystGenerativePretrainModel
from hydra.experimental import compose
from torch_geometric.data import Batch



def get_output(relaxed_props, data_list, data_scaler, energy_scaler, prop_name='stability'):
    output = []
    for relaxed_prop, item in zip(relaxed_props, data_list):
        graph_arrays = (item.frac_coords.cpu().numpy(), \
                        item.atom_types.cpu().numpy(), item.lengths[0].cpu().numpy(), \
                        item.angles[0].cpu().numpy(), \
                        item.edge_index.T.cpu().numpy(), \
                        item.to_jimages.cpu().numpy(), item.num_atoms[0].item())
        
        
        ### here, different scaler
        prop = data_scaler.inverse_transform(item.y[0][0]).cpu().item()
        

        inverse_relaxed_prop = energy_scaler.inverse_transform(relaxed_prop).cpu().item()
        
        # print('pre: ', prop)
        # print('relaxed: ', relaxed_prop)
        # print('--------')

        output.append({
            'graph_arrays': graph_arrays,
            'nmd_'+prop_name: item.y[0][0],
            'nmd_relaxed_'+prop_name: relaxed_prop,
            prop_name: prop,
            'relaxed_'+prop_name: inverse_relaxed_prop
        })
        
    return output

def main(args) -> None:
    seed_everything(42)
    model_dir = str(Path(args.model_dir).expanduser().resolve())
    cfg_dir = str(Path(args.cfg_dir).expanduser().resolve())
    if args.relax_mode == 'gradient':
        label = 'langevin' if args.add_noise else 'gradient'
    elif args.relax_mode == 'lbfgs':
        label = 'lbfgs'
    else:
        assert "wrong relax_mode"
        
    label += args.label

    print('-----label: ', label)
    
    with hydra.initialize_config_dir(model_dir):
        cfg = compose(config_name='hparams')
        ckpt = osp.join(model_dir, 'last.ckpt')

        model: CrystGenerativePretrainModel = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )
        state_dict = load_state_dict_from_checkpoint(ckpt)
        model.load_state_dict(state_dict=state_dict, strict=True)
        # model = CrystGenerativePretrainModel.load_from_checkpoint(ckpt)
        # try:
        #     model.lattice_scaler = torch.load(cfg_dir+'/lattice_scaler.pt')
        #     model.scaler = torch.load(cfg_dir+'/prop_scaler.pt')
        # except:
        #     pass

        energy_scaler = torch.load(f'{model_dir}/prop_scaler.pt')
        if torch.cuda.is_available():
            model.cuda()

    
    with hydra.initialize_config_dir(cfg_dir):
        data_cfg = compose(config_name='hparams')

        datamodule = hydra.utils.instantiate(
            data_cfg.data.datamodule, _recursive_=False, pretrain=True, scaler_path=cfg_dir
        )

        # datamodule.batch_size.train = 400
        datamodule.batch_size.train = 100

        datamodule.setup()
        data_loader = datamodule.train_dataloader(shuffle=False)
        prop_name = datamodule.train_dataset.prop
        
    # model.lattice_scaler = datamodule.lattice_scaler.copy()
    # model.scaler = datamodule.scaler.copy()
    

    outputs = []
    
    if not args.relax_stable:
        print('donot relax stable data.......')
    
    for idx, batch_ori in enumerate(tqdm(data_loader)):
        if args.end != -1 and idx not in range(args.start, args.end):
            continue

        if torch.cuda.is_available():
            batch_ori = batch_ori.cuda()

        data_list_ori = batch_ori.to_data_list()
        
        if not args.relax_stable:
            stable_threshold = datamodule.scaler.transform(0.08)
            stable_idx = (batch_ori.y <= stable_threshold)
            stable_list = [data_list_ori[i] for (i, is_stable) in enumerate(stable_idx) if is_stable]
            data_list = [data_list_ori[i] for (i, is_stable) in enumerate(stable_idx) if not is_stable]
            if len(data_list) == 0:
                print('The data in this batch are all stable.....')
                if args.return_stable:
                    props = [item.y[0][0] for item in stable_list]
                    outputs.extend(get_output(props, stable_list, datamodule.scaler, energy_scaler, prop_name))
                continue
            batch = Batch.from_data_list(data_list)
        else:
            stable_list = []
            data_list = data_list_ori
            batch = batch_ori
        
        relax_threshold = datamodule.scaler.transform(args.threshold)
        relax_idx = (batch.y <= relax_threshold)
        
        relax_data_list = [data_list[i] for (i, is_relax) in enumerate(relax_idx) if is_relax]
        print(f'************** relax num: {len(relax_data_list)}/{datamodule.batch_size.train}')
        if len(relax_data_list) == 0:
            print('There are no data need to relax in this batch.....')
            if args.return_unrelax:
                props = [item.y[0][0] for item in data_list]
                outputs.extend(get_output(props, data_list, datamodule.scaler, energy_scaler, prop_name))
            continue
            
        
        relax_batch = Batch.from_data_list(relax_data_list)

        frac_coords, lattices, pred_props, relaxed_props = model.relax(relax_batch, num_steps=args.num_steps, \
                                                        step_size=args.step_size, add_noise=args.add_noise, \
                                                            mode=args.relax_mode, relax_threshold=args.threshold)
        

        lengths, angles = lattices_to_params_shape(lattices)
        relax_batch.lengths, relax_batch.angles, relax_batch.frac_coords = lengths, angles, frac_coords
        relax_data_list = relax_batch.to_data_list()
        
        data_list_out = []
        props_out = []
        if args.return_stable:
            data_list_out += stable_list
            props_out += [item.y[0][0] for item in stable_list]
        
        if not args.return_unrelax:
            data_list = relax_data_list
        else:
            new_relaxed_props = batch.y
            pos = 0
            for i, is_relax in enumerate(relax_idx):
                if is_relax:
                    data_list[i] = relax_data_list[pos]
                    new_relaxed_props[i] = relaxed_props[pos]
                    pos += 1
            relaxed_props = new_relaxed_props

        data_list_out += data_list
        props_out += [prop for prop in relaxed_props]
        
        outputs.extend(get_output(props_out, data_list_out, datamodule.scaler, energy_scaler, prop_name))
        

    torch.save(outputs, cfg_dir+f'/relax_data_{label}_{args.step_size}_{args.num_steps}_{args.threshold}.pt')
    print('----------outputs num: ', len(outputs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--cfg_dir', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--num_steps', type=int, default=2)
    parser.add_argument('--step_size', type=float, default=0.005)
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--return_stable', type=int, default=0)
    parser.add_argument('--return_unrelax', type=int, default=0)
    parser.add_argument('--relax_stable', action='store_true')
    parser.add_argument('--relax_mode', default='gradient')
    parser.add_argument('--threshold', type=float, default=0.08)
    parser.add_argument('--start', default=-1, type=int)
    parser.add_argument('--end', default=-1, type=int)
    args = parser.parse_args()

    ## change to Bool
    args.return_stable = args.return_stable and 1
    args.return_unrelax = args.return_unrelax and 1
    main(args)
