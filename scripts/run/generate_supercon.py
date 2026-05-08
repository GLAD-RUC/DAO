from datetime import datetime
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.experimental import compose
from hydra.initialize import initialize_config_dir

import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
import omegaconf
import os
import os.path as osp

from scripts._bootstrap import set_default_env
set_default_env()

# from dao.pl_modules.PTModels_guidance import CrystGenerativePretrainModel
from dao.pl_modules.PTModels import CrystGenerativePretrainModel
from dao.pl_modules.FTModels import CrystGenerativeFinetuneModel

from dao.pl_data.dataset import CrystDataset
from torch_geometric.data import DataLoader
from dao.pl_data.datamodule import worker_init_fn
from dao.common.data_utils import get_scaler_from_data_list

from scripts.eval.eval_utils import load_model, lattices_to_params_shape, recommand_step_lr
from dao.common.utils import log_hyperparameters, PROJECT_ROOT
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# from pyxtal.symmetry import Group

import copy
import numpy as np



def diffusion(loader, model, num_evals, energy_model=None, idx_pool=None, step_lr = 1e-5, energy_guidance=False, aug=1.):

    print('*********', aug)
    
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    # pred_energy_list = []
    # target_energy_list = []
    trajs = []
    
    for idx, batch in enumerate(loader):
        if args.end != -1 and idx not in idx_pool:
            continue
        
        if torch.cuda.is_available():
            batch.cuda()
        
        batch_all_frac_coords = []
        batch_all_lattices = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        batch_traj = []
        # batch_energy = []
        # target_energy_batch = model.pred_energy(batch)
        # target_energy_list.append(target_energy_batch)

        for eval_idx in range(num_evals):
            print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            outputs, traj = model.sample(batch, energy_model=energy_model, step_lr = step_lr, energy_guidance=energy_guidance, aug=aug)
            batch_traj.append(traj)
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lattices.append(outputs['lattices'].detach().cpu())
            # batch_energy.append(outputs['energy'].detach().cpu())

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        # pred_energy_list.append(torch.stack(batch_energy, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))
        trajs.append(batch_traj)

        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    # pred_energy = torch.cat(pred_energy_list, dim=1)
    pred_energy = None
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    # target_energy = torch.cat(target_energy_list, dim=0)

    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)


    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch, pred_energy, trajs
    )




def load_data(data_cfg, model_path=None, testing=True):
    datamodule = hydra.utils.instantiate(
            data_cfg.datamodule, pretrain=False, _recursive_=False, scaler_path=model_path
        )
    
    if testing:
        datamodule.setup('test')
        test_loader = datamodule.test_dataloader()[0]
    else:
        datamodule.setup()
        train_loader = datamodule.train_dataloader(shuffle=False)
        val_loader = datamodule.val_dataloader()[0]
        test_loader = (train_loader, val_loader)

    return test_loader, datamodule



def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)

    model, _, _ = load_model(
            model_path, load_data=False, from_scratch=False)

    if torch.cuda.is_available():
        model.to('cuda')

    if args.dataset == 'supercon_real':
        data_type = 'real'
        data_path = 'data/super_conductors/real_world/output.csv'
        save_path = 'data/super_conductors/real_world/output_ori.pt'
    elif args.dataset == 'supercon_rest':
        data_type = 'rest'
        data_path = 'data/super_conductors/supercon_rest/ordered_supercon_with_random_structure_dedup_pretrain_748.csv'
        save_path = 'data/super_conductors/supercon_rest/ordered_supercon_rest_dedup_748_ori.pt'
        
    dataset = CrystDataset(name=args.dataset, 
                           prop=args.prop,
                           path=data_path,
                           save_path=save_path,
                           max_atoms=100,
                           )
    
    
    test_loader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=50,
                num_workers=0,
                worker_init_fn=worker_init_fn,
            )
    
    
    dataset.lattice_scaler = get_scaler_from_data_list(
                    dataset.cached_data,
                    key='scaled_lattice')
    dataset.scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=args.prop)
    
    model.lattice_scaler = dataset.lattice_scaler
    model.scaler = dataset.scaler
         
    
    if args.energy_model_path != '':
        print('Load energy model......')
        energy_model = CrystGenerativePretrainModel.load_from_checkpoint(args.energy_model_path)
    else:
        energy_model = None


    if torch.cuda.is_available():
        model.to('cuda')
        if energy_model is not None:
            energy_model.to('cuda')

    print('Evaluate the diffusion model.')

    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['csp' if args.num_evals == 1 else 'csp_multi'][args.dataset]


    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch, pred_energy, trajs) = diffusion(
        test_loader, model, args.num_evals, \
        idx_pool=range(args.start, args.end), \
        energy_model=energy_model,  step_lr=step_lr, energy_guidance=args.energy_guidance, aug=args.aug)

    if args.label == '':
        diff_out_name = f'eval_diff_{data_type}.pt'
    else:
        diff_out_name = f'eval_diff_{data_type}_{args.label}.pt'

    out_dir = running_dir if args.no_ft else model_path
    torch.save({
        'eval_setting': args,
        'input_data_batch': input_data_batch,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lattices': lattices,
        'lengths': lengths,
        'angles': angles,
        'pred_energy': pred_energy,
        'time': time.time() - start_time,
        'trajs': trajs,
    }, out_dir / diff_out_name)    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--energy_model_path', default='')
    parser.add_argument('--running_dir', default='')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument(
        '--aug',
        default=None,
        type=float,
        help='Augmentation factor for energy guidance (default: 20 when --energy_guidance else 1).',
    )
    parser.add_argument('--start', default=-1, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--no_ft', action='store_true')
    parser.add_argument('--stable_only', action='store_true')
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--energy_guidance', action='store_true')
    parser.add_argument('--label', default='')
    parser.add_argument('--expname', default='demo')
    parser.add_argument('--prop', default='logtc')
    args = parser.parse_args()

    if args.aug is None:
        args.aug = 20.0 if args.energy_guidance else 1.0
    main(args)
