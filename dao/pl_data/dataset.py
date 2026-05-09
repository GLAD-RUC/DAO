import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import os
from torch_geometric.data import Data
import pickle
import numpy as np
import os.path as osp
import sys
from tqdm import tqdm
sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))


from dao.common.utils import PROJECT_ROOT
from dao.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode='', path: ValueNode='',
                 prop: ValueNode='', niggli: ValueNode=True, primitive: ValueNode=False,
                 graph_method: ValueNode='crystalnn', preprocess_workers: ValueNode=30,
                 lattice_scale_method: ValueNode='scale_length', save_path: ValueNode='', tolerance: ValueNode=0.1, use_space_group: ValueNode=False, use_pos_index: ValueNode=False,
                 stable_threshold=0.08, transform_prop=True,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.stable_threshold = stable_threshold
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance
        self.transform_prop = transform_prop

        self.preprocess(save_path, preprocess_workers, prop)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
            print(save_path)
            print('==================data_num: ', len(self.cached_data))
        else:
            cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop],
            use_space_group=self.use_space_group,
            tol=self.tolerance)
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        is_stable = data_dict[self.prop] < self.stable_threshold
        if self.transform_prop:
            prop = self.scaler.transform(data_dict[self.prop])
        else:
            prop = torch.tensor(data_dict[self.prop], dtype=torch.float)

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        select_idx = np.random.choice(num_atoms, int(num_atoms*0.2))
        mask = np.ones_like(atom_types).astype('bool')
        mask[select_idx] = False

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
            y=prop.view(1, -1),
            mask=torch.from_numpy(mask),
            is_stable = is_stable,
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


class MyDataset(Dataset):
    def __init__(self, ori_path, path, prop='e_above_hull', stable_threshold=0.08, sample_size=1, transform_prop=True, **kwargs):
        super().__init__()
        self.path = path
        try:
            self.ori_data = torch.load(ori_path, map_location='cpu', weights_only=False)
        except TypeError:
            self.ori_data = torch.load(ori_path, map_location='cpu')
        self.prop = prop
        self.stable_threshold = stable_threshold
        self.transform_prop = transform_prop
        self.sample_size = sample_size
        self.cached_data = self.proprocess(path)

        self.lattice_scaler = None
        self.scaler = None

    def proprocess(self, path):
        try:
            res = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            res = torch.load(path, map_location='cpu')
        output_list = []
        for idx in range(self.sample_size):
            frac_coords = res['frac_coords'][idx]
            lengths = res['lengths'][idx]
            angles = res['angles'][idx]
            atom_types = res['atom_types'][idx]
            num_atoms = res['num_atoms'][idx]

            print(len(num_atoms), len(self.ori_data))
            assert len(num_atoms) == len(self.ori_data)

            start_idx = 0
            crystal_list = []
            for (batch_idx, num_atom) in tqdm(enumerate(num_atoms.tolist())):
                cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom).numpy()
                cur_atom_types = atom_types.narrow(0, start_idx, num_atom).numpy()

                cur_lengths = lengths[batch_idx]
                cur_angles = angles[batch_idx]

                crystal_list.append({
                    'frac_coords': cur_frac_coords,
                    'lengths': cur_lengths,
                    'angles': cur_angles,
                    'atom_types': cur_atom_types,
                    'edge_index': self.ori_data[batch_idx]['graph_arrays'][-3],
                    self.prop: self.ori_data[batch_idx][self.prop]
                })

                start_idx = start_idx + num_atom
            output_list.extend(crystal_list)
        return output_list

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        is_stable = data_dict[self.prop] < self.stable_threshold
        if self.transform_prop:
            prop = self.scaler.transform(data_dict[self.prop])
        else:
            prop = torch.tensor(data_dict[self.prop], dtype=torch.float)

        frac_coords = data_dict['frac_coords']
        atom_types = data_dict['atom_types']
        lengths = data_dict['lengths']
        angles = data_dict['angles']
        edge_indices = data_dict['edge_index']
        num_atoms = len(atom_types)

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),
            num_bonds=edge_indices.shape[0],
            num_atoms=num_atoms,
            num_nodes=num_atoms,
            y=prop.view(1, -1),
            is_stable = is_stable,
        )

        return data

    def __repr__(self) -> str:
        return f"MyDataset......"


class SimpleDataset(Dataset):
    def __init__(self, ori_path, prop='e_above_hull', stable_threshold=0.08, transform_prop=True, **kwargs):
        super().__init__()
        try:
            self.cached_data = torch.load(ori_path, map_location='cpu', weights_only=False)
        except TypeError:
            self.cached_data = torch.load(ori_path, map_location='cpu')
        self.prop = prop
        self.transform_prop = transform_prop

        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        if self.transform_prop:
            prop = self.scaler.transform(data_dict[self.prop])
        else:
            prop = torch.tensor(data_dict[self.prop], dtype=torch.float)

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),
            num_bonds=edge_indices.shape[0],
            num_atoms=num_atoms,
            num_nodes=num_atoms,
            y=prop.view(1, -1),
        )

        return data

    def __repr__(self) -> str:
        return f"SimpleDataset......"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from dao.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
