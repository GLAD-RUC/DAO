from copy import deepcopy
import hydra
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Dict, List, Optional

from dao.common.data_utils import lattice_params_to_matrix_torch
from dao.pl_modules.PTModels import BaseModule, CrystGenerativePretrainModel, SinusoidalTimeEmbeddings
from dao.common.utils import RequiresGradContext, cal_grad
from dao.pl_modules.diff_utils import d_log_p_wrapped_normal

MAX_ATOMIC_NUM = 100


class CrystFinetuneModel(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.diffuse = self.hparams.diffuse
        self.max_atoms = 100
        latent_dim = self.hparams.latent_dim + self.hparams.time_dim if self.diffuse else self.hparams.latent_dim

        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = latent_dim, diffuse=self.diffuse, \
                                            _recursive_=False, max_atoms=self.max_atoms)

        if not getattr(self.hparams, "from_scratch", False):
            try:
                print('Loding GenerativePretrainModel.......')
                pretrain_model = CrystGenerativePretrainModel.load_from_checkpoint(self.hparams.pretrain_repr)
                self.decoder = deepcopy(pretrain_model.decoder)
            except Exception as e:
                print(e)
                print('******** Load model error! ********')
                pass

        self.time_embedding = SinusoidalTimeEmbeddings(self.hparams.time_dim)

    def forward(self, batch):
        pass

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        pass

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int=0) -> torch.Tensor:
        torch.cuda.empty_cache()
        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch, mode='test')

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
            sync_dist=True,
            on_epoch=False
        )
        return loss

    def compute_stats(self, output_dict, prefix):
        pass


class CrystPredictiveFinetuneModel(CrystFinetuneModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dataset=self.hparams['data']['root_path'].split('/')[-1]
        print('************ dataset: ', self.dataset)

        feat_dim = self.decoder.hidden_dim

        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, 1),
        )

        self.predictions = []
        self.targets = []

    def forward(self, batch, mode='train'):
        """
        propery in batch:
            edge_index: [2, num_edges]
            y: [num_graphs, 1]
            frac_coords: [num_nodes, 3]
            atom_types: [num_nodes, ]
            lengths: [num_graphs, 3]
            angles: [num_graphs, 3]
            to_jimages: [num_edges, 3]
            num_atoms: [num_graphs, ]
            num_bonds: [num_graphs, ]
            num_nodes: num_nodes
            batch: [num_nodes, ], i.e. node2graph
        """
        batch_size = batch.num_graphs
        time_emb_zeros = self.time_embedding(torch.zeros(batch_size, device=self.device))
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        input_frac_coords = frac_coords
        input_lattice = lattices

        node_rep, graph_rep = self.decoder(time_emb_zeros, batch.atom_types, input_frac_coords, \
                                                        input_lattice, batch.num_atoms, batch.batch, only_rep=True)
        pred_scalar = self.predictor(graph_rep)

        tar_scalar = batch.y

        if mode == 'test':
            self.predictions.append(pred_scalar)
            self.targets.append(batch.y)

        self.scaler.match_device(pred_scalar)
        loss_scalar = F.l1_loss(pred_scalar, tar_scalar) * self.scaler.stds

        return {
                'loss' : loss_scalar,
            }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        loss = output_dict['loss']

        self.log_dict(
            {'train_loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        return loss

    @torch.no_grad()
    def pred_prop(self, batch):
        batch_size = batch.num_graphs
        frac_coords = batch.frac_coords
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_emb_zeros = self.time_embedding(torch.zeros(batch_size, device=self.device))
        node_rep, graph_rep = self.decoder(time_emb_zeros, batch.atom_types, frac_coords % 1, lattices, batch.num_atoms, batch.batch, only_rep=True)
        pred_scalar = self.predictor(graph_rep)

        return pred_scalar.squeeze(-1)

    def compute_stats(self, output_dict, prefix):
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
        }

        return log_dict, loss

    def on_test_epoch_end(self):
        # Gather all predictions and targets across all GPUs
        all_preds = self.all_gather(torch.cat(self.predictions))
        all_targets = self.all_gather(torch.cat(self.targets))

        # Ensure only the main process computes the metrics
        if self.trainer.is_global_zero:
            # Concatenate all gathered tensors
            all_preds = torch.cat([p for p in all_preds], dim=0)
            all_targets = torch.cat([t for t in all_targets], dim=0)

            loss_scalar = F.l1_loss(all_preds, all_targets) * self.scaler.stds
            print('loss: ', loss_scalar)
            self.log_dict(
                {'final_test_loss': loss_scalar},
            )

        # Clear stored predictions and targets for the next epoch
        self.predictions.clear()
        self.targets.clear()


class CrystGenerativeFinetuneModel(CrystFinetuneModel):
    def __init__(self, *args, **kwargs) -> None:
        kwargs['diffuse']=True
        super().__init__(*args, **kwargs)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)

    def forward(self, batch, mode='train'):
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.


        pred_l, pred_x, _, _, _, energy_t  = self.decoder(time_emb, batch.atom_types, input_frac_coords, \
                                                  input_lattice, batch.num_atoms, batch.batch)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
        tar_l = rand_l

        if self.hparams.finetune_energy:
            energy_0 = batch.y
            temperature = 1.
            p_label = torch.exp( -energy_0 * temperature)
            p_pred = torch.exp(-energy_t)
            loss_energy = F.mse_loss(p_pred, p_label)
        else:
            loss_energy = 0.

        loss_coord = F.mse_loss(pred_x, tar_x)
        loss_lattice = F.mse_loss(pred_l, tar_l)

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord +
            self.hparams.cost_scalar * loss_energy)

        return {
            'loss' : loss,
            'loss_lattice' : self.hparams.cost_lattice * loss_lattice,
            'loss_coord' : self.hparams.cost_coord * loss_coord,
            'loss_energy' : self.hparams.cost_scalar * loss_energy,
        }

    @torch.no_grad()
    def sample(self, batch, energy_model=None, step_lr = 1e-5, energy_guidance=False, aug=1.):
        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T,
        }}


        for t in tqdm(range(time_start, 0, -1)):
            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x, _, _, _, _ = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

            l_t_minus_05 = l_t

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1]
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))

            if energy_guidance:
                with torch.enable_grad():
                    with RequiresGradContext(x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_l, pred_x, _, _, _, energy_t = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        if energy_model is not None:
                            _, _, _, _, _, energy_t = energy_model.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(energy_t)]
                        grad_x, grad_l = grad(energy_t, [x_t_minus_05, l_t_minus_05], grad_outputs = grad_outputs, allow_unused=True)

                pred_x = pred_x * torch.sqrt(sigma_norm)
                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x
                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l
                x_t_minus_1 = x_t_minus_1 % 1.
                del grad_x, grad_l
            else:
                pred_l, pred_x, _, _, _, _ = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                pred_x = pred_x * torch.sqrt(sigma_norm)
                x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x
                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l
                x_t_minus_1 = x_t_minus_1 % 1.

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1,
                'lattices' : l_t_minus_1,
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_energy = output_dict['loss_energy']
        loss = output_dict['loss']

        self.log_dict(
            {
                'train_loss': loss,
                'lattice_loss': loss_lattice,
                'coord_loss': loss_coord,
                'energy_loss': loss_energy,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_energy = output_dict['loss_energy']
        loss = output_dict['loss']

        log_dict = {
                f'{prefix}_loss': loss,
                f'{prefix}_lattice_loss': loss_lattice,
                f'{prefix}_coord_loss': loss_coord,
                f'{prefix}_energy_loss': loss_energy,
            }

        return log_dict, loss
