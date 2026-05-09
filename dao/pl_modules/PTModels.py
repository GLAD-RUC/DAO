import math, copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

import hydra
import pytorch_lightning as pl
from torch_scatter import scatter
from tqdm import tqdm
import copy
from dao.common.utils import PROJECT_ROOT, RequiresGradContext
from dao.common.data_utils import lattice_params_to_matrix_torch
from dao.pl_modules.crysformer import CrysFormer
from dao.pl_modules.cspnet import CSPNet
from dao.pl_modules.diff_utils import d_log_p_wrapped_normal
from torch.autograd import grad
from torch.optim.lr_scheduler import LinearLR, SequentialLR


MAX_ATOMIC_NUM = 100


def mask_mse(pred, target, mask):
    assert mask.shape[0] == pred.shape[0] and mask.shape[0] == target.shape[0]
    assert pred.shape == target.shape
    loss = F.mse_loss(pred, target, reduction='none').mean(tuple(range(1, len(pred.shape))))
    mask_loss = torch.masked_select(loss, mask).mean()
    return mask_loss


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )

        if not self.hparams.optim.use_lr_scheduler:
            return [opt]

        if self.hparams.optim.scheduler_type == 'cos':
            scheduler = hydra.utils.instantiate(
                self.hparams.optim.lr_scheduler_cos, optimizer=opt
            )
        else:
            scheduler = hydra.utils.instantiate(
                self.hparams.optim.lr_scheduler, optimizer=opt
            )

        if self.hparams.optim.warm_up:
            scheduler_warm = LinearLR(opt, start_factor=0.001, end_factor=1., total_iters=self.hparams.optim.warm_epochs)
            scheduler = SequentialLR(optimizer=opt, schedulers=[scheduler_warm, scheduler], milestones=[scheduler_warm.total_iters])
            scheduler.optimizer=opt
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CrystPretrainModel(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.diffuse = self.hparams.diffuse
        self.max_atoms = 100
        latent_dim = self.hparams.latent_dim + self.hparams.time_dim if self.diffuse else self.hparams.latent_dim

        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = latent_dim, diffuse=self.diffuse, \
                                                 _recursive_=False, max_atoms=self.max_atoms)

        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch, stable_check=False)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_energy = output_dict['loss_energy']
        loss = output_dict['loss']

        self.log_dict(
            {
                'train_loss': loss,
                'lattice_loss': loss_lattice,
                'coord_loss': loss_coord,
                'loss_energy': loss_energy,
                'val_loss': loss
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def forward(self, batch):
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
            is_stable: [num_graphs, ]
        """
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        pass

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        pass


class CrystGenerativePretrainModel(CrystPretrainModel):
    def __init__(self, *args, **kwargs) -> None:
        kwargs['diffuse']=True
        kwargs['energy_guidance']=False
        kwargs['guidance_mode']='exp'
        super().__init__(*args, **kwargs)

        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)

        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)

        self.logsoftmax = nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, batch, stable_check=False):
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]

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

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        pred_l, pred_x, _, _, _, energy_t  = self.decoder(time_emb, batch.atom_types, input_frac_coords, \
                                                  input_lattice, batch.num_atoms, batch.batch)

        loss_coord = 0.
        loss_lattice = 0.
        loss_energy = 0.

        if not self.hparams.only_energy:
            tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
            tar_l = rand_l
            loss_coord = F.mse_loss(pred_x, tar_x)
            loss_lattice = F.mse_loss(pred_l, tar_l)


        if not self.hparams.only_diffusion:
            if self.hparams.exp_energy_loss:
                energy_0 = batch.y
                temperature = 1.
                p_label = torch.exp( -energy_0 * temperature)
                p_pred = torch.exp(-energy_t)
            else:
                p_label = batch.y
                p_pred = energy_t

            loss_energy = F.mse_loss(p_pred, p_label)


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
    def pred_energy(self, batch):
        batch_size = batch.num_graphs
        frac_coords = batch.frac_coords
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_emb_zeros = self.time_embedding(torch.zeros(batch_size, device=self.device))
        outputs = self.decoder(time_emb_zeros, batch.atom_types, frac_coords % 1, lattices, batch.num_atoms, batch.batch, only_rep=False)


        return outputs[-1].squeeze(-1)

    @torch.no_grad()
    def get_rep(self, batch):
        batch_size = batch.num_graphs
        frac_coords = batch.frac_coords
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)


        time_emb_zeros = self.time_embedding(torch.zeros(batch_size, device=self.device))
        node_features, graph_features = self.decoder(time_emb_zeros, batch.atom_types, frac_coords % 1, lattices, batch.num_atoms, batch.batch, only_rep=True)

        return node_features, graph_features

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

            if t == 1:
                time_embd_zeros = self.time_embedding(torch.zeros_like(times, device=self.device))
                if energy_model is not None:
                    _, _, _, _, _, energy_t = energy_model.decoder(time_embd_zeros, batch.atom_types, x_t_minus_1, l_t_minus_1, batch.num_atoms, batch.batch)
                else:
                    _, _, _, _, _, energy_t = self.decoder(time_embd_zeros, batch.atom_types, x_t_minus_1, l_t_minus_1, batch.num_atoms, batch.batch)
                energy_t = energy_t.detach()
                self.scaler.match_device(energy_t)
                energy_t = self.scaler.inverse_transform(energy_t)
            else:
                energy_t = torch.zeros_like(batch.num_atoms).unsqueeze(-1).to(self.device)


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1,
                'lattices' : l_t_minus_1,
                'energy': energy_t.squeeze(-1),
            }


        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack

    @torch.no_grad()
    def relax(self, batch, step_size=0.005, num_steps=2, add_noise=False, mode='gradient', relax_threshold=0.08, return_relax_only=False):

        batch_size = batch.num_graphs
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords: torch.tensor = batch.frac_coords % 1
        time_emb_zeros = self.time_embedding(torch.zeros(batch_size, device=lattices.device))

        lattices_base = copy.deepcopy(lattices)
        frac_coords_base = copy.deepcopy(frac_coords)
        pred_energy = None

        if mode == 'gradient':
            for step in range(num_steps):
                with torch.enable_grad():
                    with RequiresGradContext(frac_coords, lattices, requires_grad=True):
                        _, _, _, _, _, energy = self.decoder(time_emb_zeros, batch.atom_types, frac_coords, lattices, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(energy)]
                        grad_x, grad_l = grad(energy, [frac_coords, lattices], grad_outputs = grad_outputs, allow_unused=True)

                noise_scale = 2 * step_size if  add_noise else 0.
                noise_x = torch.randn_like(frac_coords) * noise_scale
                noise_l = torch.randn_like(lattices) *  noise_scale
                frac_coords = (frac_coords - step_size * grad_x + noise_x) % 1
                lattices = lattices - step_size * grad_l + noise_l

                del grad_x, grad_l

                if step == 0:
                    pred_energy = energy
        else:
            frac_coords.requires_grad = True
            lattices.requires_grad = True
            # Set up the L-BFGS optimizer
            optimizer = torch.optim.LBFGS([frac_coords, lattices], lr=step_size, max_iter=num_steps)

            def closure():
                optimizer.zero_grad()    # Clear any existing gradients
                with torch.enable_grad():
                    _, _, _, _, _, energy = self.decoder(time_emb_zeros, batch.atom_types, frac_coords, lattices, batch.num_atoms, batch.batch)
                energy = energy.mean()
                energy.backward()        # Calculate gradients w.r.t. x
                return energy            # Return the energy (for L-BFGS to minimize)

            # Run the optimization loop
            optimizer.step(closure)
            frac_coords %= 1.

        _, _, _, _, _, relaxed_energy = self.decoder(time_emb_zeros, batch.atom_types, frac_coords, lattices, batch.num_atoms, batch.batch)

        if pred_energy is None:
            pred_energy = relaxed_energy

        return frac_coords.detach(), lattices.detach(), pred_energy.detach(), relaxed_energy.detach()
