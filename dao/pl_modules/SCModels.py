import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict
import hydra
from tqdm import tqdm


from dao.common.data_utils import lattice_params_to_matrix_torch
from dao.pl_modules.PTModels import BaseModule, SinusoidalTimeEmbeddings
from dao.pl_modules.diff_utils import d_log_p_wrapped_normal

MAX_ATOMIC_NUM=100

class CrystScratchModel(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.diffuse = self.hparams.diffuse 
        self.max_atoms = 100
        latent_dim = self.hparams.latent_dim + self.hparams.time_dim if self.diffuse else self.hparams.latent_dim

        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = latent_dim, diffuse=self.diffuse, \
                                                 _recursive_=False, max_atoms=self.max_atoms)
        
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.feat_dim = self.hparams.decoder.hidden_dim
        
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        # feat_dim = self.hparams.decoder.hidden_dim

        # self.atom_predictor = nn.Sequential(
        #     nn.Linear(feat_dim, feat_dim),
        #     nn.ReLU(),
        #     nn.Linear(feat_dim, self.max_atoms),
        # )


    def forward(self, batch):
        pass

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # torch.set_grad_enabled(True)
        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # torch.set_grad_enabled(True)
        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss


    def compute_stats(self, output_dict, prefix):
        pass


class CrystPredictiveScratchModel(CrystScratchModel):
    def __init__(self, *args, **kwargs) -> None:
        kwargs['diffuse']=True
        super().__init__(*args, **kwargs)
        
        if self.hparams.powerful_predictor:
            self.predictor = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim*2),
                nn.SiLU(),
                nn.Linear(self.feat_dim*2, self.feat_dim*2),
                nn.SiLU(),
                nn.Linear(self.feat_dim*2, self.feat_dim*2),
                nn.SiLU(),
                nn.Linear(self.feat_dim*2, 1),
            )
        else:
            self.predictor = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.SiLU(),
            nn.Linear(self.feat_dim, 1),
        )

    
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

        # res = self.backbone.decoder(time_emb_zeros, batch.atom_types, input_frac_coords, \
        #                                                 input_lattice, batch.num_atoms, batch.batch, only_rep=False)
        # pred_scalar = res[-1]

        tar_scalar = batch.y
        print('pred: ', pred_scalar[:10])
        print('tar: ', tar_scalar[:10])


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
        )

        # if loss.isnan():
        #     return None

        return loss


    def compute_stats(self, output_dict, prefix):
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
        }

        # if loss.isnan():
        #     return None


        return log_dict, loss


class CrystGenerativeScratchModel(CrystScratchModel):
    def __init__(self, *args, **kwargs) -> None:
        kwargs['diffuse']=True
        super().__init__(*args, **kwargs)
        for param in self.decoder.scalar_out.parameters():
            param.requires_grad = False


    def forward(self, batch, stable_check=True):
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)
        time_emb_zeros = self.time_embedding(torch.zeros_like(times, device=self.device))

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

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        pred_l, pred_x, _, _, _, energy_t  = self.decoder(time_emb, batch.atom_types, input_frac_coords, \
                                                  input_lattice, batch.num_atoms, batch.batch)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
        tar_l = rand_l


        loss_coord = F.mse_loss(pred_x, tar_x)
        loss_lattice = F.mse_loss(pred_l, tar_l)

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)
        
        return {
            'loss' : loss,
            'loss_lattice' : self.hparams.cost_lattice * loss_lattice,
            'loss_coord' : self.hparams.cost_coord * loss_coord,
        }
    


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']


        # print(loss.item(), loss_lattice.item(), loss_coord.item(), loss_scalar.item())
        self.log_dict(
            {
                'train_loss': loss,
                'lattice_loss': loss_lattice,
                'coord_loss': loss_coord,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # if loss.isnan():
        #     return None

        return loss

    

    @torch.no_grad()
    def pred_energy(self, batch):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)


        time_emb_zeros = self.time_embedding(torch.zeros(batch_size, device=self.device))
        _, _, _, _, _, energy_T = self.decoder(time_emb_zeros, batch.atom_types, x_T % 1, l_T, batch.num_atoms, batch.batch)

        return energy_T

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps
        # time_start = 2
      
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

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x, _, _, _, _ = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x, _, _, _, pred_energy = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t

            x_t_minus_1 = x_t_minus_1 % 1.
            
            # if t == 1:
            #     _, _, _, _, _, pred_energy = self.decoder(time_emb, batch.atom_types, x_t_minus_1, l_t_minus_1, batch.num_atoms, batch.batch)
            # else:
            pred_energy = torch.zeros_like(batch.num_atoms).unsqueeze(-1).to(self.device)


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1,
                'lattices' : l_t_minus_1,
                'energy': pred_energy.squeeze(-1),                
            }
            

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack
    

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        log_dict = {
                f'{prefix}_loss': loss,
                f'{prefix}_lattice_loss': loss_lattice,
                f'{prefix}_coord_loss': loss_coord,
            }

        return log_dict, loss
