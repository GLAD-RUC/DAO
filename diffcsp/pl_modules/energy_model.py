from copy import deepcopy
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
from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import lattice_params_to_matrix_torch
from diffcsp.pl_modules.PTModels_guidance import BaseModule, CrystGenerativePretrainModel
from diffcsp.pl_modules.crysformer import CrysFormer 
from diffcsp.pl_modules.cspnet import CSPNet
from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal
# from warmup_scheduler import GradualWarmupScheduler
from torch.autograd import grad
from torch.optim.lr_scheduler import LinearLR, SequentialLR


MAX_ATOMIC_NUM=100


### Model definition

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
    


class IEPModule(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.diffuse = True 
        self.max_atoms = 100

        try:   
            pretrain_model = CrystGenerativePretrainModel.load_from_checkpoint(self.hparams.pretrain_repr)
            self.decoder = deepcopy(pretrain_model.decoder)
            # self.decoder.freeze()
            for name, param in self.decoder.named_parameters():
                if 'scalar_out' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False 
        except:
            print('******** Load model error! ********')
            assert False

        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)

        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)

        self.logsoftmax = nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=0) 


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

        energy_0 = batch.y
        temperature = 1.

        p_label = torch.exp( -energy_0 * temperature)
        # p_pred = torch.exp(-energy_t * temperature)
        p_pred = torch.exp(-energy_t)

        loss_scalar = F.mse_loss(p_pred, p_label)
        return { 'loss' : self.hparams.cost_scalar * loss_scalar }



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        ###### here
        output_dict = self(batch, stable_check=False)

        loss = output_dict['loss']

        self.log_dict(
            {
                'train_loss': loss,
                'val_loss': loss
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # if loss.isnan():
        #     return None

        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        pass

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        pass
    
