import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict
import math


from diffcsp.common.data_utils import lattice_params_to_matrix_torch
from diffcsp.pl_modules.PTModels_guidance import BaseModule, CrystGenerativePretrainModel, CrystPredictivePretrainModel

MAX_ATOMIC_NUM=100


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module, r: int, alpha: int):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        x = self.w(x) + (self.alpha // self.r) * self.w_b(self.w_a(x))
        # c = (self.alpha // self.r) * self.w_b(self.w_a(x))
        # print('w_a: ', self.w_a.weight.grad)
        # print('w_b: ', self.w_b.weight.grad)
        # print('res: ', c)
        return x




class CrystFinetuneModel(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.valid_grad=False
        self.use_lora=True
        try:   
            if not self.hparams.from_scratch:
                if self.hparams.pretrain_mode == 'pred':
                    print('Loding PredectivePretrainModel.......')
                    self.backbone = CrystPredictivePretrainModel.load_from_checkpoint(self.hparams.pretrain_repr)
                else:
                    print('Loding GenerativePretrainModel.......')
                    self.backbone = CrystGenerativePretrainModel.load_from_checkpoint(self.hparams.pretrain_repr)
                # self.backbone.freeze()
        except:
            print('******** Load model error! ********')
            assert False

        if self.use_lora:
            ####  code from https://github.com/JamesQFreeman/LoRA-ViT/blob/main/lora.py
            r=8
            dim=self.backbone.decoder.hidden_dim
            alpha=r*2

            # create for storage, then we can init them or load weights
            self.w_As = []  # These are linear layers
            self.w_Bs = []

            # lets freeze first
            self.backbone.freeze()

            # Here, we do the surgery
            for i in range(self.backbone.decoder.num_layers): 
                layer_name=f'block_{i}'
                blk = getattr(self.backbone.decoder, layer_name)

                w_q_linear = blk.attenion.fn.to_q
                w_v_linear = blk.attenion.fn.to_kv
                w_out_linear = blk.attenion.fn.to_out
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                w_a_linear_kv = nn.Linear(dim, r, bias=False)
                w_b_linear_kv = nn.Linear(r, dim*2, bias=False)
                w_a_linear_out = nn.Linear(dim, r, bias=False)
                w_b_linear_out = nn.Linear(r, dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_kv)
                self.w_Bs.append(w_b_linear_kv)
                self.w_As.append(w_a_linear_out)
                self.w_Bs.append(w_b_linear_out)
                blk.attenion.fn.to_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q, r, alpha)
                blk.attenion.fn.to_kv = _LoRALayer(w_v_linear, w_a_linear_kv, w_b_linear_kv, r, alpha)
                blk.attenion.fn.to_out = _LoRALayer(w_out_linear, w_a_linear_out, w_b_linear_out, r, alpha)

            self.reset_parameters()


    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

        # demo_layer= getattr(self.backbone.decoder, 'block_1')
        # demo_layer = demo_layer.attenion.fn.to_q
        # for name, param in demo_layer.named_parameters():
        #     print(name, param)
        # has_grad = any(param.requires_grad for param in demo_layer.parameters())
        # if has_grad:
        #     print(f"Layer has gradients enabled.")
        # else:
        #     print(f"Layer does not have gradients enabled.")
        # assert False

    def forward(self, batch):
        pass

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        pass

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int=0) -> torch.Tensor:
        if self.valid_grad:
            torch.set_grad_enabled(True)
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
        if self.valid_grad:
            torch.set_grad_enabled(True)
        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss


    def compute_stats(self, output_dict, prefix):
        pass



class CrystPredictiveFinetuneModel(CrystFinetuneModel):
    def __init__(self, *args, **kwargs) -> None:
        kwargs['diffuse']=True
        super().__init__(*args, **kwargs)

        self.valid_grad = False
        self.dataset=self.hparams['data']['root_path'].split('/')[-1]
        print('************ dataset: ', self.dataset)

        ### need to change to feature_extractor.out_feat_dim
        feat_dim = self.backbone.decoder.hidden_dim

        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, 1),
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
        time_emb_zeros = self.backbone.time_embedding(torch.zeros(batch_size, device=self.device))
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        input_frac_coords = frac_coords
        input_lattice = lattices

        node_rep, graph_rep = self.backbone.decoder(time_emb_zeros, batch.atom_types, input_frac_coords, \
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


class CrystGenerativeFinetuneModel(CrystFinetuneModel):
    def __init__(self, *args, **kwargs) -> None:
        kwargs['diffuse']=True
        super().__init__(*args, **kwargs)
        self.backbone.hparams.energy_guidance=False
        self.backbone.hparams.guidance_mode=None
        self.valid_grad=False
        for param in self.backbone.decoder.scalar_out.parameters():
            param.requires_grad = False
        
        # for param in self.backbone.decoder.parameters():
        #     param.requires_grad = False
        # for param in self.backbone.decoder.block_9.parameters():
        #     param.requires_grad = True
        # for param in self.backbone.decoder.block_10.parameters():
        #     param.requires_grad = True
        # for param in self.backbone.decoder.block_11.parameters():
        #     param.requires_grad = True
        # for param in self.backbone.decoder.coord_out.parameters():
        #     param.requires_grad = True
        # for param in self.backbone.decoder.lattice_out.parameters():
        #     param.requires_grad = True
        

    def forward(self, batch):
        return self.backbone(batch, stable_check=False)

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):
        return self.backbone.sample(batch, step_lr)

    @torch.no_grad()
    def pred_energy(self, batch):
        return self.backbone.pred_energy(batch)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_atom = output_dict['loss_atom']
        loss_scalar = output_dict['loss_scalar']
        loss = output_dict['loss']


        # print(loss.item(), loss_lattice.item(), loss_coord.item(), loss_scalar.item())
        self.log_dict(
            {
                'train_loss': loss,
                'lattice_loss': loss_lattice,
                'coord_loss': loss_coord,
                'loss_atom': loss_atom,
                'loss_scalar': loss_scalar,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # if loss.isnan():
        #     return None

        return loss


    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_atom = output_dict['loss_atom']
        loss_scalar = output_dict['loss_scalar']
        loss = output_dict['loss']

        log_dict = {
                f'{prefix}_loss': loss,
                f'{prefix}_lattice_loss': loss_lattice,
                f'{prefix}_coord_loss': loss_coord,
                f'{prefix}_loss_atom': loss_atom,
                f'{prefix}_loss_scalar': loss_scalar,
            }

        return log_dict, loss