import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from diffcsp.common.utils import CGCNN_LIST, exists
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from einops import rearrange, repeat

from diffcsp.common.data_utils import lattice_params_to_matrix_torch, get_pbc_distances, radius_graph_pbc, frac_to_cart_coords, repeat_blocks
from diffcsp.common.utils import SinusoidsEmbedding, LayerNorm

MAX_ATOMIC_NUM=100
EPS=1e-5
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

List = nn.ModuleList

# normalizations

class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn
    ):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args,**kwargs)


class ResidueNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn,
        norm_mode='pre',
    ):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.residue = GatedResidual(dim)
        self.norm_mode=norm_mode

    def forward(self, x, *args, **kwargs):
        if self.norm_mode == 'pre':
            normed_x = self.norm(x)
            x_ = self.fn(normed_x, *args,**kwargs)
            return self.residue(x, x_)
        
        elif self.norm_mode == 'post':
            x_ = self.fn(x, *args,**kwargs)
            out = self.residue(x, x_)
            return self.norm(out)
        
        return x + self.fn(x, *args,**kwargs)


# gated residual
class Residual(nn.Module):
    def forward(self, x, res):
        return x + res

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim = -1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)

# attention

class AttentionLayer(nn.Module):
    """ Message passing layer for cspnet."""

    def __init__(
        self,
        hidden_dim=128,
        head_num=8,
        dis_emb=None,
        ip=True,
        norm_edge=True,
    ):
        super(AttentionLayer, self).__init__()

        assert hidden_dim % head_num == 0

        self.head = head_num
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_dim = self.dis_dim + 9
        self.norm_edge=norm_edge

        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_kv = nn.Linear(hidden_dim, hidden_dim * 2)
        self.edges_to_kv = nn.Linear(self.edge_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        self.edge_norm = nn.LayerNorm(self.edge_dim)

        # self.edge_norm = nn.LayerNorm(self.edge_dim) if self.norm_edge else nn.Identity()

    
    def get_edge_feats(self, frac_coords, lattices, edge_index, edge2graph, frac_diff = None, norm_lattice_ip=False):
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1,-2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        if norm_lattice_ip:
            lattice_ips_flatten = F.normalize(lattice_ips_flatten, dim=-1)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edge_feats = torch.cat([lattice_ips_flatten_edges, frac_diff], dim=1)
        return edge_feats

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff = None):
        edge_num = len(edge_index[0])

        edge_feats = self.get_edge_feats(frac_coords, lattices, edge_index, edge2graph, frac_diff, norm_lattice_ip=False)
        if self.norm_edge:
            edge_feats = self.edge_norm(edge_feats)
        edge_kv = self.edges_to_kv(edge_feats)

        q = self.to_q(node_features[edge_index[0]]).reshape(edge_num, self.head, -1)  # Q
        k, v = self.to_kv(node_features[edge_index[1]]).chunk(2, dim=-1)
        k, v, edge_kv = map(lambda t: t.reshape(edge_num, self.head, -1), (k, v, edge_kv))

        ek, ev = edge_kv, edge_kv
        k = k + ek
        v = v + ev

        qk = torch.sum(q * k, dim=-1, keepdim=True) / math.sqrt(q.shape[-1])
        logits = scatter_softmax(qk, edge_index[0], dim=0)
        agg = scatter(logits * v, edge_index[0], dim = 0, reduce='sum')

        agg = self.to_out(agg.flatten(-2))
        return agg


class FeedForwardLayer(nn.Module):

    def __init__(self, dim, ff_mult = 4) -> None:
        super(FeedForwardLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        head_num=8,
        act_fn=nn.SiLU(),
        dis_emb=None,
        ip=True,
        norm_edge=True,
    ):
        
        super(TransformerBlock, self).__init__()
        self.act_fn = act_fn
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim

        assert hidden_dim % head_num == 0
        # self.attenion = PreNorm(hidden_dim, AttentionLayer(hidden_dim=hidden_dim, dis_emb=self.dis_emb, \
        #                                head_num=head_num, ip=ip))
        # self.ffn = PreNorm(hidden_dim, FeedForwardLayer(dim=hidden_dim))
        self.attention = ResidueNorm(hidden_dim, AttentionLayer(hidden_dim=hidden_dim, dis_emb=self.dis_emb, \
                                       head_num=head_num, ip=ip, norm_edge=norm_edge))
        self.ffn = ResidueNorm(hidden_dim, FeedForwardLayer(dim=hidden_dim))

        # self.attn_residue = GatedResidual(dim=hidden_dim)
        # self.ffn_residue = GatedResidual(dim=hidden_dim)
        # self.attn_residue = Residual()
        # self.ffn_residue = Residual()


    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff = None):
        # attn_out = self.attenion(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff)
        # attn_out = self.attn_residue(attn_out, node_features)

        # ffn_out = self.ffn(attn_out)
        # node_out = self.ffn_residue(ffn_out, attn_out)

        attn_out = self.attention(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff)

        node_out = self.ffn(attn_out)

        return node_out


class CrysFormer(nn.Module):
    def __init__(
        self,
        hidden_dim = 256,
        latent_dim = 256,
        num_layers = 4,
        max_atoms = 100,
        act_fn = 'silu',
        dis_emb = 'sin',
        num_freqs = 10,
        edge_style = 'fc',
        cutoff = 6.0,
        head_num = 8,
        max_neighbors = 20,
        diffuse= False,
        ln=False,
        ip = True,
        smooth = False,
        norm_edge=True
    ):
        super(CrysFormer, self).__init__()

        self.ip = ip
        self.smooth = smooth
        self.hidden_dim = hidden_dim
        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            ######## here + 1, for mask nodes
            # self.node_embedding = nn.Embedding(max_atoms+1, hidden_dim)
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        
        self.embedding_in = nn.Sequential(
            nn.Embedding(101, 92),
            nn.Linear(92, hidden_dim),
        )
        self.embedding_in[0].weight.data.copy_(torch.tensor(CGCNN_LIST))  # CGCNN Embedding

        for param in self.embedding_in[0].parameters():
            param.requires_grad = False

        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies = num_freqs)
        elif dis_emb == 'none':
            self.dis_emb = None
        for i in range(0, num_layers):
            self.add_module(
                "block_%d" % i, TransformerBlock(hidden_dim, head_num, self.act_fn, self.dis_emb, ip=ip, norm_edge=norm_edge)
            )            
        self.num_layers = num_layers
        self.coord_out = nn.Linear(hidden_dim, 3, bias = False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias = False)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.ln = ln
        self.diffuse = diffuse
        
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        
        # self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)
        # self.scalar_out = nn.Linear(hidden_dim, 1)

        self.type_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, MAX_ATOMIC_NUM)
        )
 
        self.scalar_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
       

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self, edge_index, cell_offsets, neighbors, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_vector_new,
        )

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):

        if self.edge_style == 'fc':
            lis = [torch.ones(n,n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)

            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.
        elif self.edge_style == 'knn':
            lattice_nodes = lattices[node2graph]
            cart_coords = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)
            
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords, None, None, num_atoms, self.cutoff, self.max_neighbors,
                device=num_atoms.device, lattices=lattices)

            j_index, i_index = edge_index
            distance_vectors = frac_coords[j_index] - frac_coords[i_index]
            distance_vectors += to_jimages.float()

            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(edge_index, to_jimages, num_bonds, distance_vectors)

            return edge_index_new, -edge_vector_new
    

    def forward(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph, only_rep=False):
        # print(self.embedding_in[0].weight)

        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        
        # node_features = self.node_embedding(atom_types)
        node_features = self.embedding_in(atom_types)

        if t is not None:
            t_per_atom = t.repeat_interleave(num_atoms, dim=0)

            node_features = torch.cat([node_features, t_per_atom], dim=1)
            node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            node_features = self._modules["block_%d" % i](node_features, frac_coords, lattices, edges, edge2graph, frac_diff = frac_diff)

        if self.ln:
            node_features = self.final_layer_norm(node_features)
        
        coords = self.coord_out(node_features)

        graph_features = scatter(node_features, node2graph, dim = 0, reduce = 'mean')

        # print(graph_features.shape)
        # assert False
        
        if only_rep:
            return node_features, graph_features

        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.view(-1, 3, 3)
        if self.ip:
            lattice_out = torch.einsum('bij,bjk->bik', lattice_out, lattices)
    
        type_out = self.type_out(node_features) 
        scalar_out = self.scalar_out(graph_features)

        return lattice_out, coords, node_features, graph_features, type_out, scalar_out