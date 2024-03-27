import torch
import torch.nn as nn
from torch_geometric.nn import Set2Set


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.sp = nn.Softplus()
        self.shift = nn.Parameter(torch.log(torch.tensor([2.])), requires_grad=False)

    def forward(self, x):
        return self.sp(x) - self.shift


class MegBlock(nn.Module):
    def __init__(self,
                 edge_input_size,  # The dimension of edge features
                 node_input_size,  # The dimension of node features
                 state_input_size,  # The dimension of state features
                 inner_skip=False,  # use inner or outer skip connection
                 hidden_size=32,  # The dimension of hidden features
                 pool_method='sum'  # Pooling operation
                 ):
        super(MegBlock, self).__init__()
        self.inner_skip = inner_skip
        assert pool_method in ["mean", "sum"], 'pool_method only can be "mean" or "sum"'
        self.pool_method = pool_method

        self.preprocess_e = nn.Sequential(nn.Linear(edge_input_size, 2 * hidden_size),
                                          ShiftedSoftplus(),
                                          nn.Linear(2 * hidden_size, hidden_size),
                                          ShiftedSoftplus())

        self.preprocess_v = nn.Sequential(nn.Linear(node_input_size, 2 * hidden_size),
                                          ShiftedSoftplus(),
                                          nn.Linear(2 * hidden_size, hidden_size),
                                          ShiftedSoftplus())

        self.preprocess_u = nn.Sequential(nn.Linear(state_input_size, 2 * hidden_size),
                                          ShiftedSoftplus(),
                                          nn.Linear(2 * hidden_size, hidden_size),
                                          ShiftedSoftplus())

        self.phi_e_n = nn.Sequential(nn.Linear(4 * hidden_size, 2 * hidden_size),
                                     ShiftedSoftplus(),
                                     nn.Linear(2 * hidden_size, 2 * hidden_size),
                                     ShiftedSoftplus(),
                                     nn.Linear(2 * hidden_size, hidden_size),
                                     ShiftedSoftplus())

        self.phi_v_n = nn.Sequential(nn.Linear(3 * hidden_size, 2 * hidden_size),
                                     ShiftedSoftplus(),
                                     nn.Linear(2 * hidden_size, 2 * hidden_size),
                                     ShiftedSoftplus(),
                                     nn.Linear(2 * hidden_size, hidden_size),
                                     ShiftedSoftplus())

        self.phi_u_n = nn.Sequential(nn.Linear(3 * hidden_size, 2 * hidden_size),
                                     ShiftedSoftplus(),
                                     nn.Linear(2 * hidden_size, 2 * hidden_size),
                                     ShiftedSoftplus(),
                                     nn.Linear(2 * hidden_size, hidden_size),
                                     ShiftedSoftplus())

    def forward(self,
                nodes,  # Node features
                num_atoms,  # Number of atoms in each sample
                node_index,  # The sample to which each node belongs
                state,  # state features
                max_num_nbrs,  # Maximum number of neighbors
                num_pairs,  # Number of edges in each sample
                edge_index,  # The sample to which each edge belongs
                index,  # The constituent atom index of each edge
                edges  # Edge features
                ):
        if self.inner_skip:
            e = self.preprocess_e(edges)
            v = self.preprocess_v(nodes)
            u = self.preprocess_u(state)

            e_skip = e
            v_skip = v
            u_skip = u
        else:
            e_skip = edges
            v_skip = nodes
            u_skip = state

            e = self.preprocess_e(edges)
            v = self.preprocess_v(nodes)
            u = self.preprocess_u(state)

        # Step 1: Update edge attributes      ek' = phi_e(ek, vrk, vsk, u)
        nodes_center = v[index[:, 0]]
        nodes_nbr = v[index[:, 1]]
        u_e = torch.repeat_interleave(u, num_pairs, dim=0)
        concated_e = torch.cat([nodes_center, e, nodes_nbr, u_e], dim=-1)
        e_p = self.phi_e_n(concated_e)

        # Step 2: Aggregate edge attributes per node   Ei' = {(ek', rk, sk)} with rk =i, k=1:Ne
        e_p_av = torch.zeros_like(v)
        e_p_av.index_add_(0, index[:, 0], e_p)
        if self.pool_method == "mean":
            e_p_av /= max_num_nbrs

        # Step 3: Update node attributes  v_i' = phi_v(\bar e_i, vi, u)
        u_v = torch.repeat_interleave(u, num_atoms, dim=0)
        concated_v = torch.cat([e_p_av, v, u_v], -1)
        v_p = self.phi_v_n(concated_v)

        # Step 4: Aggregate edge attributes globally   E' = {(e_k', rk, sk)} k = 1:Ne  \bar e' = rho_e_u(E')
        e_p_au = torch.zeros_like(u)
        e_p_au.index_add_(0, edge_index, e_p)
        if self.pool_method == "mean":
            e_p_au /= num_pairs.unsqueeze(-1)

        # Step 5: Aggregate node attributes globally   V' = {v'} i = 1:Nv  \bar v' = rho_v_u(V')
        v_p_au = torch.zeros_like(u)
        v_p_au.index_add_(0, node_index, v_p)
        if self.pool_method == "mean":
            v_p_au /= num_atoms.unsqueeze(-1)

        # Step 6: Update global attribute   u' = phi_u(\bar e', \bar v', u)
        concated_u = torch.cat([e_p_au, v_p_au, u], -1)
        u_p = self.phi_u_n(concated_u)

        return e_skip + e_p, v_skip + v_p, u_skip + u_p


class Model(nn.Module):
    def __init__(self,
                 cutoff: float = 5.0,  # Cutoff radius
                 edge_embedding_size: int = 16,  # Edge embedding dimension
                 node_input_size: int = 13,  # Node input dimension
                 node_embedding_size: int = 16,  # Node embedding dimension
                 state_input_size: int = 1,  # State input dimension
                 inner_skip=False,  # use inner or outer skip connection
                 hidden_size: int = 32,  # Hidden dimension
                 n_blocks: int = 3,  # The number of MEGNet block
                 pool_method='sum',  # Pooling operation
                 n_set2set: int = 1,  # processing_steps of Set2Set
                 normalization: bool = False,  # Whether normalize
                 target_mean=0.0,
                 target_stddev=1.0
                 ):
        super(Model, self).__init__()
        self.cutoff = cutoff
        self.edge_embedding_size = edge_embedding_size

        # Setup atom embeddings
        # self.atom_embedding = nn.Embedding(95, node_embedding_size)
        self.atom_embedding = nn.Linear(node_input_size, node_embedding_size)

        self.megblocks = nn.ModuleList([MegBlock(2*edge_embedding_size, node_embedding_size, state_input_size,
                                                 inner_skip, hidden_size, pool_method)])
        assert n_blocks >= 1, "n_blocks must >= 1"
        for i in range(n_blocks - 1):
            self.megblocks.append(MegBlock(hidden_size, hidden_size, hidden_size,
                                           inner_skip, hidden_size, pool_method))

        self.se = Set2Set(hidden_size, n_set2set)
        self.sv = Set2Set(in_channels=hidden_size, processing_steps=n_set2set)

        self.readout = nn.Sequential(nn.Linear(5 * hidden_size, hidden_size),
                                     ShiftedSoftplus(),
                                     nn.Linear(hidden_size, hidden_size // 2),
                                     ShiftedSoftplus(),
                                     nn.Linear(hidden_size // 2, 1))

        self.register_buffer("normalization", torch.tensor(normalization))
        self.register_buffer("normalize_mean", torch.tensor(target_mean))
        self.register_buffer("normalize_stddev", torch.tensor(target_stddev))

    def forward(self, input_dict):
        nodes = self.atom_embedding(input_dict['atoms_embed'])

        num_atoms = input_dict['num_atoms']

        node_index = torch.arange(num_atoms.shape[0], device=num_atoms.device)
        node_index = torch.repeat_interleave(node_index, num_atoms)

        state = input_dict['state'].unsqueeze(-1)

        max_num_nbrs = input_dict['max_num_nbrs']

        num_pairs = input_dict['num_pairs']

        edge_index = torch.arange(num_pairs.shape[0], device=num_pairs.device)
        edge_index = torch.repeat_interleave(edge_index, num_pairs)

        index = input_dict['index']

        bonds = input_dict['nbrs_fea']
        ches, vdws = torch.split(bonds, 1, dim=1)
        ches = torch.where(ches < self.cutoff,
                           torch.sin(ches
                                     * (torch.arange(self.edge_embedding_size, device=ches.device) + 1)
                                     * torch.pi
                                     / self.cutoff)
                           / ches,
                           torch.tensor(0.0, device=ches.device, dtype=ches.dtype))
        vdws = torch.where(vdws < self.cutoff,
                           torch.sin(vdws
                                     * (torch.arange(self.edge_embedding_size, device=vdws.device) + 1)
                                     * torch.pi
                                     / self.cutoff)
                           / vdws,
                           torch.tensor(0.0, device=vdws.device, dtype=vdws.dtype))
        ches = ches * 0.5 * (torch.cos(torch.pi * ches / self.cutoff) + 1)
        vdws = vdws * 0.5 * (torch.cos(torch.pi * vdws / self.cutoff) + 1)
        edges = torch.cat([ches, vdws], 1)

        for block in self.megblocks:
            edges, nodes, state = block(nodes, num_atoms, node_index, state,
                                        max_num_nbrs, num_pairs, edge_index, index, edges)

        edges = self.se(edges, edge_index)
        nodes = self.sv(nodes, node_index)

        evu = torch.cat((edges, nodes, state), -1)
        predict = self.readout(evu)
        predict.squeeze_()

        # de-normalization
        if self.normalization:
            predict *= self.normalize_stddev
            predict += self.normalize_mean

        return predict
