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

        self.preprocess_e_che = nn.Sequential(nn.Linear(edge_input_size, 2 * hidden_size),
                                              ShiftedSoftplus(),
                                              nn.Linear(2 * hidden_size, hidden_size),
                                              ShiftedSoftplus())
        self.preprocess_e_vdw = nn.Sequential(nn.Linear(edge_input_size, 2 * hidden_size),
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

        self.phi_e_che_n = nn.Sequential(nn.Linear(4 * hidden_size, 2 * hidden_size),
                                         ShiftedSoftplus(),
                                         nn.Linear(2 * hidden_size, 2 * hidden_size),
                                         ShiftedSoftplus(),
                                         nn.Linear(2 * hidden_size, hidden_size),
                                         ShiftedSoftplus())
        self.phi_e_vdw_n = nn.Sequential(nn.Linear(4 * hidden_size, 2 * hidden_size),
                                         ShiftedSoftplus(),
                                         nn.Linear(2 * hidden_size, 2 * hidden_size),
                                         ShiftedSoftplus(),
                                         nn.Linear(2 * hidden_size, hidden_size),
                                         ShiftedSoftplus())

        self.phi_v_che_n = nn.Sequential(nn.Linear(3 * hidden_size, 2 * hidden_size),
                                         ShiftedSoftplus(),
                                         nn.Linear(2 * hidden_size, 2 * hidden_size),
                                         ShiftedSoftplus(),
                                         nn.Linear(2 * hidden_size, hidden_size),
                                         ShiftedSoftplus())

        self.phi_v_vdw_n = nn.Sequential(nn.Linear(3 * hidden_size, 2 * hidden_size),
                                         ShiftedSoftplus(),
                                         nn.Linear(2 * hidden_size, 2 * hidden_size),
                                         ShiftedSoftplus(),
                                         nn.Linear(2 * hidden_size, hidden_size),
                                         ShiftedSoftplus())

        self.phi_u_che_n = nn.Sequential(nn.Linear(3 * hidden_size, 2 * hidden_size),
                                         ShiftedSoftplus(),
                                         nn.Linear(2 * hidden_size, 2 * hidden_size),
                                         ShiftedSoftplus(),
                                         nn.Linear(2 * hidden_size, hidden_size),
                                         ShiftedSoftplus())

        self.phi_u_vdw_n = nn.Sequential(nn.Linear(3 * hidden_size, 2 * hidden_size),
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
                che_max_num_nbrs,  # Maximum number of intralayer neighbors
                che_num_pairs,  # Number of intralayer edges in each sample
                che_edge_index,  # The sample to which each intralayer edge belongs
                che_index,  # The constituent atom index of each intralayer edge
                che_edges,  # Intralayer edge features
                vdw_max_num_nbrs,  # Maximum number of interlayer neighbors
                vdw_num_pairs,  # Number of interlayer edges in each sample
                vdw_edge_index,  # The sample to which each interlayer edge belongs
                vdw_index,  # The constituent atom index of each interlayer edge
                vdw_edges  # Interlayer edge features
                ):
        if self.inner_skip:
            e_che = self.preprocess_e_che(che_edges)
            e_vdw = self.preprocess_e_vdw(vdw_edges)
            v = self.preprocess_v(nodes)
            u = self.preprocess_u(state)

            e_che_skip = e_che
            e_vdw_skip = e_vdw
            v_skip = v
            u_skip = u
        else:
            e_che_skip = che_edges
            e_vdw_skip = vdw_edges
            v_skip = nodes
            u_skip = state

            e_che = self.preprocess_e_che(che_edges)
            e_vdw = self.preprocess_e_vdw(vdw_edges)
            v = self.preprocess_v(nodes)
            u = self.preprocess_u(state)

        # Intralayer interaction
        # Step 1: update e
        che_nodes_center = v[che_index[:, 0]]
        che_nodes_nbr = v[che_index[:, 1]]
        che_u_e = torch.repeat_interleave(u, che_num_pairs, dim=0)
        che_concated = torch.cat([che_nodes_center, e_che, che_nodes_nbr, che_u_e], dim=-1)
        che_e_p = self.phi_e_che_n(che_concated)
        # Step 2: e_p aggregate to v
        che_e_v = torch.zeros_like(v)
        che_e_v.index_add_(0, che_index[:, 0], che_e_p)
        if self.pool_method == "mean":
            che_e_v /= che_max_num_nbrs
        # Step 3: update v
        che_u_v = torch.repeat_interleave(u, num_atoms, dim=0)
        che_concated_v = torch.cat([che_e_v, v, che_u_v], -1)
        che_v_p = self.phi_v_che_n(che_concated_v)
        # Step 4: e_p aggregate to u
        che_e_u = torch.zeros_like(u)
        che_e_u.index_add_(0, che_edge_index, che_e_p)
        if self.pool_method == "mean":
            che_e_u /= che_num_pairs.unsqueeze(-1)
        # Step 5: v_p aggregate to u
        che_v_u = torch.zeros_like(u)
        che_v_u.index_add_(0, node_index, che_v_p)
        if self.pool_method == "mean":
            che_v_u /= num_atoms.unsqueeze(-1)
        # Step 6: update u
        che_concated_u = torch.cat([che_e_u, che_v_u, u], -1)
        che_u_p = self.phi_u_che_n(che_concated_u)

        # Interlayer interaction
        # Step 1: update e
        vdw_nodes_center = v[vdw_index[:, 0]]
        vdw_nodes_nbr = v[vdw_index[:, 1]]
        vdw_u_e = torch.repeat_interleave(u, vdw_num_pairs, dim=0)
        vdw_concated = torch.cat([vdw_nodes_center, e_vdw, vdw_nodes_nbr, vdw_u_e], dim=-1)
        vdw_e_p = self.phi_e_vdw_n(vdw_concated)
        # Step 2: e_p aggregate to v
        vdw_e_v = torch.zeros_like(v)
        vdw_e_v.index_add_(0, vdw_index[:, 0], vdw_e_p)
        if self.pool_method == "mean":
            vdw_e_v /= vdw_max_num_nbrs
        # Step 3: update v
        vdw_u_v = torch.repeat_interleave(u, num_atoms, dim=0)
        vdw_concated_v = torch.cat([vdw_e_v, v, vdw_u_v], -1)
        vdw_v_p = self.phi_v_vdw_n(vdw_concated_v)
        # Step 4: e_p aggregate to u
        vdw_e_u = torch.zeros_like(u)
        vdw_e_u.index_add_(0, vdw_edge_index, vdw_e_p)
        if self.pool_method == "mean":
            vdw_e_u /= vdw_num_pairs.unsqueeze(-1)
        # Step 5: v_p aggregate to u
        vdw_v_u = torch.zeros_like(u)
        vdw_v_u.index_add_(0, node_index, vdw_v_p)
        if self.pool_method == "mean":
            vdw_v_u /= num_atoms.unsqueeze(-1)
        # Step 6: update u
        vdw_concated_u = torch.cat([vdw_e_u, vdw_v_u, u], -1)
        vdw_u_p = self.phi_u_vdw_n(vdw_concated_u)

        return e_che_skip + che_e_p, e_vdw_skip + vdw_e_p, v_skip + che_v_p + vdw_v_p, u_skip + che_u_p + vdw_u_p


class Model(nn.Module):
    def __init__(self,
                 che_cutoff: float = 5.0,  # Intralayer cutoff radius
                 vdw_cutoff: float = 5.0,  # Interlayer cutoff radius
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
        self.che_cutoff = che_cutoff
        self.vdw_cutoff = vdw_cutoff
        self.edge_embedding_size = edge_embedding_size

        # Setup atom embeddings
        # self.atom_embedding = nn.Embedding(95, node_embedding_size)
        self.atom_embedding = nn.Linear(node_input_size, node_embedding_size)

        self.megblocks = nn.ModuleList([MegBlock(edge_embedding_size, node_embedding_size, state_input_size,
                                                 inner_skip, hidden_size, pool_method)])
        assert n_blocks >= 1, "n_blocks must >= 1"
        for i in range(n_blocks - 1):
            self.megblocks.append(MegBlock(hidden_size, hidden_size, hidden_size,
                                           inner_skip, hidden_size, pool_method))

        self.se_che = Set2Set(hidden_size, n_set2set)
        self.se_vdw = Set2Set(hidden_size, n_set2set)
        self.sv = Set2Set(in_channels=hidden_size, processing_steps=n_set2set)

        self.readout = nn.Sequential(nn.Linear(7 * hidden_size, hidden_size),
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

        che_max_num_nbrs = input_dict['che_max_num_nbrs']
        vdw_max_num_nbrs = input_dict['vdw_max_num_nbrs']

        che_num_pairs = input_dict['che_num_pairs']
        vdw_num_pairs = input_dict['vdw_num_pairs']

        che_edge_index = torch.arange(che_num_pairs.shape[0], device=che_num_pairs.device)
        che_edge_index = torch.repeat_interleave(che_edge_index, che_num_pairs)
        vdw_edge_index = torch.arange(vdw_num_pairs.shape[0], device=vdw_num_pairs.device)
        vdw_edge_index = torch.repeat_interleave(vdw_edge_index, vdw_num_pairs)

        che_index = input_dict['che_index']
        vdw_index = input_dict['vdw_index']

        che_edge_dist = input_dict['che_nbrs_fea']
        vdw_edge_dist = input_dict['vdw_nbrs_fea']
        che_edges = torch.where((che_edge_dist < self.che_cutoff).unsqueeze(-1),
                                torch.sin(che_edge_dist.unsqueeze(-1)
                                          * (torch.arange(self.edge_embedding_size, device=che_edge_dist.device) + 1)
                                          * torch.pi
                                          / self.che_cutoff)
                                / che_edge_dist.unsqueeze(-1),
                                torch.tensor(0.0, device=che_edge_dist.device, dtype=che_edge_dist.dtype))
        vdw_edges = torch.where((vdw_edge_dist < self.vdw_cutoff).unsqueeze(-1),
                                torch.sin(vdw_edge_dist.unsqueeze(-1)
                                          * (torch.arange(self.edge_embedding_size, device=vdw_edge_dist.device) + 1)
                                          * torch.pi
                                          / self.vdw_cutoff)
                                / vdw_edge_dist.unsqueeze(-1),
                                torch.tensor(0.0, device=vdw_edge_dist.device, dtype=vdw_edge_dist.dtype))
        che_edges = che_edges * 0.5 * (torch.cos(torch.pi * che_edge_dist / self.che_cutoff) + 1).unsqueeze(-1)
        vdw_edges = vdw_edges * 0.5 * (torch.cos(torch.pi * vdw_edge_dist / self.vdw_cutoff) + 1).unsqueeze(-1)

        for block in self.megblocks:
            che_edges, vdw_edges, nodes, state = block(nodes, num_atoms, node_index, state,
                                                       che_max_num_nbrs, che_num_pairs, che_edge_index, che_index, che_edges,
                                                       vdw_max_num_nbrs, vdw_num_pairs, vdw_edge_index, vdw_index, vdw_edges)

        che_edges = self.se_che(che_edges, che_edge_index)
        vdw_edges = self.se_vdw(vdw_edges, vdw_edge_index)
        nodes = self.sv(nodes, node_index)

        evu = torch.cat((che_edges, vdw_edges, nodes, state), -1)
        predict = self.readout(evu)
        predict.squeeze_()

        # de-normalization
        if self.normalization:
            predict *= self.normalize_stddev
            predict += self.normalize_mean

        return predict
