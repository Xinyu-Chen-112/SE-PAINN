import torch
from torch import nn


class PainnMessage(nn.Module):
    def __init__(self,
                 edge_embedding_size: int = 20,  # Edge embedding dimension
                 hidden_size: int = 128  # Hidden dimension
                 ):
        super(PainnMessage, self).__init__()
        self.hidden_size = hidden_size

        self.scalar_message_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                nn.PReLU(),
                                                nn.Linear(hidden_size, 3 * hidden_size),
                                                nn.PReLU())

        self.filter_layer = nn.Sequential(nn.Linear(2 * edge_embedding_size, hidden_size),
                                          nn.PReLU(),
                                          nn.Linear(hidden_size, 3 * hidden_size),
                                          nn.PReLU())

    def forward(self,
                node_scalar,  # Scalar node features
                node_vector,  # Vector node features
                edge,  # The constituent atom index of each edge
                edge_diff,  # Vector edge features
                edge_dist,  # Scalar edge features
                rbf_dist  # Edge features after radial basis function expansion
                ):
        scalar_out = self.scalar_message_mlp(node_scalar)

        filter_weight = self.filter_layer(rbf_dist)

        filter_out = scalar_out[edge[:, 1]] * filter_weight
        gate_state_vector, message_scalar, gate_edge_vector = torch.split(filter_out, self.hidden_size, dim=1)

        state_vector = node_vector[edge[:, 1]] * gate_state_vector.unsqueeze(1)
        edge_vector = (edge_diff / edge_dist.unsqueeze(-1)).unsqueeze(-1) * gate_edge_vector.unsqueeze(1)
        message_vector = state_vector + edge_vector

        residual_scalar = torch.zeros_like(node_scalar)
        residual_scalar.index_add_(0, edge[:, 0], message_scalar)
        residual_vector = torch.zeros_like(node_vector)
        residual_vector.index_add_(0, edge[:, 0], message_vector)

        messaged_node_scalar = node_scalar + residual_scalar
        messaged_node_vector = node_vector + residual_vector

        return messaged_node_scalar, messaged_node_vector


class PainnUpdate(nn.Module):
    def __init__(self,
                 hidden_size: int = 128  # Hidden dimension
                 ):
        super(PainnUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.update_u = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.PReLU())
        self.update_v = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.PReLU())

        self.update_mlp = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size),
                                        nn.PReLU(),
                                        nn.Linear(hidden_size, 3 * hidden_size),
                                        nn.PReLU())

    def forward(self,
                node_scalar,  # Scalar node features
                node_vector,  # Vector node features
                ):
        u = self.update_u(node_vector)
        v = self.update_v(node_vector)

        v_norm = torch.linalg.norm(v, dim=1)
        mlp_input = torch.cat((v_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)

        a_vv, a_sv, a_ss = torch.split(mlp_output, self.hidden_size, dim=1)

        delta_v = u * a_vv.unsqueeze(1)

        inner_prod = torch.sum(u * v, dim=1)
        delta_s = inner_prod * a_sv + a_ss

        updated_node_scalar = node_scalar + delta_s
        updated_node_vector = node_vector + delta_v

        return updated_node_scalar, updated_node_vector


class Model(nn.Module):
    def __init__(self,
                 num_interactions: int = 3,  # The number of PAINN layer
                 cutoff: float = 5.0,  # Cutoff radius
                 node_input_size: int = 13,  # Node input dimension
                 edge_embedding_size: int = 20,  # Edge embedding dimension
                 hidden_size: int = 128,  # Hidden dimension
                 normalization: bool = False,  # Whether normalize
                 atomwise_normalization: bool = False,  # Whether average to each atom
                 target_mean=0.0,
                 target_stddev=1.0):
        super(Model, self).__init__()
        self.cutoff = cutoff
        self.edge_embedding_size = edge_embedding_size
        self.hidden_size = hidden_size

        # Setup atom embeddings
        # self.atom_embedding = nn.Embedding(84, hidden_size)
        self.atom_embedding = nn.Sequential(nn.Linear(node_input_size, 64),
                                            nn.PReLU(),
                                            nn.Linear(64, hidden_size),
                                            nn.PReLU())

        self.message_layers = nn.ModuleList([PainnMessage(edge_embedding_size, hidden_size)
                                             for _ in range(num_interactions)])
        self.update_layers = nn.ModuleList([PainnUpdate(hidden_size)
                                            for _ in range(num_interactions)])

        # Setup readout function
        self.readout_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                         nn.PReLU(),
                                         nn.Linear(hidden_size, hidden_size),
                                         nn.PReLU(),
                                         nn.Linear(hidden_size, 64),
                                         nn.PReLU(),
                                         nn.Linear(64, 1))

        self.register_buffer("normalization", torch.tensor(normalization))
        self.register_buffer("atomwise_normalization", torch.tensor(atomwise_normalization))
        self.register_buffer("normalize_mean", torch.tensor(target_mean))
        self.register_buffer("normalize_stddev", torch.tensor(target_stddev))

    def forward(self, input_dict):
        num_atoms = input_dict['num_atoms']
        atoms = input_dict['atoms']

        edge = input_dict['pairs']
        edge_diff = input_dict['diff']
        edge_dist = input_dict['distance']
        node_scalar = self.atom_embedding(input_dict['atoms_embed'])
        node_vector = torch.zeros((atoms.shape[0], 3, self.hidden_size),
                                  device=node_scalar.device, dtype=node_scalar.dtype)

        bonds = input_dict['bonds']
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
        rbf_dist = torch.cat([ches, vdws], 1)

        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            node_scalar, node_vector = message_layer(node_scalar, node_vector,
                                                     edge, edge_diff, edge_dist, rbf_dist)
            node_scalar, node_vector = update_layer(node_scalar, node_vector)

        node_scalar = self.readout_mlp(node_scalar)
        node_scalar.squeeze_()

        node_idx = torch.arange(num_atoms.shape[0], device=node_scalar.device)
        node_idx = torch.repeat_interleave(node_idx, num_atoms)

        predict = torch.zeros_like(num_atoms, dtype=torch.float32)
        predict.index_add_(0, node_idx, node_scalar)

        if self.atomwise_normalization:
            predict /= num_atoms

        # de-normalization
        if self.normalization:
            predict *= self.normalize_stddev
            predict += self.normalize_mean

        return predict
