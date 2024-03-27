import torch
from torch import nn


class PainnMessage(nn.Module):
    def __init__(self,
                 edge_embedding_size: int = 20,  # Edge embedding dimension
                 hidden_size: int = 128  # Hidden dimension
                 ):
        super(PainnMessage, self).__init__()
        self.hidden_size = hidden_size

        self.che_scalar_message_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                    nn.PReLU(),
                                                    nn.Linear(hidden_size, 3 * hidden_size),
                                                    nn.PReLU())
        self.che_filter_layer = nn.Sequential(nn.Linear(edge_embedding_size, hidden_size),
                                              nn.PReLU(),
                                              nn.Linear(hidden_size, 3 * hidden_size),
                                              nn.PReLU())

        self.vdw_scalar_message_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                    nn.PReLU(),
                                                    nn.Linear(hidden_size, 3 * hidden_size),
                                                    nn.PReLU())
        self.vdw_filter_layer = nn.Sequential(nn.Linear(edge_embedding_size, hidden_size),
                                              nn.PReLU(),
                                              nn.Linear(hidden_size, 3 * hidden_size),
                                              nn.PReLU())

    def forward(self,
                node_scalar,  # Scalar node features
                node_vector,  # Vector node features
                che_edge,  # The constituent atom index of each intralayer edge
                che_edge_diff,  # Vector intralayer edge features
                che_edge_dist,  # Scalar intralayer edge features
                che_rbf_dist,  # Intralayer edge features after radial basis function expansion
                vdw_edge,  # The constituent atom index of each interlayer edge
                vdw_edge_diff,  # Vector interlayer edge features
                vdw_edge_dist,  # Scalar interlayer edge features
                vdw_rbf_dist  # Interlayer edge features after radial basis function expansion
                ):
        # Intralayer interaction
        che_scalar_out = self.che_scalar_message_mlp(node_scalar)
        che_filter_weight = self.che_filter_layer(che_rbf_dist)
        che_filter_out = che_scalar_out[che_edge[:, 1]] * che_filter_weight
        che_gate_vector, che_message_scalar, che_gate_edge = torch.split(che_filter_out, self.hidden_size, dim=1)
        che_state_vector = node_vector[che_edge[:, 1]] * che_gate_vector.unsqueeze(1)
        che_edge_vector = (che_edge_diff / che_edge_dist.unsqueeze(-1)).unsqueeze(-1) * che_gate_edge.unsqueeze(1)
        che_message_vector = che_state_vector + che_edge_vector
        che_residual_scalar = torch.zeros_like(node_scalar)
        che_residual_scalar.index_add_(0, che_edge[:, 0], che_message_scalar)
        che_residual_vector = torch.zeros_like(node_vector)
        che_residual_vector.index_add_(0, che_edge[:, 0], che_message_vector)

        # Interlayer interaction
        vdw_scalar_out = self.vdw_scalar_message_mlp(node_scalar)
        vdw_filter_weight = self.vdw_filter_layer(vdw_rbf_dist)
        vdw_filter_out = vdw_scalar_out[vdw_edge[:, 1]] * vdw_filter_weight
        vdw_gate_vector, vdw_message_scalar, vdw_gate_edge = torch.split(vdw_filter_out, self.hidden_size, dim=1)
        vdw_state_vector = node_vector[vdw_edge[:, 1]] * vdw_gate_vector.unsqueeze(1)
        vdw_edge_vector = (vdw_edge_diff / vdw_edge_dist.unsqueeze(-1)).unsqueeze(-1) * vdw_gate_edge.unsqueeze(1)
        vdw_message_vector = vdw_state_vector + vdw_edge_vector
        vdw_residual_scalar = torch.zeros_like(node_scalar)
        vdw_residual_scalar.index_add_(0, vdw_edge[:, 0], vdw_message_scalar)
        vdw_residual_vector = torch.zeros_like(node_vector)
        vdw_residual_vector.index_add_(0, vdw_edge[:, 0], vdw_message_vector)

        messaged_node_scalar = node_scalar + che_residual_scalar + vdw_residual_scalar
        messaged_node_vector = node_vector + che_residual_vector + vdw_residual_vector
        
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
                 che_cutoff: float = 5.0,  # Intralayer cutoff radius
                 vdw_cutoff: float = 5.0,  # Interlayer cutoff radius
                 node_input_size: int = 13,  # Node input dimension
                 edge_embedding_size: int = 20,  # Edge embedding dimension
                 hidden_size: int = 128,  # Hidden dimension
                 normalization: bool = False,  # Whether normalize
                 atomwise_normalization: bool = False,  # Whether average to each atom
                 target_mean=0.0,
                 target_stddev=1.0):
        super(Model, self).__init__()
        self.che_cutoff = che_cutoff
        self.vdw_cutoff = vdw_cutoff
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

        che_edge = input_dict['che_pairs']
        che_edge_diff = input_dict['che_diff']
        che_edge_dist = input_dict['che_distance']
        vdw_edge = input_dict['vdw_pairs']
        vdw_edge_diff = input_dict['vdw_diff']
        vdw_edge_dist = input_dict['vdw_distance']

        node_scalar = self.atom_embedding(input_dict['atoms_embed'])
        node_vector = torch.zeros((atoms.shape[0], 3, self.hidden_size),
                                  device=node_scalar.device, dtype=node_scalar.dtype)

        che_rbf_dist = torch.where((che_edge_dist < self.che_cutoff).unsqueeze(-1),
                                   torch.sin(che_edge_dist.unsqueeze(-1)
                                             * (torch.arange(self.edge_embedding_size, device=che_edge_dist.device) + 1)
                                             * torch.pi
                                             / self.che_cutoff)
                                   / che_edge_dist.unsqueeze(-1),
                                   torch.tensor(0.0, device=che_edge_dist.device, dtype=che_edge_dist.dtype))
        vdw_rbf_dist = torch.where((vdw_edge_dist < self.vdw_cutoff).unsqueeze(-1),
                                   torch.sin(vdw_edge_dist.unsqueeze(-1)
                                             * (torch.arange(self.edge_embedding_size, device=vdw_edge_dist.device) + 1)
                                             * torch.pi
                                             / self.vdw_cutoff)
                                   / vdw_edge_dist.unsqueeze(-1),
                                   torch.tensor(0.0, device=vdw_edge_dist.device, dtype=vdw_edge_dist.dtype))
        che_rbf_dist = che_rbf_dist * 0.5 * (torch.cos(torch.pi * che_edge_dist / self.che_cutoff) + 1).unsqueeze(-1)
        vdw_rbf_dist = vdw_rbf_dist * 0.5 * (torch.cos(torch.pi * vdw_edge_dist / self.vdw_cutoff) + 1).unsqueeze(-1)

        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            node_scalar, node_vector = message_layer(node_scalar, node_vector,
                                                     che_edge, che_edge_diff, che_edge_dist, che_rbf_dist,
                                                     vdw_edge, vdw_edge_diff, vdw_edge_dist, vdw_rbf_dist)
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
