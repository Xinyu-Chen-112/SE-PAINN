import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self,
                 edge_embedding_size: int = 20,  # Edge embedding dimension
                 hidden_size: int = 128  # Hidden dimension
                 ):
        super(ConvLayer, self).__init__()
        self.hidden_size = hidden_size

        self.che_filter_layer = nn.Linear(edge_embedding_size, hidden_size)
        self.che_fc_full = nn.Linear(3 * hidden_size, 2 * hidden_size)

        self.vdw_filter_layer = nn.Linear(edge_embedding_size, hidden_size)
        self.vdw_fc_full = nn.Linear(3 * hidden_size, 2 * hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self,
                nodes,  # Node features
                che_rbf_edges,  # Intralayer edge features after radial basis function expansion
                che_nbrs_idx,  # Intralayer neighbor index
                vdw_rbf_edges,  # Interlayer edge features after radial basis function expansion
                vdw_nbrs_idx,  # Interlayer neighbor index
                ):

        # Intralayer interaction
        che_N, che_M = che_nbrs_idx.shape
        # convolution
        che_edges = self.che_filter_layer(che_rbf_edges)
        che_nbrs_fea = nodes[che_nbrs_idx]
        che_total_fea = torch.cat([nodes.unsqueeze(1).expand(che_N, che_M, self.hidden_size),
                                   che_edges,
                                   che_nbrs_fea],
                                  dim=2)
        che_gated_fea = self.che_fc_full(che_total_fea)
        che_nbr_filter, che_nbr_core = torch.split(che_gated_fea, self.hidden_size, dim=2)
        che_nbr_filter = self.sigmoid(che_nbr_filter)
        che_nbr_core = self.softplus(che_nbr_core)
        che_nbr_aggred = torch.sum(che_nbr_filter * che_nbr_core, dim=1)

        # Interlayer interaction
        vdw_N, vdw_M = vdw_nbrs_idx.shape
        # convolution
        vdw_edges = self.vdw_filter_layer(vdw_rbf_edges)
        vdw_nbrs_fea = nodes[vdw_nbrs_idx]
        vdw_total_fea = torch.cat([nodes.unsqueeze(1).expand(vdw_N, vdw_M, self.hidden_size),
                                   vdw_edges,
                                   vdw_nbrs_fea],
                                  dim=2)
        vdw_gated_fea = self.vdw_fc_full(vdw_total_fea)
        vdw_nbr_filter, vdw_nbr_core = torch.split(vdw_gated_fea, self.hidden_size, dim=2)
        vdw_nbr_filter = self.sigmoid(vdw_nbr_filter)
        vdw_nbr_core = self.softplus(vdw_nbr_core)
        vdw_nbr_aggred = torch.sum(vdw_nbr_filter * vdw_nbr_core, dim=1)

        new_nodes = self.softplus(nodes + che_nbr_aggred + vdw_nbr_aggred)

        return new_nodes


class Model(nn.Module):
    def __init__(self,
                 che_cutoff,  # Intralayer cutoff radius
                 vdw_cutoff,  # Interlayer cutoff radius
                 node_input_size: int = 13,  # Original node dimension
                 edge_embedding_size: int = 20,  # Edge embedding dimension
                 hidden_size: int = 128,  # Hidden dimension
                 n_conv: int = 3,  # The number of convolutional layers
                 h_size: int = 128,  # Hidden dimension after pooling
                 n_h: int = 1,  # The number of convolutional layers after pooling
                 classification: bool = False,  # Is it a classification task
                 normalization: bool = False,  # Whether normalize
                 target_mean=0.0,
                 target_stddev=1.0
                 ):
        super(Model, self).__init__()
        self.che_cutoff = che_cutoff
        self.vdw_cutoff = vdw_cutoff
        self.edge_embedding_size = edge_embedding_size
        self.classification = classification

        self.embedding = nn.Linear(node_input_size, hidden_size)

        self.convs = nn.ModuleList([ConvLayer(edge_embedding_size=edge_embedding_size, hidden_size=hidden_size) for _ in range(n_conv)])

        self.conv_to_fc = nn.Linear(hidden_size, h_size)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_size, h_size)
                                      for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])

        if classification:
            self.dropout = nn.Dropout()
            self.fc_out = nn.Linear(h_size, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(h_size, 1)

        self.register_buffer("normalization", torch.tensor(normalization))
        self.register_buffer("normalize_mean", torch.tensor(target_mean))
        self.register_buffer("normalize_stddev", torch.tensor(target_stddev))

    def forward(self, input_dict):
        orig_nodes = input_dict['atoms_embed']
        nodes = self.embedding(orig_nodes)

        che_edge_dist = input_dict['che_nbrs_fea']
        vdw_edge_dist = input_dict['vdw_nbrs_fea']
        che_rbf_edges = torch.where((che_edge_dist < self.che_cutoff).unsqueeze(-1),
                                    torch.sin(che_edge_dist.unsqueeze(-1)
                                              * (torch.arange(self.edge_embedding_size, device=che_edge_dist.device) + 1)
                                              * torch.pi
                                              / self.che_cutoff)
                                    / che_edge_dist.unsqueeze(-1),
                                    torch.tensor(0.0, device=che_edge_dist.device, dtype=che_edge_dist.dtype))
        vdw_rbf_edges = torch.where((vdw_edge_dist < self.vdw_cutoff).unsqueeze(-1),
                                    torch.sin(vdw_edge_dist.unsqueeze(-1)
                                              * (torch.arange(self.edge_embedding_size, device=vdw_edge_dist.device) + 1)
                                              * torch.pi
                                              / self.vdw_cutoff)
                                    / vdw_edge_dist.unsqueeze(-1),
                                    torch.tensor(0.0, device=vdw_edge_dist.device, dtype=vdw_edge_dist.dtype))
        che_rbf_edges = che_rbf_edges * 0.5 * (torch.cos(torch.pi * che_edge_dist / self.che_cutoff) + 1).unsqueeze(-1)
        vdw_rbf_edges = vdw_rbf_edges * 0.5 * (torch.cos(torch.pi * vdw_edge_dist / self.vdw_cutoff) + 1).unsqueeze(-1)

        che_nbrs_idx = input_dict['che_nbrs_idx']
        vdw_nbrs_idx = input_dict['vdw_nbrs_idx']

        for conv_func in self.convs:
            nodes = conv_func(nodes,
                              che_rbf_edges, che_nbrs_idx,
                              vdw_rbf_edges, vdw_nbrs_idx)

        num_atoms = input_dict['num_atoms']
        starts = torch.cumsum(torch.cat((torch.tensor([0], device=num_atoms.device), num_atoms[:-1])), dim=0)
        ends = torch.cumsum(num_atoms, dim=0)
        crystal_atom_idx = []
        for start, end in zip(starts, ends):
            crystal_atom_idx.append(torch.arange(start, end))

        crys_fea = [torch.mean(nodes[idx_map], dim=0, keepdim=True) for idx_map in crystal_atom_idx]
        crys_fea = torch.cat(crys_fea, dim=0)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        predict = self.fc_out(crys_fea)
        predict.squeeze_()

        if self.classification:
            predict = self.logsoftmax(predict)
        elif self.normalization:
            predict *= self.normalize_stddev
            predict += self.normalize_mean

        return predict
