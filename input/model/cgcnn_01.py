import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self,
                 edge_embedding_size: int = 20,  # Edge embedding dimension
                 hidden_size: int = 128  # Hidden dimension
                 ):
        super(ConvLayer, self).__init__()
        self.hidden_size = hidden_size

        self.filter_layer = nn.Linear(2 * edge_embedding_size, hidden_size)
        self.fc_full = nn.Linear(3 * hidden_size, 2 * hidden_size)
        self.bn1 = nn.BatchNorm1d(2 * hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self,
                nodes,  # Node features
                rbf_edges,  # Edge features after radial basis function expansion
                nbrs_idx  # Neighbor index
                ):
        N, M = nbrs_idx.shape

        # convolution
        edges = self.filter_layer(rbf_edges)
        nbrs_fea = nodes[nbrs_idx]
        total_fea = torch.cat([nodes.unsqueeze(1).expand(N, M, self.hidden_size),
                               edges,
                               nbrs_fea],
                              dim=2)
        total_gated_fea = self.fc_full(total_fea)
        nbr_filter, nbr_core = torch.split(total_gated_fea, self.hidden_size, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        nbr_aggred = torch.sum(nbr_filter * nbr_core, dim=1)

        new_nodes = self.softplus(nodes + nbr_aggred)

        return new_nodes


class Model(nn.Module):
    def __init__(self,
                 cutoff,  # Cutoff radius
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
        self.cutoff = cutoff
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

        bonds = input_dict['nbrs_fea']
        ches, vdws = torch.split(bonds, 1, dim=2)
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
        rbf_edges = torch.cat([ches, vdws], 2)

        nbrs_idx = input_dict['nbrs_idx']

        for conv_func in self.convs:
            nodes = conv_func(nodes, rbf_edges, nbrs_idx)

        num_atoms = input_dict['num_atoms']
        starts = torch.cumsum(torch.cat((torch.tensor([0], device= num_atoms.device), num_atoms[:-1])), dim=0)
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
