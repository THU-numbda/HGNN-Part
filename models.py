import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, VGAE, HypergraphConv
from torch_geometric.data import Data
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mul
import math
from typing import Union

class HyperData:
    def __init__(self, x, hyperedge_index, hyperedge_weight=None):
        self.x = x
        self.hyperedge_index = hyperedge_index
        self.hyperedge_weight = hyperedge_weight

    def to(self, device):
        self.x = self.x.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        if self.hyperedge_weight is not None:
            self.hyperedge_weight = self.hyperedge_weight.to(device)
        return self

class NewHyperData(Data):
    def __init__(self, x, hyperedge_index):
        super().__init__()
        self.x = x
        self.hyperedge_index = hyperedge_index

    def to(self, device):
        self.x = self.x.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        return self

class HypergraphPartitionModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, latent_dim=64, num_partitions=2):
        super().__init__()
        self.conv1 = HypergraphConv(input_dim, hidden_dim , dropout=0.2)
        self.conv2 = HypergraphConv(hidden_dim, hidden_dim, dropout=0.2)
        self.conv3 = HypergraphConv(hidden_dim, hidden_dim, dropout=0.2)
        self.conv4 = HypergraphConv(hidden_dim, hidden_dim, dropout=0.2)
        self.conv5 = HypergraphConv(hidden_dim, latent_dim, dropout=0.2)
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 2)
        self.num_partitions = num_partitions
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, data: NewHyperData):
        # x = F.dropout(data.x, p=0.2, training=self.training)
        x = F.elu(self.conv1(data.x, data.hyperedge_index))
        x = F.elu(self.conv2(x, data.hyperedge_index))
        x = F.elu(self.conv3(x, data.hyperedge_index))
        x = F.elu(self.conv4(x, data.hyperedge_index))
        x = F.elu(self.conv5(x, data.hyperedge_index))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc_out(x), dim=-1)
        return x
    
    def hyperedge_cut_loss(self, Y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        row, col = W._indices()[0], W._indices()[1]
        sorted_col, sorted_idx = torch.sort(col)
        sorted_row = row[sorted_idx]
        Y_expanded = Y[sorted_row]
        prod_per_edge_per_partition = torch.zeros(W.shape[1], self.num_partitions, device=Y.device)
        for j in range(self.num_partitions):
            prod_per_edge_per_partition[:, j] = scatter_mul(Y_expanded[:, j], sorted_col, dim=0)
        return W.shape[1] - torch.sum(torch.sum(prod_per_edge_per_partition, dim=1))

    def normalized_hyperedge_cut_loss(self, Y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        cut_loss = self.hyperedge_cut_loss(Y, W)
        partition_sizes = Y.sum(dim=0)
        return cut_loss * torch.sum(partition_sizes.pow(-1))

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, latent_dim=64, use_hypergraph=False):
        super().__init__()
        # 图卷积层
        self.conv1 = HypergraphConv(input_dim, hidden_dim) if use_hypergraph else GraphConv(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, hidden_dim) if use_hypergraph else GraphConv(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.conv3 = HypergraphConv(hidden_dim, hidden_dim) if use_hypergraph else GraphConv(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        # self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # self.fc_logstd = nn.Linear(hidden_dim, latent_dim)
        self.fc_mu = HypergraphConv(hidden_dim, latent_dim) if use_hypergraph else GraphConv(hidden_dim, latent_dim)
        self.fc_logstd = HypergraphConv(hidden_dim, latent_dim) if use_hypergraph else GraphConv(hidden_dim, latent_dim)
        self.mask_token = nn.Parameter(torch.randn(1, input_dim))  # Mask token for input
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, data: Union[Data, NewHyperData]):
        x = data.x if isinstance(data, Data) else data.x
        edge_index = data.hyperedge_index if isinstance(data, NewHyperData) else data.edge_index
        # x = F.dropout(x, p=0.2, training=self.training)
        x_clone = x.clone()
        if self.training:
            num_nodes = x_clone.size(0)
            num_mask = max(1, int(0.2 * num_nodes))
            perm = torch.randperm(num_nodes, device=x_clone.device)
            masked_indices = perm[:num_mask]
            x_clone[masked_indices] = self.mask_token.to(x.dtype)
        x1 = self.ln1(F.relu(self.conv1(x_clone, edge_index)))
        x2 = self.ln2(F.relu(self.conv2(x1, edge_index)))
        x3 = self.ln3(F.relu(self.fc(x1 + x2)))
        x4 = self.ln4(F.relu(self.conv3(x3, edge_index)))
        mu = self.fc_mu(x4, edge_index)
        logstd = self.fc_logstd(x4, edge_index)
        return mu, logstd

class PartitionDecoder(nn.Module):
    def __init__(self, latent_dim=64, num_partitions=2):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, num_partitions)  # 输出分区数量

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc_out(x), dim=-1)
        return x

class GraphPartitionModel(VGAE):
    def __init__(self, input_dim=5, hidden_dim=256, latent_dim=64, num_partitions=2, use_hypergraph=False):
        super().__init__(VariationalEncoder(input_dim, hidden_dim, latent_dim, use_hypergraph), PartitionDecoder(latent_dim, num_partitions))
        self.num_partitions = num_partitions
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.use_hypergraph = use_hypergraph

    def normalized_cut_loss(self, Y: torch.Tensor, W: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        ncut = torch.tensor(0.0, device=Y.device)
        for g in range(self.num_partitions):
            Y_g = Y[:, g]
            indices = W._indices()
            values = W._values()
            row, col = indices[0, :], indices[1, :]
            cut_values = Y_g[row] * (1 - Y_g[col]) * values
            ncut += cut_values.sum() / (Y_g @ D).sum()
        return ncut
    
    def hyperedge_cut_loss(self, Y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        row, col = W._indices()[0], W._indices()[1]
        sorted_col, sorted_idx = torch.sort(col)
        sorted_row = row[sorted_idx]
        Y_expanded = Y[sorted_row]
        prod_per_edge_per_partition = torch.zeros(W.shape[1], self.num_partitions, device=Y.device)
        for j in range(self.num_partitions):
            prod_per_edge_per_partition[:, j] = scatter_mul(Y_expanded[:, j], sorted_col, dim=0)
        return W.shape[1] - torch.sum(torch.sum(prod_per_edge_per_partition, dim=1))
    
    def balance_loss(self, Y: torch.Tensor):
        ideal_size = Y.shape[0] / self.num_partitions
        partition_sizes = Y.sum(dim=0)
        return torch.sum((partition_sizes - ideal_size).pow(2))
    
    def normalized_hyperedge_cut_loss(self, Y: torch.Tensor, W: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        cut_loss = self.hyperedge_cut_loss(Y, W)
        partition_degrees = (D * Y).sum(dim=0)
        return cut_loss * torch.sum(partition_degrees.pow(-1))
    
    def combined_loss(self, Y, W, D, alpha=0.0005, beta=5, gamma=1):
        kl_loss = self.kl_loss()
        ncut_loss = self.normalized_hyperedge_cut_loss(Y, W, D) if self.use_hypergraph else self.normalized_cut_loss(Y, W, D)
        balance_loss_val = self.balance_loss(Y)
        return alpha * kl_loss + beta * ncut_loss + gamma * balance_loss_val, kl_loss, ncut_loss, balance_loss_val
    
    def combined_loss_uncertainty(self, Y, W, D):
        kl_loss = self.kl_loss()
        ncut_loss = self.normalized_hyperedge_cut_loss(Y, W, D) if self.use_hypergraph else self.normalized_cut_loss(Y, W, D)
        balance_loss_val = self.balance_loss(Y)
        precision_kl = torch.exp(-self.log_vars[0])
        precision_ncut = torch.exp(-self.log_vars[1])
        precision_balance = torch.exp(-self.log_vars[2])
        loss = precision_kl * kl_loss + 0.5 * self.log_vars[0] + \
               precision_ncut * ncut_loss + 0.5 * self.log_vars[1] + \
               precision_balance * balance_loss_val + 0.5 * self.log_vars[2]
        return loss, kl_loss, ncut_loss, balance_loss_val

    def forward(self, data: Data):
        z = self.encode(data)
        Y = self.decode(z)
        return Y
    
    def sample(self, data: Data, m=1):
        mu, logstd = self.encoder(data)
        logstd = torch.clamp(logstd, max=10)
        samples = []
        for _ in range(m):
            z = mu + torch.randn_like(logstd, device=data.x.device) * torch.exp(logstd)
            Y = self.decode(z)
            partition = torch.argmax(Y, dim=-1)
            samples.append(partition.cpu().numpy())
        return samples
