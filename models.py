import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, VGAE
from torch_geometric.data import Data
import torch.nn as nn

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, latent_dim=64):
        super().__init__()
        # 图卷积层
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.conv5 = GraphConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logstd = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data: Data):
        x = data.x
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        x = F.relu(self.conv4(x, data.edge_index))
        x = F.relu(self.conv5(x, data.edge_index))
        mu = self.fc_mu(x)
        logstd = self.fc_logstd(x)
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
    def __init__(self, input_dim=5, hidden_dim=256, latent_dim=64, num_partitions=2):
        super().__init__(VariationalEncoder(input_dim, hidden_dim, latent_dim), PartitionDecoder(latent_dim, num_partitions))
        self.num_partitions = num_partitions

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
    
    def balance_loss(self, Y: torch.Tensor):
        ideal_size = Y.shape[0] / self.num_partitions
        partition_sizes = Y.sum(dim=0)
        return torch.sum((partition_sizes - ideal_size).pow(2))
    
    def combined_loss(self, Y, W, D, alpha=0.0005, beta=5, gamma=1):
        kl_loss = self.kl_loss() / Y.shape[0]
        ncut_loss = self.normalized_cut_loss(Y, W, D)
        balance_loss_val = self.balance_loss(Y)
        return alpha * kl_loss + beta * ncut_loss + gamma * balance_loss_val, kl_loss, ncut_loss, balance_loss_val

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
