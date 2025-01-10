import torch
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from transformers import set_seed
import os
from models import GraphPartitionModel

class ISPDDataset(Dataset):
    def __init__(self, root):
        super().__init__(root)
        self.data_files = os.listdir(root)

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        filename = self.data_files[idx]
        data = torch.load(os.path.join(self.root, filename, filename + '.pt'))
        D = torch.load(os.path.join(self.root, filename, filename + '.D.pt'))
        return data, D

if __name__ == '__main__':
    torch.cuda.set_device(3)
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ISPDDataset('./dataset/ISPD98')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    input_dim = 5
    latent_dim = 64
    hidden_dim = 256
    num_partitions = 2
    num_epochs = 50
    model = GraphPartitionModel(input_dim, hidden_dim, latent_dim, num_partitions)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(num_epochs):
        losses, kl_losses, ncut_losses, balance_losses = [], [], [], []
        min_ncut_loss = float('inf')
        for data, D in dataloader:
            data = data.to(device)
            D = D.to(device).squeeze(0)
            W = torch.sparse_coo_tensor(data.edge_index, data.edge_attr, (len(data.x), len(data.x)))
            W = W.to(device)
            # indices = W._indices()
            # row, col = indices[0, :], indices[1, :]
            # values = torch.ones_like(row, dtype=torch.float)
            # D_ = torch.zeros_like(D).squeeze(2)
            # D_.scatter_add_(dim=1, index=row.unsqueeze(0), src=values.unsqueeze(0))
            optimizer.zero_grad()
            Y = model(data)
            loss, kl_loss, ncut_loss, balance_loss = model.combined_loss(Y, W, D)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            losses.append(loss.item())
            kl_losses.append(kl_loss.item())
            ncut_losses.append(ncut_loss.item())
            balance_losses.append(balance_loss.item())
        print(f'Epoch {epoch + 1}, Loss: {np.mean(losses):.4e}, KL Loss: {np.mean(kl_losses):.4e}, Ncut Loss: {np.mean(ncut_losses):.4e}, Balance Loss: {np.mean(balance_losses):.4e}')
        if np.mean(ncut_losses) < min_ncut_loss:
            min_ncut_loss = np.mean(ncut_losses)
            torch.save(model.state_dict(), './dataset/models/model.best.pt')
    torch.save(model.state_dict(), './dataset/models/model.pt')
