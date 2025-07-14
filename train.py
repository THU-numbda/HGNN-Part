import torch
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from transformers import set_seed
import os
from models import GraphPartitionModel, HyperData

class ISPDDataset(Dataset):
    def __init__(self, root):
        super().__init__(root)
        self.data_files = os.listdir(root)

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        filename = self.data_files[idx]
        pt = torch.load(os.path.join(self.root, filename, filename + '.pt'))
        data = HyperData(x=pt.x, hyperedge_index=pt.hyperedge_index)
        return data

if __name__ == '__main__':
    torch.cuda.set_device(1)
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ISPDDataset('/data1/tongsb/GraphPart/dataset/pt/train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    input_dim = 7
    latent_dim = 64
    hidden_dim = 256
    num_partitions = 2
    num_epochs = 50
    model = GraphPartitionModel(input_dim, hidden_dim, latent_dim, num_partitions, True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(num_epochs):
        losses, kl_losses, hyperedge_cut_losses, balance_losses = [], [], [], []
        min_ncut_loss = float('inf')
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            Y = model(data)
            num_nodes, num_nets = data.x.shape[0], data.hyperedge_index[1][-1].item() + 1
            W = torch.sparse_coo_tensor(data.hyperedge_index, torch.ones(data.hyperedge_index.shape[1]).to(device), (num_nodes, num_nets)).to(device)
            D = torch.sparse.sum(W, dim=1).to_dense().unsqueeze(1)
            loss, kl_loss, hyperedge_cut_loss, balance_loss = model.combined_loss(Y, W, D)
            if not loss.isnan():
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                optimizer.step()
                losses.append(loss.item())
                kl_losses.append(kl_loss.item())
                hyperedge_cut_losses.append(hyperedge_cut_loss.item())
                balance_losses.append(balance_loss.item())
            else:
                print('Loss is NaN!')
            del data, W, Y, loss, kl_loss, hyperedge_cut_loss, balance_loss
            torch.cuda.empty_cache()
        print(f'Epoch {epoch + 1}, Loss: {np.mean(losses):.4e}, KL Loss: {np.mean(kl_losses):.4e}, Ncut Loss: {np.mean(hyperedge_cut_losses):.4e}, Balance Loss: {np.mean(balance_losses):.4e}')
        if np.mean(hyperedge_cut_losses) < min_ncut_loss:
            min_ncut_loss = np.mean(hyperedge_cut_losses)
            torch.save(model.state_dict(), './models/model.best.pt')
    torch.save(model.state_dict(), './models/model.pt')
