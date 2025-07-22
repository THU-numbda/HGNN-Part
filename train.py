import torch
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from transformers import set_seed
import os
import argparse
import gc
from models import GraphPartitionModel, HyperData

# Ensure models directory exists
os.makedirs('./models', exist_ok=True)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--alpha', type=float, default=0.0005)
    parser.add_argument('--beta', type=float, default=5)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    dataset = ISPDDataset('/data1/tongsb/GraphPart/dataset/pt/train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    input_dim = 7
    latent_dim = 64
    hidden_dim = 256
    num_partitions = 2
    num_epochs = args.epochs
    model = GraphPartitionModel(input_dim, hidden_dim, latent_dim, num_partitions, True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # Add weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    patience = 10
    
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
            loss, kl_loss, hyperedge_cut_loss, balance_loss = model.combined_loss(Y, W, D, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            if not loss.isnan():
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Enable gradient clipping
                optimizer.step()
                losses.append(loss.item())
                kl_losses.append(kl_loss.item())
                hyperedge_cut_losses.append(hyperedge_cut_loss.item())
                balance_losses.append(balance_loss.item())
            else:
                print(f'Loss is NaN at epoch {epoch + 1}! Skipping this batch.')
            # More efficient memory management - only clean up heavy objects
            del W, Y, D
            if 'loss' in locals():
                del loss, kl_loss, hyperedge_cut_loss, balance_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        epoch_loss = np.mean(losses) if losses else float('inf')
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4e}, KL Loss: {np.mean(kl_losses):.4e}, Ncut Loss: {np.mean(hyperedge_cut_losses):.4e}, Balance Loss: {np.mean(balance_losses):.4e}, Peak Memory: {peak_memory_gb:.2f} GB')
        
        # Learning rate scheduling
        scheduler.step(epoch_loss)
        
        # Early stopping and best model saving
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), './models/model.best.pt')
            print(f'New best model saved at epoch {epoch + 1}')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break
            
    torch.save(model.state_dict(), f'./models/model.{args.lr:.0e}.{args.alpha:.0e}.{args.beta:.0e}.{args.gamma:.0e}.pt')
    print(f'Final model saved as model.{args.lr:.0e}.{args.alpha:.0e}.{args.beta:.0e}.{args.gamma:.0e}.pt')
