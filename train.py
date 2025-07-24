import torch
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from transformers import set_seed
import os
import argparse
import gc
from models import GraphPartitionModel, HyperData
import swanlab

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
    parser.add_argument('--alpha', type=float, default=0.001)  # KL regularization
    parser.add_argument('--beta', type=float, default=20)     # Cut loss priority
    parser.add_argument('--gamma', type=float, default=1.5)   # Balance constraint
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    # Initialize SwanLab
    swanlab.init(
        project="GraphPart",
        experiment_name=f"vgae_lr{args.lr}_a{args.alpha}_b{args.beta}_g{args.gamma}",
        config={
            "learning_rate": args.lr,
            "alpha": args.alpha,
            "beta": args.beta,  
            "gamma": args.gamma,
            "epochs": args.epochs,
            "input_dim": 7,
            "latent_dim": 64,
            "hidden_dim": 256,
            "num_partitions": 2,
            "batch_size": 1,
            "weight_decay": 1e-5,
            "patience": 10
        }
    )
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
        epoch_kl_loss = np.mean(kl_losses) if kl_losses else 0.0
        epoch_cut_loss = np.mean(hyperedge_cut_losses) if hyperedge_cut_losses else 0.0
        epoch_balance_loss = np.mean(balance_losses) if balance_losses else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics to SwanLab
        swanlab.log({
            "epoch": epoch + 1,
            "train/total_loss": epoch_loss,
            "train/kl_loss": epoch_kl_loss,
            "train/cut_loss": epoch_cut_loss,
            "train/balance_loss": epoch_balance_loss,
            "train/learning_rate": current_lr,
            "system/peak_memory_gb": peak_memory_gb,
            "system/batches_processed": len(losses)
        })
        
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4e}, KL Loss: {epoch_kl_loss:.4e}, Ncut Loss: {epoch_cut_loss:.4e}, Balance Loss: {epoch_balance_loss:.4e}, Peak Memory: {peak_memory_gb:.2f} GB')
        
        # Learning rate scheduling
        scheduler.step(epoch_loss)
        
        # Early stopping and best model saving
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), './models/model.best.pt')
            # Log best model metrics to SwanLab
            swanlab.log({
                "train/best_loss": best_loss,
                "train/best_epoch": epoch + 1
            })
            print(f'New best model saved at epoch {epoch + 1}')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            swanlab.log({"train/early_stopped": True, "train/final_epoch": epoch + 1})
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break
            
    # Save final model
    final_model_path = f'./models/model.{args.lr:.0e}.{args.alpha:.0e}.{args.beta:.0e}.{args.gamma:.0e}.pt'
    torch.save(model.state_dict(), final_model_path)
    
    # Log final training summary
    swanlab.log({
        "train/final_best_loss": best_loss,
        "train/total_epochs_completed": epoch + 1
    })
    
    print(f'Final model saved as model.{args.lr:.0e}.{args.alpha:.0e}.{args.beta:.0e}.{args.gamma:.0e}.pt')
    
    # Finish SwanLab run
    swanlab.finish()
