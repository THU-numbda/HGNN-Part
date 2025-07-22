"""
Enhanced training script with hyperparameter optimization integration
"""
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from pathlib import Path
import json
import time
from torch_geometric.loader import DataLoader
from transformers import set_seed

from hyperopt_config import HyperOptConfig, NormalizationMethod
from convergence_monitor import ConvergenceMonitor, AdaptiveLossWeighting, QualityEvaluator, EarlyStoppingManager
from feature_normalizer import FeatureNormalizer, create_enhanced_features
from models import GraphPartitionModel, HyperData
from logger import setup_logger

class EnhancedTrainer:
    """Enhanced trainer with advanced monitoring and adaptive techniques"""
    
    def __init__(self, config_dict: dict, data_path: str, output_dir: str = "./training_results"):
        self.config = config_dict
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger('training', str(self.output_dir / 'logs'))
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize monitoring systems
        self.convergence_monitor = ConvergenceMonitor(
            target_cut_loss=0.02,
            target_balance_loss=1e-3,
            stability_window=self.config.get('stability_window', 10)
        )
        
        self.quality_evaluator = QualityEvaluator()
        
        self.early_stopping = EarlyStoppingManager(
            patience=self.config.get('patience', 15),
            min_delta=self.config.get('min_delta', 1e-5),
            restore_best_weights=True
        )
        
    def create_model(self) -> nn.Module:
        """Create model with configuration"""
        model = GraphPartitionModel(
            input_dim=7,
            hidden_dim=self.config.get('hidden_dim', 256),
            latent_dim=self.config.get('latent_dim', 64),
            num_partitions=2,
            use_hypergraph=True
        ).to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
        
        return model
    
    def create_optimizer_and_scheduler(self, model: nn.Module, loss_weighter: nn.Module):
        """Create optimizer and learning rate scheduler"""
        # Separate parameter groups for model and loss weighter
        param_groups = [
            {'params': model.parameters(), 'lr': self.config.get('learning_rate', 5e-5)},
        ]
        
        if hasattr(loss_weighter, 'parameters') and any(p.requires_grad for p in loss_weighter.parameters()):
            param_groups.append({
                'params': loss_weighter.parameters(), 
                'lr': self.config.get('learning_rate', 5e-5) * 0.1  # Lower LR for loss weights
            })
        
        # Create optimizer
        optimizer_type = self.config.get('optimizer_type', 'adam')
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.get('learning_rate', 5e-5),
                weight_decay=self.config.get('weight_decay', 1e-5),
                betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.999))
            )
        elif optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                param_groups,
                lr=self.config.get('learning_rate', 5e-5),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        else:  # adam
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.config.get('learning_rate', 5e-5),
                weight_decay=self.config.get('weight_decay', 1e-5),
                betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.999))
            )
        
        # Create scheduler
        scheduler_type = self.config.get('scheduler_type', 'plateau')
        if scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.get('scheduler_factor', 0.5),
                patience=self.config.get('scheduler_patience', 5),
                verbose=True,
                min_lr=1e-7
            )
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=1e-7
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.get('scheduler_step_size', 30),
                gamma=self.config.get('scheduler_factor', 0.5)
            )
        else:  # exponential
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.config.get('scheduler_factor', 0.98)
            )
        
        return optimizer, scheduler
    
    def prepare_data(self):
        """Prepare enhanced dataset with configurable normalization"""
        # Load base hypergraph data
        dataset_files = list(Path(self.data_path).glob("*.hgr"))
        if not dataset_files:
            raise ValueError(f"No .hgr files found in {self.data_path}")
        
        enhanced_data = []
        for file_path in dataset_files:
            # Load hypergraph
            with open(file_path, 'r') as f:
                lines = f.readlines()
                num_nets, num_nodes = map(int, lines[0].split())
                hypergraph_vertices = list(range(num_nodes))
                hypergraph_edges = []
                for line in lines[1:]:
                    if line.strip():
                        edge = [int(node) - 1 for node in line.split()]
                        if len(edge) <= 1000:  # Filter large edges
                            hypergraph_edges.append(edge)
            
            # Create enhanced features
            try:
                features = create_enhanced_features(
                    hypergraph_vertices, hypergraph_edges, 
                    file_path.name, num_nodes, len(hypergraph_edges),
                    norm_method=self.config.get('norm_method', NormalizationMethod.STANDARD),
                    spectral_norm=self.config.get('spectral_norm', True),
                    feature_scaling=self.config.get('feature_scaling', 1.0)
                )
                
                # Create hyperedge index
                hyperedge_index = torch.tensor(np.array([
                    np.concatenate(hypergraph_edges),
                    np.repeat(np.arange(len(hypergraph_edges)), [len(e) for e in hypergraph_edges])
                ]), dtype=torch.long)
                
                # Create data object
                x = torch.tensor(features, dtype=torch.float)
                data = HyperData(x=x, hyperedge_index=hyperedge_index)
                data.filename = file_path.name
                data.hypergraph_edges = hypergraph_edges
                data.hypergraph_vertices = hypergraph_vertices
                
                enhanced_data.append(data)
                
            except Exception as e:
                self.logger.warning(f"Failed to process {file_path.name}: {str(e)}")
                continue
        
        if not enhanced_data:
            raise ValueError("No data could be processed successfully")
        
        self.logger.info(f"Prepared {len(enhanced_data)} datasets")
        return enhanced_data
    
    def train_epoch(self, model, loss_weighter, optimizer, dataloader, epoch):
        """Train for one epoch with advanced monitoring"""
        model.train()
        
        epoch_losses = []
        epoch_kl_losses = []
        epoch_cut_losses = []
        epoch_balance_losses = []
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            Y = model(data)
            
            # Compute individual losses
            num_nodes, num_nets = data.x.shape[0], data.hyperedge_index[1][-1].item() + 1
            W = torch.sparse_coo_tensor(
                data.hyperedge_index, 
                torch.ones(data.hyperedge_index.shape[1], device=self.device), 
                (num_nodes, num_nets),
                device=self.device
            )
            D = torch.sparse.sum(W, dim=1).to_dense().unsqueeze(1)
            
            kl_loss = model.kl_loss()
            cut_loss = model.hyperedge_cut_loss(Y, W)
            balance_loss = model.balance_loss(Y)
            
            # Adaptive loss weighting
            total_loss, weight_info = loss_weighter(kl_loss, cut_loss, balance_loss)
            
            # Check for NaN
            if torch.isnan(total_loss):
                self.logger.warning(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
                continue
            
            # Backward pass with gradient clipping
            total_loss.backward()
            if self.config.get('gradient_clip', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=self.config.get('gradient_clip', 1.0)
                )
            optimizer.step()
            
            # Record losses
            epoch_losses.append(total_loss.item())
            epoch_kl_losses.append(kl_loss.item())
            epoch_cut_losses.append(cut_loss.item())
            epoch_balance_losses.append(balance_loss.item())
            
            # Cleanup
            del W, D, Y, total_loss, kl_loss, cut_loss, balance_loss
            
        # Update loss weighter epoch
        if hasattr(loss_weighter, 'step_epoch'):
            loss_weighter.step_epoch()
        
        return {
            'total_loss': np.mean(epoch_losses) if epoch_losses else float('inf'),
            'kl_loss': np.mean(epoch_kl_losses) if epoch_kl_losses else float('inf'),
            'cut_loss': np.mean(epoch_cut_losses) if epoch_cut_losses else float('inf'),
            'balance_loss': np.mean(epoch_balance_losses) if epoch_balance_losses else float('inf'),
        }
    
    def evaluate_model(self, model, data):
        """Evaluate model partition quality"""
        model.eval()
        
        with torch.no_grad():
            # Sample multiple partitions
            num_samples = self.config.get('num_samples', 10)
            partitions = model.sample(data, m=num_samples)
            
            best_cut = float('inf')
            best_partition = None
            
            # Evaluate each partition
            for partition in partitions:
                quality_metrics = self.quality_evaluator.evaluate_partition_quality(
                    partition, data.hypergraph_vertices, data.hypergraph_edges
                )
                
                if quality_metrics['cut'] < best_cut:
                    best_cut = quality_metrics['cut']
                    best_partition = partition
            
            return best_partition, quality_metrics
    
    def train(self) -> dict:
        """Main training loop with comprehensive monitoring"""
        self.logger.info("Starting enhanced training")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2, default=str)}")
        
        # Setup
        set_seed(self.config.get('seed', 42))
        
        # Prepare data
        enhanced_data = self.prepare_data()
        dataloader = DataLoader(enhanced_data, batch_size=1, shuffle=True)
        
        # Create model and training components
        model = self.create_model()
        
        loss_weighter = AdaptiveLossWeighting(
            initial_alpha=self.config.get('alpha', 0.001),
            initial_beta=self.config.get('beta', 5.0),
            initial_gamma=self.config.get('gamma', 2.0),
            adaptive=self.config.get('adaptive_weights', True),
            annealing=self.config.get('loss_annealing', True)
        ).to(self.device)
        
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, loss_weighter)
        
        # Training history
        history = {
            'train_loss': [], 'kl_loss': [], 'cut_loss': [], 'balance_loss': [],
            'quality_scores': [], 'stability_scores': [], 'convergence_info': []
        }
        
        best_metrics = {
            'cut_loss': float('inf'),
            'balance_loss': float('inf'),
            'quality_score': 0.0
        }
        
        # Training loop
        max_epochs = self.config.get('epochs', 100)
        start_time = time.time()
        
        for epoch in range(max_epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(model, loss_weighter, optimizer, dataloader, epoch)
            
            # Update convergence monitoring
            convergence_info = self.convergence_monitor.update(
                train_metrics['cut_loss'],
                train_metrics['balance_loss'],
                train_metrics['kl_loss']
            )
            
            # Evaluate model quality (on first dataset for speed)
            if epoch % 5 == 0:  # Every 5 epochs
                _, quality_metrics = self.evaluate_model(model, enhanced_data[0])
                quality_score = quality_metrics.get('overall_quality', 0.0)
            else:
                quality_score = history['quality_scores'][-1] if history['quality_scores'] else 0.0
            
            # Update learning rate scheduler
            if hasattr(scheduler, 'step'):
                if self.config.get('scheduler_type') == 'plateau':
                    scheduler.step(train_metrics['total_loss'])
                else:
                    scheduler.step()
            
            # Record history
            for key in ['total_loss', 'kl_loss', 'cut_loss', 'balance_loss']:
                history[key.replace('total_', 'train_')].append(train_metrics[key])
            history['quality_scores'].append(quality_score)
            history['stability_scores'].append(convergence_info['stability_metrics']['overall_stability'])
            history['convergence_info'].append(convergence_info)
            
            # Update best metrics
            if train_metrics['cut_loss'] < best_metrics['cut_loss']:
                best_metrics['cut_loss'] = train_metrics['cut_loss']
            if train_metrics['balance_loss'] < best_metrics['balance_loss']:
                best_metrics['balance_loss'] = train_metrics['balance_loss']
            if quality_score > best_metrics['quality_score']:
                best_metrics['quality_score'] = quality_score
            
            # Logging
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            
            self.logger.info(
                f"Epoch {epoch+1:3d}/{max_epochs} | "
                f"Loss: {train_metrics['total_loss']:.4f} | "
                f"Cut: {train_metrics['cut_loss']:.4f} | "
                f"Balance: {train_metrics['balance_loss']:.6f} | "
                f"KL: {train_metrics['kl_loss']:.4f} | "
                f"Quality: {quality_score:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Check convergence
            if convergence_info['converged']:
                self.logger.info(f"🎯 Target convergence achieved at epoch {epoch+1}!")
                self.logger.info(f"Cut loss: {train_metrics['cut_loss']:.6f} <= {self.convergence_monitor.target_cut_loss}")
                self.logger.info(f"Balance loss: {train_metrics['balance_loss']:.6f} <= {self.convergence_monitor.target_balance_loss}")
                break
            
            # Early stopping
            combined_score = train_metrics['cut_loss'] + train_metrics['balance_loss'] * 10  # Weight balance more
            should_stop = self.early_stopping(combined_score, model, epoch)
            if should_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Save intermediate results
            if (epoch + 1) % 20 == 0:
                self._save_checkpoint(model, optimizer, scheduler, epoch, history, best_metrics)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.2f} minutes")
        
        # Final evaluation and save
        final_results = self._final_evaluation(model, enhanced_data, history, best_metrics)
        self._save_final_results(model, final_results)
        
        return final_results
    
    def _save_checkpoint(self, model, optimizer, scheduler, epoch, history, best_metrics):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history,
            'best_metrics': best_metrics,
            'config': self.config
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
    def _final_evaluation(self, model, enhanced_data, history, best_metrics):
        """Comprehensive final evaluation"""
        self.logger.info("Running final evaluation...")
        
        model.eval()
        final_results = {
            'training_history': history,
            'best_metrics': best_metrics,
            'final_evaluation': {},
            'convergence_achieved': self.convergence_monitor.convergence_achieved,
            'config': self.config
        }
        
        # Evaluate on all datasets
        dataset_results = []
        for i, data in enumerate(enhanced_data):
            best_partition, quality_metrics = self.evaluate_model(model, data)
            
            eval_result = {
                'dataset': data.filename,
                'cut': quality_metrics['cut'],
                'imbalance': quality_metrics['imbalance'],
                'quality_score': quality_metrics['overall_quality'],
                'balance_score': quality_metrics['balance_score']
            }
            dataset_results.append(eval_result)
            
            self.logger.info(f"Dataset {data.filename}: Cut={quality_metrics['cut']}, "
                           f"Imbalance={quality_metrics['imbalance']:.4f}, "
                           f"Quality={quality_metrics['overall_quality']:.3f}")
        
        final_results['final_evaluation']['datasets'] = dataset_results
        
        # Aggregate metrics
        if dataset_results:
            avg_cut = np.mean([r['cut'] for r in dataset_results])
            avg_imbalance = np.mean([r['imbalance'] for r in dataset_results])
            avg_quality = np.mean([r['quality_score'] for r in dataset_results])
            
            final_results['final_evaluation']['aggregate'] = {
                'avg_cut': avg_cut,
                'avg_imbalance': avg_imbalance,
                'avg_quality': avg_quality,
                'target_cut_achieved': best_metrics['cut_loss'] <= 0.02,
                'target_balance_achieved': best_metrics['balance_loss'] <= 1e-3
            }
            
            self.logger.info(f"Final Results - Avg Cut: {avg_cut:.2f}, "
                           f"Avg Imbalance: {avg_imbalance:.4f}, "
                           f"Avg Quality: {avg_quality:.3f}")
        
        return final_results
    
    def _save_final_results(self, model, results):
        """Save final model and results"""
        # Save model
        model_path = self.output_dir / "final_model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            # Make results JSON serializable
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Final results saved to {self.output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        if 'final_evaluation' in results and 'aggregate' in results['final_evaluation']:
            agg = results['final_evaluation']['aggregate']
            print(f"🎯 Target Cut Loss (≤0.02): {'✅' if agg.get('target_cut_achieved', False) else '❌'}")
            print(f"🎯 Target Balance Loss (≤1e-3): {'✅' if agg.get('target_balance_achieved', False) else '❌'}")
            print(f"📊 Average Cut: {agg.get('avg_cut', 'N/A')}")
            print(f"📊 Average Quality: {agg.get('avg_quality', 'N/A'):.3f}")
            print(f"📁 Results saved to: {self.output_dir}")
        
        print("="*60)
    
    def _make_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'value'):  # Enums
            return obj.value
        else:
            return obj

def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="Enhanced GraphPart Training")
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--output-dir', type=str, default='./training_results', help='Output directory')
    
    # Individual parameters (override config file)
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--alpha', type=float, help='KL loss weight')
    parser.add_argument('--beta', type=float, help='Cut loss weight')
    parser.add_argument('--gamma', type=float, help='Balance loss weight')
    parser.add_argument('--hidden-dim', type=int, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--norm-method', type=str, help='Normalization method')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with command line arguments
    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.alpha is not None:
        config['alpha'] = args.alpha
    if args.beta is not None:
        config['beta'] = args.beta
    if args.gamma is not None:
        config['gamma'] = args.gamma
    if args.hidden_dim is not None:
        config['hidden_dim'] = args.hidden_dim
    if args.dropout is not None:
        config['dropout_rate'] = args.dropout
    if args.norm_method is not None:
        # Convert string to enum
        for norm_enum in NormalizationMethod:
            if norm_enum.value == args.norm_method:
                config['norm_method'] = norm_enum
                break
    
    # Create trainer and run
    trainer = EnhancedTrainer(config, args.data_path, args.output_dir)
    results = trainer.train()
    
    return results

if __name__ == '__main__':
    main()