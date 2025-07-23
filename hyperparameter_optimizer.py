"""
Multi-strategy hyperparameter optimizer with Optuna, Ray Tune, and custom strategies
"""
import optuna
import numpy as np
import torch
import json
import warnings
from typing import Dict, Any, List, Tuple
from pathlib import Path
import pickle
import multiprocessing as mp

from hyperopt_config import HyperOptConfig, NormalizationMethod
from logger import setup_logger

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class MultiObjectiveOptimizer:
    """Multi-objective hyperparameter optimizer"""
    
    def __init__(self, data_path: str, output_dir: str = "./hyperopt_results", 
                 n_trials: int = 200, n_parallel: int = 4):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_trials = n_trials
        self.n_parallel = min(n_parallel, mp.cpu_count())
        
        self.logger = setup_logger('hyperopt', str(self.output_dir / 'logs'))
        self.config = HyperOptConfig()
        
        # Initialize Optuna study with cut loss priority
        storage_url = f"sqlite:///{self.output_dir}/optuna_study.db"
        self.study = optuna.create_study(
            directions=['minimize', 'minimize', 'maximize', 'maximize'],  # cut, balance_penalty, quality, stability
            study_name='graphpart_hyperopt',
            storage=storage_url,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=15,  # More startup trials for better cut loss focus
                n_warmup_steps=25,
                interval_steps=3  # More aggressive pruning
            )
        )
        
    def optimize(self) -> Dict[str, Any]:
        """Run comprehensive hyperparameter optimization"""
        self.logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        self.logger.info(f"Using {self.n_parallel} parallel workers")
        
        # Multi-stage optimization strategy
        results = {}
        
        # Stage 1: Coarse search with high pruning
        self.logger.info("Stage 1: Coarse search for promising regions")
        coarse_results = self._run_coarse_search(self.n_trials // 3)
        results['coarse_search'] = coarse_results
        
        # Stage 2: Fine-tuning around best configurations
        self.logger.info("Stage 2: Fine-tuning around best configurations")
        fine_results = self._run_fine_tuning(self.n_trials // 3)
        results['fine_tuning'] = fine_results
        
        # Stage 3: Final validation and ensemble
        self.logger.info("Stage 3: Final validation")
        final_results = self._run_final_validation(self.n_trials // 3)
        results['final_validation'] = final_results
        
        # Analyze and report best configurations
        best_config = self._analyze_results()
        results['best_config'] = best_config
        
        # Save comprehensive results
        self._save_results(results)
        
        return results
    
    def _run_coarse_search(self, n_trials: int) -> Dict:
        """Run coarse-grained parameter search"""
        def coarse_objective(trial):
            params = self._sample_coarse_parameters(trial)
            return self._objective_function(trial, params, max_epochs=100)
        
        # Use self.study directly instead of creating separate study
        self.study.optimize(coarse_objective, n_trials=n_trials, n_jobs=self.n_parallel)
        
        return {
            'n_trials': len(self.study.trials),
            'best_trials': self.study.best_trials[:5],  # Top 5 trials
            'study': self.study
        }
    
    def _run_fine_tuning(self, n_trials: int) -> Dict:
        """Fine-tune around best configurations from coarse search"""
        # Get best parameters from coarse search for fine-tuning
        best_trials = self.study.best_trials[:3] if self.study.best_trials else []
        
        def fine_objective(trial):
            if best_trials:
                # Sample around best configurations  
                # Use current trial index relative to current optimization stage
                base_idx = len(self.study.trials) % len(best_trials)
                base_params = best_trials[base_idx].params
                params = self._sample_fine_parameters(trial, base_params)
            else:
                params = self._sample_parameters(trial)
            return self._objective_function(trial, params, max_epochs=200)
        
        # Continue using self.study
        current_trial_count = len(self.study.trials)
        self.study.optimize(fine_objective, n_trials=n_trials, n_jobs=self.n_parallel)
        
        return {
            'n_trials': len(self.study.trials) - current_trial_count,
            'best_trials': self.study.best_trials[:3],
            'study': self.study
        }
    
    def _run_final_validation(self, n_trials: int) -> Dict:
        """Final validation with extended training"""
        def final_objective(trial):
            # Use only the very best parameter combinations
            if self.study.best_trials:
                best_params = self.study.best_trials[0].params
                params = self._sample_final_parameters(trial, best_params)
            else:
                params = self._sample_parameters(trial)
            return self._objective_function(trial, params, max_epochs=300, final_validation=True)
        
        # Continue using self.study, but disable pruning for final validation
        # Temporarily disable pruner
        original_pruner = self.study.pruner
        self.study.pruner = None
        
        current_trial_count = len(self.study.trials)
        self.study.optimize(final_objective, n_trials=min(n_trials, 20), n_jobs=min(2, self.n_parallel))
        
        # Restore original pruner
        self.study.pruner = original_pruner
        
        return {
            'n_trials': len(self.study.trials) - current_trial_count,
            'best_trials': self.study.best_trials,
            'study': self.study
        }
    
    def _sample_coarse_parameters(self, trial) -> Dict[str, Any]:
        """Sample parameters for coarse search with broader ranges"""
        params = {}
        
        # Use broader ranges for coarse search, prioritizing cut loss
        params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        params['beta'] = trial.suggest_uniform('beta', 15.0, 40.0)  # Higher cut loss weight
        params['gamma'] = trial.suggest_uniform('gamma', 0.3, 3.0)  # Lower balance weight
        params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e-2)
        
        # Architecture parameters
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        params['dropout_rate'] = trial.suggest_uniform('dropout_rate', 0.0, 0.4)
        params['mask_rate'] = trial.suggest_uniform('mask_rate', 0.1, 0.3)
        
        # Normalization
        params['norm_method'] = trial.suggest_categorical('norm_method', 
            [NormalizationMethod.STANDARD, NormalizationMethod.ROBUST, NormalizationMethod.UNIT_NORM])
        
        return params
    
    def _sample_fine_parameters(self, trial, base_params: Dict) -> Dict[str, Any]:
        """Sample parameters around a good configuration for fine-tuning"""
        params = base_params.copy()
        
        # Fine-tune around base values with cut loss priority
        if 'learning_rate' in base_params:
            base_lr = base_params['learning_rate']
            params['learning_rate'] = trial.suggest_loguniform('learning_rate', 
                base_lr * 0.3, base_lr * 3.0)
        
        if 'beta' in base_params:
            base_beta = base_params['beta']
            # Keep beta high for cut loss priority
            params['beta'] = trial.suggest_uniform('beta', 
                max(10.0, base_beta * 0.7), base_beta * 1.3)
        
        if 'gamma' in base_params:
            base_gamma = base_params['gamma']
            # Keep gamma low but sufficient for balance
            params['gamma'] = trial.suggest_uniform('gamma',
                max(0.1, base_gamma * 0.5), min(base_gamma * 1.5, 3.0))
        
        # Add additional parameters for fine-tuning
        params['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
        params['scheduler_factor'] = trial.suggest_uniform('scheduler_factor', 0.3, 0.7)
        params['gradient_clip'] = trial.suggest_uniform('gradient_clip', 0.5, 2.0)
        
        return params
    
    def _sample_final_parameters(self, trial, best_params: Dict) -> Dict[str, Any]:
        """Sample parameters for final validation with minimal variation"""
        params = best_params.copy()
        
        # Very small variations for final validation
        if 'learning_rate' in best_params:
            base_lr = best_params['learning_rate']
            params['learning_rate'] = trial.suggest_uniform('learning_rate',
                base_lr * 0.8, base_lr * 1.2)
        
        # Add ensemble parameters
        params['num_seeds'] = trial.suggest_int('num_seeds', 3, 5)  # Multiple seeds for robustness
        
        return params
    
    def _sample_parameters(self, trial) -> Dict[str, Any]:
        """Sample full parameter set"""
        params = {}
        
        # Training parameters
        params['epochs'] = trial.suggest_int('epochs', 80, 250)
        params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 5e-3)
        params['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
        
        # Architecture
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [128, 256, 512, 768])
        params['latent_dim'] = trial.suggest_categorical('latent_dim', [32, 64, 128, 256])
        
        # Regularization
        params['dropout_rate'] = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
        params['mask_rate'] = trial.suggest_uniform('mask_rate', 0.1, 0.4)
        
        # Loss weights - Prioritize cut loss minimization with balance constraint
        params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 5e-3)  # KL regularization
        params['beta'] = trial.suggest_uniform('beta', 10.0, 50.0)  # Much higher for cut loss priority
        params['gamma'] = trial.suggest_uniform('gamma', 0.5, 5.0)   # Lower range, just enough for balance
        
        # Advanced features
        params['adaptive_weights'] = trial.suggest_categorical('adaptive_weights', [True, False])
        params['loss_annealing'] = trial.suggest_categorical('loss_annealing', [True, False])
        
        # Feature normalization
        params['norm_method'] = trial.suggest_categorical('norm_method', list(NormalizationMethod))
        params['feature_scaling'] = trial.suggest_uniform('feature_scaling', 0.7, 1.5)
        params['spectral_norm'] = trial.suggest_categorical('spectral_norm', [True, False])
        
        # Optimization
        params['scheduler_type'] = trial.suggest_categorical('scheduler_type', 
            ['plateau', 'cosine', 'step'])
        params['scheduler_patience'] = trial.suggest_int('scheduler_patience', 5, 15)
        params['scheduler_factor'] = trial.suggest_uniform('scheduler_factor', 0.2, 0.7)
        params['gradient_clip'] = trial.suggest_uniform('gradient_clip', 0.3, 3.0)
        
        return params
    
    def _objective_function(self, trial, params: Dict, max_epochs: int = 500, 
                          final_validation: bool = False) -> Tuple[float, float, float, float]:
        """
        Objective function prioritizing cut loss minimization with balance constraint
        Balance constraint: imbalance < 2% (balance_loss < 0.02)
        Returns: (cut_loss, balance_penalty, -quality_score, -stability_score)
        """
        try:
            # Train model with given parameters
            results = self._train_with_params(params, max_epochs)
            
            cut_loss = results['final_cut_loss']
            balance_loss = results['final_balance_loss']
            quality_score = results['quality_score']
            stability_score = results['stability_score']
            
            # Convert balance loss to imbalance percentage (approximation)
            # balance_loss is squared relative error, so imbalance ≈ sqrt(balance_loss)
            imbalance_pct = torch.sqrt(torch.tensor(balance_loss) / 2.0).item() * 100.0
            
            # Apply penalty if balance constraint is violated (>4% imbalance)
            balance_penalty = balance_loss
            if imbalance_pct > 4.0:
                # Heavy penalty for constraint violation
                balance_penalty = balance_loss * 10.0 + (imbalance_pct - 4.0) ** 2

            # Early pruning if cut loss too high or severe imbalance
            if not final_validation and (cut_loss > 0.025 or imbalance_pct > 10.0) and trial.number > 10:
                raise optuna.TrialPruned()
            
            # Log trial results with imbalance percentage
            self.logger.info(f"Trial {trial.number}: cut={cut_loss:.4f}, balance={balance_loss:.6f}, "
                           f"imbalance={imbalance_pct:.2f}%, quality={quality_score:.3f}, stability={stability_score:.3f}")
            
            # Return objectives (minimize cut/balance_penalty, maximize quality/stability)
            return cut_loss, balance_penalty, -quality_score, -stability_score
            
        except Exception as e:
            self.logger.error(f"Trial failed: {str(e)}")
            # Return worst possible values for failed trials
            return 1.0, 1.0, -0.0, -0.0
    
    def _train_with_params(self, params: Dict, max_epochs: int) -> Dict:
        """Train model with given parameters and return real metrics"""
        import numpy as np
        
        # Import necessary modules
        from train import ISPDDataset
        from torch_geometric.loader import DataLoader
        from models import GraphPartitionModel
        
        try:
            # Setup training
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create dataset
            dataset = ISPDDataset(self.data_path)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            # Initialize model with parameters
            model = GraphPartitionModel(
                input_dim=7,
                hidden_dim=params.get('hidden_dim', 256),
                latent_dim=params.get('latent_dim', 64),
                num_partitions=2,
                use_hypergraph=True
            ).to(device)
            
            # Setup optimizer
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=params.get('learning_rate', 5e-5),
                weight_decay=params.get('weight_decay', 1e-5)
            )
            
            # Setup scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=params.get('scheduler_factor', 0.5), 
                patience=params.get('scheduler_patience', 8), verbose=False
            )
            
            # Training parameters
            alpha = params.get('alpha', 0.001)
            beta = params.get('beta', 20.0) 
            gamma = params.get('gamma', 1.5)
            
            # Training loop
            model.train()
            best_cut_loss = float('inf')
            best_balance_loss = float('inf')
            epochs_without_improvement = 0
            early_stop_patience = 15
            
            cut_losses_history = []
            balance_losses_history = []
            
            for epoch in range(max_epochs):
                epoch_losses = []
                epoch_cut_losses = []
                epoch_balance_losses = []
                epoch_kl_losses = []
                
                for batch_idx, data in enumerate(dataloader):
                    data = data.to(device)
                    optimizer.zero_grad()
                    
                    try:
                        Y = model(data)
                        num_nodes = data.x.shape[0]
                        num_nets = data.hyperedge_index[1][-1].item() + 1
                        
                        # Build sparse incidence matrix
                        W = torch.sparse_coo_tensor(
                            data.hyperedge_index, 
                            torch.ones(data.hyperedge_index.shape[1]).to(device), 
                            (num_nodes, num_nets)
                        ).to(device)
                        
                        # Node degrees  
                        D = torch.sparse.sum(W, dim=1).to_dense().unsqueeze(1)
                        
                        # Calculate losses
                        loss, kl_loss, cut_loss, balance_loss = model.combined_loss(
                            Y, W, D, alpha=alpha, beta=beta, gamma=gamma
                        )
                        
                        if not torch.isnan(loss):
                            loss.backward()
                            
                            # Gradient clipping
                            if 'gradient_clip' in params:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip'])
                            
                            optimizer.step()
                            
                            # Record losses
                            epoch_losses.append(loss.item())
                            epoch_cut_losses.append(cut_loss.item())
                            epoch_balance_losses.append(balance_loss.item()) 
                            epoch_kl_losses.append(kl_loss.item())
                        
                        # Early pruning for Optuna
                        # if trial and batch_idx > 5:  # After a few batches
                        #     avg_cut = np.mean(epoch_cut_losses)
                        #     if avg_cut > 0.6:  # Poor performance
                        #         trial.report(avg_cut, epoch)
                        #         if trial.should_prune():
                        #             raise optuna.TrialPruned()
                        
                    except Exception as e:
                        self.logger.warning(f"Batch {batch_idx} failed in epoch {epoch}: {e}")
                        continue
                
                # Calculate epoch metrics
                if epoch_losses:
                    avg_loss = np.mean(epoch_losses)
                    avg_cut_loss = np.mean(epoch_cut_losses)
                    avg_balance_loss = np.mean(epoch_balance_losses)
                    
                    cut_losses_history.append(avg_cut_loss)
                    balance_losses_history.append(avg_balance_loss)
                    
                    # Learning rate scheduling
                    scheduler.step(avg_loss)
                    
                    # Track best losses
                    improved = False
                    if avg_cut_loss < best_cut_loss:
                        best_cut_loss = avg_cut_loss
                        best_balance_loss = avg_balance_loss
                        improved = True
                    
                    if improved:
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    
                    # Report to Optuna for intermediate pruning
                    # if trial:
                    #     trial.report(avg_cut_loss, epoch)
                    #     if trial.should_prune():
                    #         raise optuna.TrialPruned()
                    
                    # Early stopping
                    if epochs_without_improvement >= early_stop_patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                        
                    # Log progress for important epochs
                    if epoch % max(1, max_epochs // 10) == 0 or epoch == max_epochs - 1:
                        imbalance_pct = np.sqrt(avg_balance_loss) * 100
                        self.logger.debug(
                            f"Epoch {epoch+1}: cut={avg_cut_loss:.4f}, "
                            f"balance={avg_balance_loss:.6f} ({imbalance_pct:.2f}%), loss={avg_loss:.4f}"
                        )
            
            # Calculate final metrics
            final_cut_loss = best_cut_loss
            final_balance_loss = best_balance_loss
            
            # Calculate quality and stability scores
            quality_score = self._calculate_quality_score(cut_losses_history, balance_losses_history)
            stability_score = self._calculate_stability_score(cut_losses_history, balance_losses_history)
            
            return {
                'final_cut_loss': float(final_cut_loss),
                'final_balance_loss': float(final_balance_loss),
                'quality_score': float(quality_score),
                'stability_score': float(stability_score),
                'converged_epoch': len(cut_losses_history),
                'imbalance_pct': float(np.sqrt(final_balance_loss) * 100)
            }
            
        except optuna.TrialPruned:
            # Re-raise pruning exception
            raise
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            # Return poor results for failed training
            return {
                'final_cut_loss': 1.0,
                'final_balance_loss': 1.0,
                'quality_score': 0.0,
                'stability_score': 0.0,
                'converged_epoch': 0,
                'imbalance_pct': 100.0
            }
    
    def _calculate_quality_score(self, cut_losses: List[float], balance_losses: List[float]) -> float:
        """Calculate quality score based on loss trajectories"""
        if not cut_losses or not balance_losses:
            return 0.0
        
        # Quality based on final losses and convergence
        final_cut = cut_losses[-1]
        final_balance = balance_losses[-1]
        
        # Score components
        cut_score = max(0, 1.0 - final_cut / 0.025)  # Good if cut < 0.025
        balance_score = max(0, 1.0 - final_balance / 0.001)  # Good if balance < 0.001 (2% imbalance)
        convergence_score = len(cut_losses) / max(50, len(cut_losses))  # Faster convergence is better
        
        return np.mean([cut_score, balance_score, convergence_score])
    
    def _calculate_stability_score(self, cut_losses: List[float], balance_losses: List[float]) -> float:
        """Calculate stability score based on loss variance"""
        if len(cut_losses) < 5 or len(balance_losses) < 5:
            return 0.0
        
        # Look at last 20% of training for stability
        window = max(5, len(cut_losses) // 5)
        recent_cut = cut_losses[-window:]
        recent_balance = balance_losses[-window:]
        
        # Lower variance = higher stability
        cut_var = np.var(recent_cut)
        balance_var = np.var(recent_balance)
        
        cut_stability = 1.0 / (1.0 + cut_var * 100)  # Normalized stability
        balance_stability = 1.0 / (1.0 + balance_var * 1000)
        
        return np.mean([cut_stability, balance_stability])
    
    def _analyze_results(self) -> Dict:
        """Analyze optimization results and find best configurations"""
        if not self.study.best_trials:
            return {}
        
        # Multi-criteria analysis
        best_trials = self.study.best_trials[:10]  # Top 10 trials
        
        analysis = {
            'best_overall': best_trials[0].params if best_trials else {},
            'best_cut_loss': None,
            'best_balance_loss': None,
            'best_quality': None,
            'parameter_importance': {},
            'convergence_analysis': {}
        }
        
        # Find specialized best configurations
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                cut_loss, balance_loss, neg_quality, _ = trial.values
                
                # Best cut loss (primary objective)
                if analysis['best_cut_loss'] is None or cut_loss < analysis['best_cut_loss']['cut_loss']:
                    analysis['best_cut_loss'] = {
                        'params': trial.params, 
                        'cut_loss': cut_loss,
                        'balance_loss': balance_loss,
                        'imbalance_pct': torch.sqrt(torch.tensor(balance_loss)).item() * 100
                    }
                
                # Best balance loss  
                if analysis['best_balance_loss'] is None or balance_loss < analysis['best_balance_loss']['balance_loss']:
                    analysis['best_balance_loss'] = {'params': trial.params, 'balance_loss': balance_loss}
                
                # Best quality
                quality = -neg_quality
                if analysis['best_quality'] is None or quality > analysis['best_quality']['quality']:
                    analysis['best_quality'] = {'params': trial.params, 'quality': quality}
        
        # Parameter importance analysis
        try:
            importance = optuna.importance.get_param_importances(self.study)
            analysis['parameter_importance'] = importance
        except:
            pass
        
        return analysis
    
    def _save_results(self, results: Dict):
        """Save comprehensive results"""
        # Save main results
        with open(self.output_dir / 'optimization_results.json', 'w') as f:
            # Convert non-serializable objects
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Save best configurations
        if 'best_config' in results and results['best_config']:
            with open(self.output_dir / 'best_config.json', 'w') as f:
                json.dump(results['best_config'], f, indent=2)
        
        # Save Optuna study
        with open(self.output_dir / 'optuna_study.pkl', 'wb') as f:
            pickle.dump(self.study, f)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
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
        elif hasattr(obj, '__dict__'):  # Custom objects
            return str(obj)
        else:
            return obj

def main():
    """Main optimization entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphPart Hyperparameter Optimization")
    parser.add_argument('--data-path', type=str, default='/data1/tongsb/GraphPart/dataset/pt/train', help='Path to training data')
    parser.add_argument('--output-dir', type=str, default='./hyperopt_results', help='Output directory')
    parser.add_argument('--n-trials', type=int, default=200, help='Number of optimization trials')
    parser.add_argument('--n-parallel', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = MultiObjectiveOptimizer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_parallel=args.n_parallel
    )
    
    results = optimizer.optimize()
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETED")
    print("="*50)
    
    if 'best_config' in results and results['best_config']:
        best = results['best_config']
        print(f"Best Cut Loss Configuration: {best.get('best_cut_loss', 'N/A')}")
        print(f"Best Balance Loss Configuration: {best.get('best_balance_loss', 'N/A')}")
        print(f"Best Quality Configuration: {best.get('best_quality', 'N/A')}")
        print(f"Results saved to: {args.output_dir}")
    
    return results

if __name__ == '__main__':
    main()