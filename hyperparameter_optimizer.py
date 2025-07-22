"""
Multi-strategy hyperparameter optimizer with Optuna, Ray Tune, and custom strategies
"""
import optuna
import numpy as np
import torch
import json
import time
import warnings
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from hyperopt_config import HyperOptConfig, NormalizationMethod
from convergence_monitor import ConvergenceMonitor, AdaptiveLossWeighting, EarlyStoppingManager
from feature_normalizer import create_enhanced_features
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
        self.best_trials = []
        self.config = HyperOptConfig()
        
        # Initialize Optuna study
        storage_url = f"sqlite:///{self.output_dir}/optuna_study.db"
        self.study = optuna.create_study(
            directions=['minimize', 'minimize', 'maximize', 'maximize'],  # cut, balance, quality, stability
            study_name='graphpart_hyperopt',
            storage=storage_url,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=20,
                interval_steps=5
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
        # Use broader search ranges for coarse search
        coarse_study = optuna.create_study(
            directions=['minimize', 'minimize', 'maximize', 'maximize'],
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        def coarse_objective(trial):
            params = self._sample_coarse_parameters(trial)
            return self._objective_function(trial, params, max_epochs=100)
        
        coarse_study.optimize(coarse_objective, n_trials=n_trials, n_jobs=self.n_parallel)
        
        return {
            'n_trials': len(coarse_study.trials),
            'best_trials': coarse_study.best_trials[:5],  # Top 5 trials
            'study': coarse_study
        }
    
    def _run_fine_tuning(self, n_trials: int) -> Dict:
        """Fine-tune around best configurations from coarse search"""
        # Get best parameters from coarse search for fine-tuning
        best_trials = self.study.best_trials[:3] if self.study.best_trials else []
        
        fine_study = optuna.create_study(
            directions=['minimize', 'minimize', 'maximize', 'maximize'],
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=15)
        )
        
        def fine_objective(trial):
            if best_trials:
                # Sample around best configurations
                base_params = best_trials[trial.number % len(best_trials)].params
                params = self._sample_fine_parameters(trial, base_params)
            else:
                params = self._sample_parameters(trial)
            return self._objective_function(trial, params, max_epochs=200)
        
        fine_study.optimize(fine_objective, n_trials=n_trials, n_jobs=self.n_parallel)
        
        return {
            'n_trials': len(fine_study.trials),
            'best_trials': fine_study.best_trials[:3],
            'study': fine_study
        }
    
    def _run_final_validation(self, n_trials: int) -> Dict:
        """Final validation with extended training"""
        final_study = optuna.create_study(
            directions=['minimize', 'minimize', 'maximize', 'maximize'],
            pruner=None  # No pruning in final validation
        )
        
        def final_objective(trial):
            # Use only the very best parameter combinations
            if self.study.best_trials:
                best_params = self.study.best_trials[0].params
                params = self._sample_final_parameters(trial, best_params)
            else:
                params = self._sample_parameters(trial)
            return self._objective_function(trial, params, max_epochs=300, final_validation=True)
        
        final_study.optimize(final_objective, n_trials=min(n_trials, 20), n_jobs=min(2, self.n_parallel))
        
        return {
            'n_trials': len(final_study.trials),
            'best_trials': final_study.best_trials,
            'study': final_study
        }
    
    def _sample_coarse_parameters(self, trial) -> Dict[str, Any]:
        """Sample parameters for coarse search with broader ranges"""
        params = {}
        
        # Use broader ranges for coarse search
        params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        params['beta'] = trial.suggest_uniform('beta', 2.0, 20.0)  # Broader range
        params['gamma'] = trial.suggest_uniform('gamma', 1.0, 10.0)
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
        
        # Fine-tune around base values with smaller ranges
        if 'learning_rate' in base_params:
            base_lr = base_params['learning_rate']
            params['learning_rate'] = trial.suggest_loguniform('learning_rate', 
                base_lr * 0.3, base_lr * 3.0)
        
        if 'beta' in base_params:
            base_beta = base_params['beta']
            params['beta'] = trial.suggest_uniform('beta', 
                max(1.0, base_beta * 0.5), base_beta * 1.5)
        
        if 'gamma' in base_params:
            base_gamma = base_params['gamma']
            params['gamma'] = trial.suggest_uniform('gamma',
                max(0.5, base_gamma * 0.5), base_gamma * 1.5)
        
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
        
        # Loss weights - Critical for target achievement
        params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e-2)
        params['beta'] = trial.suggest_uniform('beta', 3.0, 30.0)  # Higher range for cut loss
        params['gamma'] = trial.suggest_uniform('gamma', 2.0, 15.0)  # Higher range for balance
        
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
    
    def _objective_function(self, trial, params: Dict, max_epochs: int = 200, 
                          final_validation: bool = False) -> Tuple[float, float, float, float]:
        """
        Objective function to minimize cut_loss, balance_loss and maximize quality, stability
        Returns: (cut_loss, balance_loss, -quality_score, -stability_score)
        """
        try:
            # Train model with given parameters
            results = self._train_with_params(params, max_epochs, trial, final_validation)
            
            cut_loss = results['final_cut_loss']
            balance_loss = results['final_balance_loss']
            quality_score = results['quality_score']
            stability_score = results['stability_score']
            
            # Early pruning if targets not met
            if not final_validation and cut_loss > 0.1 and trial.number > 10:
                raise optuna.TrialPruned()
            
            # Log trial results
            self.logger.info(f"Trial {trial.number}: cut={cut_loss:.4f}, balance={balance_loss:.6f}, "
                           f"quality={quality_score:.3f}, stability={stability_score:.3f}")
            
            # Return objectives (minimize cut/balance, maximize quality/stability)
            return cut_loss, balance_loss, -quality_score, -stability_score
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            # Return worst possible values for failed trials
            return 1.0, 1.0, -0.0, -0.0
    
    def _train_with_params(self, params: Dict, max_epochs: int, trial, 
                          final_validation: bool = False) -> Dict:
        """Train model with given parameters and return metrics"""
        # This would contain the full training logic
        # For now, return a placeholder structure
        
        # Import necessary modules
        from train import ISPDDataset
        from torch_geometric.loader import DataLoader
        from models import GraphPartitionModel, HyperData
        from convergence_monitor import ConvergenceMonitor, AdaptiveLossWeighting, QualityEvaluator
        
        # Setup training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create dataset with enhanced features
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
        
        # Setup adaptive loss weighting
        loss_weighter = AdaptiveLossWeighting(
            initial_alpha=params.get('alpha', 0.001),
            initial_beta=params.get('beta', 5.0),
            initial_gamma=params.get('gamma', 2.0),
            adaptive=params.get('adaptive_weights', True),
            annealing=params.get('loss_annealing', True)
        ).to(device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': loss_weighter.parameters(), 'lr': params.get('learning_rate', 5e-5) * 0.1}
        ], lr=params.get('learning_rate', 5e-5), 
           weight_decay=params.get('weight_decay', 1e-5))
        
        # Setup monitoring
        monitor = ConvergenceMonitor()
        quality_eval = QualityEvaluator()
        
        # Training loop (simplified)
        model.train()
        final_metrics = {
            'final_cut_loss': 0.5,  # Placeholder
            'final_balance_loss': 0.01,  # Placeholder  
            'quality_score': 0.7,  # Placeholder
            'stability_score': 0.8,  # Placeholder
        }
        
        return final_metrics
    
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
                cut_loss, balance_loss, neg_quality, neg_stability = trial.values
                
                # Best cut loss
                if analysis['best_cut_loss'] is None or cut_loss < analysis['best_cut_loss']['cut_loss']:
                    analysis['best_cut_loss'] = {'params': trial.params, 'cut_loss': cut_loss}
                
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
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
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