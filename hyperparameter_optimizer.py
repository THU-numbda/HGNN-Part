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
import pandas as pd
import multiprocessing as mp
import swanlab

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
        
        # Initialize SwanLab for hyperparameter optimization tracking
        self.swanlab_run = swanlab.init(
            project="GraphPart",
            experiment_name=f"multi_objective_opt_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "data_path": data_path,
                "n_trials": n_trials,
                "n_parallel": n_parallel,
                "optimization_strategy": "multi_stage",
                "objectives": ["cut_loss", "balance_penalty", "quality_score", "stability_score"]
            }
        )
        
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
        """Sample parameters for coarse search with broader ranges (仅包含实际使用的参数)"""
        params = {}
        
        # 核心训练参数
        params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        params['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
        
        # 损失函数权重 - 优先cut loss
        params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e-2)
        params['beta'] = trial.suggest_uniform('beta', 15.0, 40.0)  # 更高cut loss权重
        params['gamma'] = trial.suggest_uniform('gamma', 0.3, 3.0)  # 较低balance权重
        
        # 模型架构参数
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        params['latent_dim'] = trial.suggest_categorical('latent_dim', [32, 64, 128])
        
        # 优化器调度参数
        params['scheduler_factor'] = trial.suggest_uniform('scheduler_factor', 0.2, 0.7)
        params['scheduler_patience'] = trial.suggest_int('scheduler_patience', 5, 15)
        
        # 训练技巧（可选）
        params['gradient_clip'] = trial.suggest_uniform('gradient_clip', 0.5, 2.0)
        
        return params
    
    def _sample_fine_parameters(self, trial, base_params: Dict) -> Dict[str, Any]:
        """Sample parameters around a good configuration for fine-tuning (仅包含实际使用的参数)"""
        params = base_params.copy()
        
        # 在基础参数附近细调核心参数
        if 'learning_rate' in base_params:
            base_lr = base_params['learning_rate']
            params['learning_rate'] = trial.suggest_loguniform('learning_rate', 
                base_lr * 0.3, base_lr * 3.0)
        
        if 'weight_decay' in base_params:
            base_wd = base_params['weight_decay']
            params['weight_decay'] = trial.suggest_loguniform('weight_decay',
                base_wd * 0.1, base_wd * 10.0)
        
        if 'beta' in base_params:
            base_beta = base_params['beta']
            # 保持beta高权重用于cut loss优先
            params['beta'] = trial.suggest_uniform('beta', 
                max(10.0, base_beta * 0.7), base_beta * 1.3)
        
        if 'gamma' in base_params:
            base_gamma = base_params['gamma']
            # 保持gamma在合理范围
            params['gamma'] = trial.suggest_uniform('gamma',
                max(0.1, base_gamma * 0.5), min(base_gamma * 1.5, 3.0))
        
        if 'alpha' in base_params:
            base_alpha = base_params['alpha']
            params['alpha'] = trial.suggest_loguniform('alpha',
                base_alpha * 0.1, base_alpha * 10.0)
        
        # 调度器参数微调
        if 'scheduler_factor' in base_params:
            base_factor = base_params['scheduler_factor']
            params['scheduler_factor'] = trial.suggest_uniform('scheduler_factor',
                max(0.1, base_factor * 0.7), min(base_factor * 1.3, 0.8))
        
        if 'scheduler_patience' in base_params:
            base_patience = base_params['scheduler_patience']
            params['scheduler_patience'] = trial.suggest_int('scheduler_patience',
                max(3, base_patience - 3), base_patience + 3)
        
        # 梯度裁剪微调
        if 'gradient_clip' in base_params:
            base_clip = base_params['gradient_clip']
            params['gradient_clip'] = trial.suggest_uniform('gradient_clip',
                max(0.1, base_clip * 0.5), base_clip * 2.0)
        
        return params
    
    def _sample_final_parameters(self, trial, best_params: Dict) -> Dict[str, Any]:
        """Sample parameters for final validation with minimal variation (仅包含实际使用的参数)"""
        params = best_params.copy()
        
        # 最小变化用于最终验证
        if 'learning_rate' in best_params:
            base_lr = best_params['learning_rate']
            params['learning_rate'] = trial.suggest_loguniform('learning_rate',
                base_lr * 0.8, base_lr * 1.2)
        
        # 其他参数保持不变或最小调整
        if 'weight_decay' in best_params:
            base_wd = best_params['weight_decay']
            params['weight_decay'] = trial.suggest_loguniform('weight_decay',
                base_wd * 0.5, base_wd * 2.0)
        
        return params
    
    def _sample_parameters(self, trial) -> Dict[str, Any]:
        """Sample full parameter set (仅包含实际使用的参数)"""
        params = {}
        
        # 核心训练参数
        params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 5e-3)
        params['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
        
        # 模型架构参数
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [128, 256, 512, 768])
        params['latent_dim'] = trial.suggest_categorical('latent_dim', [32, 64, 128, 256])
        
        # 损失函数权重 - 优先cut loss最小化和balance约束
        params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 5e-3)  # KL正则化
        params['beta'] = trial.suggest_uniform('beta', 10.0, 50.0)  # 更高cut loss权重
        params['gamma'] = trial.suggest_uniform('gamma', 0.5, 5.0)   # balance权重
        
        # 优化器调度参数
        params['scheduler_patience'] = trial.suggest_int('scheduler_patience', 5, 15)
        params['scheduler_factor'] = trial.suggest_uniform('scheduler_factor', 0.2, 0.7)
        
        # 训练技巧
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
            
            # Log trial results to SwanLab
            trial_data = {
                f"hyperopt/trial_{trial.number}/cut_loss": cut_loss,
                f"hyperopt/trial_{trial.number}/balance_loss": balance_loss,
                f"hyperopt/trial_{trial.number}/balance_penalty": balance_penalty,
                f"hyperopt/trial_{trial.number}/imbalance_pct": imbalance_pct,
                f"hyperopt/trial_{trial.number}/quality_score": quality_score,
                f"hyperopt/trial_{trial.number}/stability_score": stability_score,
                f"hyperopt/trial_{trial.number}/converged_epoch": results.get('converged_epoch', 0),
                
                # Log hyperparameters
                f"hyperopt/trial_{trial.number}/learning_rate": params.get('learning_rate', 0),
                f"hyperopt/trial_{trial.number}/alpha": params.get('alpha', 0),
                f"hyperopt/trial_{trial.number}/beta": params.get('beta', 0),
                f"hyperopt/trial_{trial.number}/gamma": params.get('gamma', 0),
                f"hyperopt/trial_{trial.number}/hidden_dim": params.get('hidden_dim', 0),
                f"hyperopt/trial_{trial.number}/latent_dim": params.get('latent_dim', 0),
                
                # Summary metrics for comparison
                "hyperopt/current_best_cut": cut_loss,
                "hyperopt/current_best_balance": balance_loss,
                "hyperopt/trials_completed": trial.number + 1
            }
            swanlab.log(trial_data)
            
            # Log trial results with imbalance percentage
            self.logger.info(f"Trial {trial.number}: cut={cut_loss:.4f}, balance={balance_loss:.6f}, "
                           f"imbalance={imbalance_pct:.2f}%, quality={quality_score:.3f}, stability={stability_score:.3f}")
            
            # Return objectives (minimize cut/balance_penalty, maximize quality/stability)
            return cut_loss, balance_penalty, -quality_score, -stability_score
            
        except Exception as e:
            self.logger.error(f"Trial failed: {str(e)}")
            # Return worst possible values for failed trials
            swanlab.log({
                f"hyperopt/trial_{trial.number}/failed": True,
                f"hyperopt/trial_{trial.number}/error": str(e)
            })
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
    
    def _merge_best_parameters(self) -> Dict:
        """从不同trial中合并最佳参数组合"""
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            return {}
        
        # 按目标值排序找到最佳试验
        best_cut_trial = min(completed_trials, key=lambda t: t.values[0])  # 最低cut loss
        best_balance_trial = min(completed_trials, key=lambda t: t.values[1])  # 最低balance loss
        
        merged_config = {}
        
        # 优先使用cut loss最佳的核心参数
        core_params = ['learning_rate', 'alpha', 'beta', 'gamma', 'hidden_dim', 'latent_dim', 'weight_decay']
        for param in core_params:
            if param in best_cut_trial.params:
                merged_config[param] = best_cut_trial.params[param]
        
        # 补充其他参数 - 从性能最佳的trial中获取
        all_params_found = set()
        for trial in completed_trials:
            all_params_found.update(trial.params.keys())
        
        for param in all_params_found:
            if param not in merged_config:
                # 从最佳性能trial中获取（按cut loss排序）
                best_trial_for_param = min(
                    [t for t in completed_trials if param in t.params],
                    key=lambda t: t.values[0]
                )
                merged_config[param] = best_trial_for_param.params[param]
        
        self.logger.info(f"Merged parameters from {len(completed_trials)} trials, found {len(merged_config)} parameters")
        return merged_config
    
    def _get_complete_best_config(self) -> Dict:
        """获取包含所有参数的完整最佳配置"""
        if not self.study.best_trials:
            return {}
        
        # 首先尝试合并最佳参数
        merged_config = self._merge_best_parameters()
        
        # 定义默认参数（仅包含实际使用的参数）
        default_params = {
            # 核心训练参数
            'learning_rate': 5e-5,
            'weight_decay': 1e-5,
            
            # 模型架构参数
            'hidden_dim': 256,
            'latent_dim': 64,
            
            # 损失函数权重
            'alpha': 0.001,     # KL散度权重
            'beta': 20.0,       # Cut损失权重
            'gamma': 1.5,       # Balance损失权重
            
            # 优化器调度参数
            'scheduler_factor': 0.5,
            'scheduler_patience': 8,
            
            # 训练技巧
            'gradient_clip': 1.0
        }
        
        # 填充缺失参数
        complete_config = merged_config.copy()
        missing_params = []
        
        for param, default_value in default_params.items():
            if param not in complete_config:
                complete_config[param] = default_value
                missing_params.append(param)
        
        if missing_params:
            self.logger.info(f"Filled {len(missing_params)} missing parameters with defaults: {missing_params}")
        
        # 参数验证和调整
        complete_config = self._validate_and_adjust_config(complete_config)
        
        return complete_config
    
    def _validate_and_adjust_config(self, config: Dict) -> Dict:
        """验证和调整配置参数（仅检查实际使用的参数）"""
        adjusted_config = config.copy()
        
        # 确保数值参数在合理范围内
        adjustments = []
        
        # 学习率范围检查
        if adjusted_config.get('learning_rate', 0) < 1e-6:
            adjusted_config['learning_rate'] = 1e-5
            adjustments.append('learning_rate (too small)')
        elif adjusted_config.get('learning_rate', 0) > 1e-1:
            adjusted_config['learning_rate'] = 1e-2
            adjustments.append('learning_rate (too large)')
        
        # 权重衰减检查
        if adjusted_config.get('weight_decay', 0) < 1e-7:
            adjusted_config['weight_decay'] = 1e-6
            adjustments.append('weight_decay (too small)')
        elif adjusted_config.get('weight_decay', 0) > 1e-2:
            adjusted_config['weight_decay'] = 1e-3
            adjustments.append('weight_decay (too large)')
        
        # Beta和Gamma权重检查
        if adjusted_config.get('beta', 0) < 1.0:
            adjusted_config['beta'] = 5.0
            adjustments.append('beta (too small)')
        elif adjusted_config.get('beta', 0) > 100.0:
            adjusted_config['beta'] = 50.0
            adjustments.append('beta (too large)')
            
        if adjusted_config.get('gamma', 0) < 0.1:
            adjusted_config['gamma'] = 0.5
            adjustments.append('gamma (too small)')
        elif adjusted_config.get('gamma', 0) > 10.0:
            adjusted_config['gamma'] = 5.0
            adjustments.append('gamma (too large)')
        
        # Alpha权重检查
        if adjusted_config.get('alpha', 0) < 1e-6:
            adjusted_config['alpha'] = 1e-5
            adjustments.append('alpha (too small)')
        elif adjusted_config.get('alpha', 0) > 1e-1:
            adjusted_config['alpha'] = 1e-2
            adjustments.append('alpha (too large)')
        
        # 模型维度检查
        if adjusted_config.get('hidden_dim', 0) not in [128, 256, 512, 768]:
            adjusted_config['hidden_dim'] = 256
            adjustments.append('hidden_dim (invalid value)')
        
        if adjusted_config.get('latent_dim', 0) not in [32, 64, 128, 256]:
            adjusted_config['latent_dim'] = 64
            adjustments.append('latent_dim (invalid value)')
        
        # 调度器参数检查
        if adjusted_config.get('scheduler_factor', 0) <= 0 or adjusted_config.get('scheduler_factor', 0) >= 1:
            adjusted_config['scheduler_factor'] = 0.5
            adjustments.append('scheduler_factor (invalid range)')
        
        if adjusted_config.get('scheduler_patience', 0) < 1:
            adjusted_config['scheduler_patience'] = 5
            adjustments.append('scheduler_patience (too small)')
        elif adjusted_config.get('scheduler_patience', 0) > 50:
            adjusted_config['scheduler_patience'] = 15
            adjustments.append('scheduler_patience (too large)')
        
        # 梯度裁剪检查
        if adjusted_config.get('gradient_clip', 0) <= 0:
            adjusted_config['gradient_clip'] = 0.5
            adjustments.append('gradient_clip (too small)')
        elif adjusted_config.get('gradient_clip', 0) > 10.0:
            adjusted_config['gradient_clip'] = 3.0
            adjustments.append('gradient_clip (too large)')
        
        if adjustments:
            self.logger.info(f"Adjusted invalid parameters: {adjustments}")
        
        return adjusted_config
    
    def _analyze_results(self) -> Dict:
        """Analyze optimization results and find best configurations"""
        if not self.study.best_trials:
            return {}
        
        # Multi-criteria analysis
        best_trials = self.study.best_trials[:10]  # Top 10 trials
        
        # 获取完整的最佳配置（新的核心功能）
        complete_best_config = self._get_complete_best_config()
        
        analysis = {
            'best_overall': complete_best_config,  # 使用完整配置替代原始best_overall
            'best_overall_incomplete': best_trials[0].params if best_trials else {},  # 保留原始数据供参考
            'best_cut_loss': None,
            'best_balance_loss': None,
            'best_quality': None,
            'parameter_importance': {},
            'convergence_analysis': {},
            'config_completeness': {
                'complete_params_count': len(complete_best_config),
                'merged_from_trials': len(self.study.trials),
                'has_all_core_params': all(param in complete_best_config for param in [
                    'learning_rate', 'alpha', 'beta', 'gamma', 'hidden_dim', 'latent_dim', 'weight_decay'
                ])
            }
        }
        
        # Track best configurations for SwanLab logging
        best_cut_loss = float('inf')
        best_balance_loss = float('inf')
        best_quality = -float('inf')
        
        # Find specialized best configurations
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                cut_loss, balance_loss, neg_quality, _ = trial.values
                quality = -neg_quality
                
                # Best cut loss (primary objective)
                if analysis['best_cut_loss'] is None or cut_loss < analysis['best_cut_loss']['cut_loss']:
                    analysis['best_cut_loss'] = {
                        'params': trial.params, 
                        'cut_loss': cut_loss,
                        'balance_loss': balance_loss,
                        'imbalance_pct': torch.sqrt(torch.tensor(balance_loss)).item() * 100,
                        'trial_id': trial.number
                    }
                    
                    if cut_loss < best_cut_loss:
                        best_cut_loss = cut_loss
                
                # Best balance loss  
                if analysis['best_balance_loss'] is None or balance_loss < analysis['best_balance_loss']['balance_loss']:
                    analysis['best_balance_loss'] = {
                        'params': trial.params, 
                        'balance_loss': balance_loss,
                        'trial_id': trial.number
                    }
                    
                    if balance_loss < best_balance_loss:
                        best_balance_loss = balance_loss
                
                # Best quality
                if analysis['best_quality'] is None or quality > analysis['best_quality']['quality']:
                    analysis['best_quality'] = {
                        'params': trial.params, 
                        'quality': quality,
                        'trial_id': trial.number
                    }
                    
                    if quality > best_quality:
                        best_quality = quality
        
        # Log best configurations to SwanLab
        if analysis['best_cut_loss']:
            swanlab.log({
                "hyperopt/final/best_cut_loss": best_cut_loss,
                "hyperopt/final/best_cut_trial_id": analysis['best_cut_loss']['trial_id'],
                "hyperopt/final/best_cut_imbalance_pct": analysis['best_cut_loss']['imbalance_pct']
            })
        
        if analysis['best_balance_loss']:
            swanlab.log({
                "hyperopt/final/best_balance_loss": best_balance_loss,
                "hyperopt/final/best_balance_trial_id": analysis['best_balance_loss']['trial_id']
            })
        
        if analysis['best_quality']:
            swanlab.log({
                "hyperopt/final/best_quality_score": best_quality,
                "hyperopt/final/best_quality_trial_id": analysis['best_quality']['trial_id']
            })
        
        # Parameter importance analysis
        try:
            importance = optuna.importance.get_param_importances(self.study)
            analysis['parameter_importance'] = importance
            
            # Log parameter importance to SwanLab
            for param, imp in importance.items():
                swanlab.log({f"hyperopt/param_importance/{param}": imp})
                
        except Exception as e:
            self.logger.warning(f"Could not compute parameter importance: {e}")
            pass
        
        # 添加配置比较信息
        self._add_config_comparison(analysis)
        
        return analysis
    
    def _add_config_comparison(self, analysis: Dict):
        """添加不同配置之间的比较信息"""
        if not analysis['best_overall']:
            return
        
        complete_config = analysis['best_overall']
        incomplete_config = analysis['best_overall_incomplete']
        
        # 比较完整配置和不完整配置的差异
        missing_in_incomplete = set(complete_config.keys()) - set(incomplete_config.keys())
        different_values = {}
        
        for key in incomplete_config.keys():
            if key in complete_config and incomplete_config[key] != complete_config[key]:
                different_values[key] = {
                    'incomplete': incomplete_config[key],
                    'complete': complete_config[key]
                }
        
        analysis['config_comparison'] = {
            'missing_params_in_original': list(missing_in_incomplete),
            'different_values': different_values,
            'params_added_count': len(missing_in_incomplete),
            'params_changed_count': len(different_values)
        }
        
        self.logger.info(f"Config comparison: {len(missing_in_incomplete)} params added, {len(different_values)} params changed")
    
    def get_complete_best_config(self) -> Dict:
        """公共接口：获取完整的最佳配置"""
        return self._get_complete_best_config()
    
    def _save_results(self, results: Dict):
        """Save comprehensive results"""
        # Save main results
        with open(self.output_dir / 'optimization_results.json', 'w') as f:
            # Convert non-serializable objects
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Save best configurations (原始和完整)
        if 'best_config' in results and results['best_config']:
            # 保存原有的best_config（向后兼容）
            with open(self.output_dir / 'best_config.json', 'w') as f:
                json.dump(results['best_config'], f, indent=2)
            
            # 保存完整的最佳配置（新功能）
            complete_config = results['best_config'].get('best_overall', {})
            if complete_config:
                with open(self.output_dir / 'best_config_complete.json', 'w') as f:
                    json.dump(complete_config, f, indent=2)
                
                # 同时保存一个带元数据的完整配置文件
                complete_with_metadata = {
                    'config': complete_config,
                    'metadata': {
                        'generated_by': 'GraphPart Multi-stage Hyperparameter Optimizer',
                        'timestamp': pd.Timestamp.now().isoformat(),
                        'total_trials': len(self.study.trials),
                        'best_trial_id': results['best_config'].get('best_overall_incomplete', {}).get('trial_id', 'unknown'),
                        'completeness_info': results['best_config'].get('config_completeness', {}),
                        'comparison_info': results['best_config'].get('config_comparison', {}),
                        'note': 'This configuration combines the best parameters from multiple optimization trials and fills missing parameters with validated defaults.'
                    }
                }
                
                with open(self.output_dir / 'best_config_complete_with_metadata.json', 'w') as f:
                    json.dump(complete_with_metadata, f, indent=2)
                
                # Log best config parameters to SwanLab (no file saving)
                swanlab.log({
                    "hyperopt/best_config/learning_rate": complete_config.get('learning_rate', 0),
                    "hyperopt/best_config/alpha": complete_config.get('alpha', 0),
                    "hyperopt/best_config/beta": complete_config.get('beta', 0),
                    "hyperopt/best_config/gamma": complete_config.get('gamma', 0),
                    "hyperopt/best_config/hidden_dim": complete_config.get('hidden_dim', 0),
                    "hyperopt/best_config/latent_dim": complete_config.get('latent_dim', 0),
                    "hyperopt/best_config/weight_decay": complete_config.get('weight_decay', 0)
                })
                
                # Log final summary to SwanLab
                swanlab.log({
                    "hyperopt/summary/total_trials": len(self.study.trials),
                    "hyperopt/summary/successful_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                    "hyperopt/summary/complete_config_params": len(complete_config),
                    "hyperopt/summary/optimization_completed": True
                })
                
                self.logger.info(f"Saved complete best config with {len(complete_config)} parameters")
        
        # Save Optuna study
        with open(self.output_dir / 'optuna_study.pkl', 'wb') as f:
            pickle.dump(self.study, f)
        
        self.logger.info(f"Results saved to {self.output_dir}")
        self.logger.info("Files generated:")
        self.logger.info("  - optimization_results.json (complete results)")
        self.logger.info("  - best_config.json (original best config)")
        self.logger.info("  - best_config_complete.json (complete best config)")
        self.logger.info("  - best_config_complete_with_metadata.json (complete config with metadata)")
        self.logger.info("  - optuna_study.pkl (Optuna study object)")
        
        # Finish SwanLab run
        swanlab.finish()
    
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