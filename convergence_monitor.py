"""
Advanced loss convergence monitoring and adaptive training
"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Dict, List, Optional, Tuple
import math

class ConvergenceMonitor:
    """Monitor loss convergence with advanced criteria"""
    
    def __init__(self, target_cut_loss=0.02, target_balance_loss=1e-3, 
                 stability_window=10, min_improvement=1e-5):
        self.target_cut_loss = target_cut_loss
        self.target_balance_loss = target_balance_loss
        self.stability_window = stability_window
        self.min_improvement = min_improvement
        
        # Loss history tracking
        self.cut_losses = deque(maxlen=stability_window)
        self.balance_losses = deque(maxlen=stability_window)
        self.kl_losses = deque(maxlen=stability_window)
        self.total_losses = deque(maxlen=stability_window)
        
        # Convergence metrics
        self.best_cut_loss = float('inf')
        self.best_balance_loss = float('inf')
        self.epochs_since_improvement = 0
        self.convergence_achieved = False
        self.stability_score = 0.0
        
    def update(self, cut_loss: float, balance_loss: float, kl_loss: float) -> Dict:
        """Update loss history and check convergence"""
        self.cut_losses.append(cut_loss)
        self.balance_losses.append(balance_loss)
        self.kl_losses.append(kl_loss)
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability()
        
        # Check for improvement
        improved = False
        if cut_loss < self.best_cut_loss - self.min_improvement:
            self.best_cut_loss = cut_loss
            improved = True
            
        if balance_loss < self.best_balance_loss - self.min_improvement:
            self.best_balance_loss = balance_loss
            improved = True
            
        if improved:
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
            
        # Check convergence criteria
        self.convergence_achieved = (
            cut_loss <= self.target_cut_loss and 
            balance_loss <= self.target_balance_loss and
            stability_metrics['cut_stability'] > 0.8 and
            stability_metrics['balance_stability'] > 0.8
        )
        
        return {
            'converged': self.convergence_achieved,
            'cut_target_met': cut_loss <= self.target_cut_loss,
            'balance_target_met': balance_loss <= self.target_balance_loss,
            'epochs_since_improvement': self.epochs_since_improvement,
            'stability_metrics': stability_metrics,
            'should_stop': self.epochs_since_improvement > 20 and not self.convergence_achieved
        }
    
    def _calculate_stability(self) -> Dict:
        """Calculate stability metrics for recent losses"""
        if len(self.cut_losses) < self.stability_window:
            return {'cut_stability': 0.0, 'balance_stability': 0.0, 'overall_stability': 0.0}
        
        cut_arr = np.array(self.cut_losses)
        balance_arr = np.array(self.balance_losses)
        
        # Calculate coefficient of variation (lower is more stable)
        cut_cv = np.std(cut_arr) / (np.mean(cut_arr) + 1e-8)
        balance_cv = np.std(balance_arr) / (np.mean(balance_arr) + 1e-8)
        
        # Convert to stability score (higher is more stable)
        cut_stability = max(0.0, 1.0 - cut_cv)
        balance_stability = max(0.0, 1.0 - balance_cv)
        overall_stability = (cut_stability + balance_stability) / 2.0
        
        return {
            'cut_stability': cut_stability,
            'balance_stability': balance_stability,
            'overall_stability': overall_stability,
            'cut_cv': cut_cv,
            'balance_cv': balance_cv
        }

class AdaptiveLossWeighting(nn.Module):
    """Adaptive loss weighting based on convergence progress"""
    
    def __init__(self, initial_alpha=0.001, initial_beta=5.0, initial_gamma=2.0, 
                 adaptive=True, annealing=True):
        super().__init__()
        
        if adaptive:
            # Learnable loss weights
            self.log_alpha = nn.Parameter(torch.tensor(math.log(initial_alpha)))
            self.log_beta = nn.Parameter(torch.tensor(math.log(initial_beta))) 
            self.log_gamma = nn.Parameter(torch.tensor(math.log(initial_gamma)))
        else:
            # Fixed weights
            self.register_buffer('log_alpha', torch.tensor(math.log(initial_alpha)))
            self.register_buffer('log_beta', torch.tensor(math.log(initial_beta)))
            self.register_buffer('log_gamma', torch.tensor(math.log(initial_gamma)))
            
        self.adaptive = adaptive
        self.annealing = annealing
        self.epoch = 0
        self.target_cut = 0.02
        self.target_balance = 1e-3
        
    def forward(self, kl_loss, cut_loss, balance_loss):
        """Compute weighted loss with adaptive weights"""
        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)
        gamma = torch.exp(self.log_gamma)
        
        if self.annealing and self.training:
            # Anneal KL weight over time
            kl_anneal = min(1.0, self.epoch / 50.0)
            alpha = alpha * kl_anneal
            
            # Increase cut and balance weights if not meeting targets
            if cut_loss > self.target_cut:
                beta = beta * (1 + (cut_loss / self.target_cut - 1) * 0.5)
            if balance_loss > self.target_balance:
                gamma = gamma * (1 + (balance_loss / self.target_balance - 1) * 0.5)
        
        total_loss = alpha * kl_loss + beta * cut_loss + gamma * balance_loss
        
        return total_loss, {'alpha': alpha.item(), 'beta': beta.item(), 'gamma': gamma.item()}
    
    def step_epoch(self):
        """Call at the end of each epoch"""
        self.epoch += 1

class QualityEvaluator:
    """Evaluate partition quality against traditional methods"""
    
    def __init__(self):
        self.quality_history = []
        
    def evaluate_partition_quality(self, model_partition, hypergraph_vertices, 
                                 hypergraph_edges, baseline_cut=None):
        """
        Evaluate partition quality:
        1. Cut quality vs baseline
        2. Balance quality
        3. Consistency metrics
        """
        from utils import evaluate_partition
        
        # Basic metrics
        cut, imbalance = evaluate_partition(model_partition, hypergraph_vertices, 
                                          hypergraph_edges, 2)
        
        # Partition balance (ideal is 0)
        partition_sizes = np.bincount(model_partition.astype(int))
        ideal_size = len(hypergraph_vertices) / 2
        balance_score = 1.0 - abs(partition_sizes[0] - ideal_size) / ideal_size
        
        # Cut quality (lower cut is better)
        if baseline_cut is not None:
            cut_quality = max(0.0, 1.0 - (cut / baseline_cut - 1.0))
        else:
            # Normalize by number of edges as rough baseline
            cut_quality = max(0.0, 1.0 - cut / len(hypergraph_edges))
            
        # Overall quality score
        quality_score = 0.6 * cut_quality + 0.4 * balance_score
        
        metrics = {
            'cut': cut,
            'imbalance': imbalance,
            'balance_score': balance_score,
            'cut_quality': cut_quality,
            'overall_quality': quality_score
        }
        
        self.quality_history.append(metrics)
        return metrics
    
    def get_quality_trend(self, window=10):
        """Get recent quality trend"""
        if len(self.quality_history) < 2:
            return 0.0
            
        recent = self.quality_history[-window:]
        if len(recent) < 2:
            return 0.0
            
        scores = [m['overall_quality'] for m in recent]
        trend = (scores[-1] - scores[0]) / len(scores)
        return trend

class EarlyStoppingManager:
    """Advanced early stopping with multiple criteria"""
    
    def __init__(self, patience=15, min_delta=1e-5, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.best_model_state = None
        self.should_stop = False
        
    def __call__(self, score, model, epoch):
        """Check if training should stop"""
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights and self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)
                
        return self.should_stop