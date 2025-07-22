"""
Hyperparameter optimization configuration for GraphPart
Target: cut_loss < 0.02, balance_loss < 1e-3, high-quality partitioning
"""
import numpy as np
from enum import Enum

class NormalizationMethod(Enum):
    STANDARD = "standard"  # Current method: normalize to degree feature norm
    MIN_MAX = "min_max"    # Min-max scaling [0, 1]
    ROBUST = "robust"      # Robust scaling using median and IQR
    UNIT_NORM = "unit_norm" # L2 unit normalization
    Z_SCORE = "z_score"    # Z-score standardization
    POWER = "power"        # Power transformation + normalization

class HyperOptConfig:
    """Comprehensive hyperparameter search space"""
    
    # Training parameters
    SEARCH_SPACE = {
        # Basic training parameters
        'epochs': {'type': 'int', 'low': 50, 'high': 300},
        'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},
        'weight_decay': {'type': 'loguniform', 'low': 1e-6, 'high': 1e-3},
        
        # Model architecture
        'hidden_dim': {'type': 'choice', 'choices': [128, 256, 512, 768]},
        'latent_dim': {'type': 'choice', 'choices': [32, 64, 128, 256]},
        'num_layers': {'type': 'int', 'low': 2, 'high': 6},
        
        # Regularization
        'dropout_rate': {'type': 'uniform', 'low': 0.0, 'high': 0.5},
        'mask_rate': {'type': 'uniform', 'low': 0.1, 'high': 0.4},
        'layer_dropout': {'type': 'uniform', 'low': 0.0, 'high': 0.3},
        
        # Loss function weights - Critical for achieving target losses
        'alpha': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},  # KL weight
        'beta': {'type': 'uniform', 'low': 1.0, 'high': 50.0},      # Cut weight (high)
        'gamma': {'type': 'uniform', 'low': 0.5, 'high': 20.0},     # Balance weight (high)
        
        # Advanced loss weighting
        'adaptive_weights': {'type': 'categorical', 'choices': [True, False]},
        'loss_annealing': {'type': 'categorical', 'choices': [True, False]},
        
        # Feature normalization
        'norm_method': {'type': 'categorical', 'choices': list(NormalizationMethod)},
        'feature_scaling': {'type': 'uniform', 'low': 0.5, 'high': 2.0},
        'spectral_norm': {'type': 'categorical', 'choices': [True, False]},
        
        # Learning rate scheduling
        'scheduler_type': {'type': 'categorical', 'choices': ['plateau', 'cosine', 'step', 'exponential']},
        'scheduler_patience': {'type': 'int', 'low': 3, 'high': 15},
        'scheduler_factor': {'type': 'uniform', 'low': 0.1, 'high': 0.8},
        
        # Early stopping
        'patience': {'type': 'int', 'low': 10, 'high': 30},
        'min_delta': {'type': 'loguniform', 'low': 1e-6, 'high': 1e-3},
        
        # Gradient optimization
        'gradient_clip': {'type': 'uniform', 'low': 0.1, 'high': 5.0},
        'optimizer_type': {'type': 'categorical', 'choices': ['adam', 'adamw', 'rmsprop']},
        'beta1': {'type': 'uniform', 'low': 0.8, 'high': 0.95},
        'beta2': {'type': 'uniform', 'low': 0.9, 'high': 0.999},
        
        # Batch and sampling
        'sampling_temperature': {'type': 'uniform', 'low': 0.1, 'high': 2.0},
        'num_samples': {'type': 'int', 'low': 5, 'high': 20},
        
        # Model-specific parameters
        'use_residual': {'type': 'categorical', 'choices': [True, False]},
        'use_attention': {'type': 'categorical', 'choices': [True, False]},
        'activation': {'type': 'categorical', 'choices': ['relu', 'gelu', 'swish', 'leaky_relu']},
    }
    
    # Target objectives - Multi-objective optimization
    OBJECTIVES = {
        'cut_loss': {'target': 0.02, 'weight': 10.0, 'direction': 'minimize'},
        'balance_loss': {'target': 1e-3, 'weight': 5.0, 'direction': 'minimize'},
        'kl_loss': {'target': 0.5, 'weight': 1.0, 'direction': 'stabilize'},  # Keep flexible
        'partition_quality': {'target': 0.95, 'weight': 8.0, 'direction': 'maximize'},
        'convergence_stability': {'target': 0.9, 'weight': 3.0, 'direction': 'maximize'},
    }
    
    # Convergence criteria
    CONVERGENCE_CRITERIA = {
        'cut_loss_threshold': 0.02,
        'balance_loss_threshold': 1e-3,
        'min_improvement': 1e-5,
        'stability_window': 10,
        'max_trials': 200,
        'max_epochs_per_trial': 300,
    }
    
    # Pruning criteria for early trial termination
    PRUNING_CRITERIA = {
        'min_epochs_before_pruning': 20,
        'cut_loss_prune_threshold': 0.5,
        'balance_loss_prune_threshold': 0.1,
        'no_improvement_epochs': 15,
    }