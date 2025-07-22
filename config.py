# Configuration file for GraphPart
import torch

class Config:
    # Model parameters
    INPUT_DIM = 7
    HIDDEN_DIM = 256
    LATENT_DIM = 64
    NUM_PARTITIONS = 2
    
    # Training parameters
    DEFAULT_LR = 5e-5
    DEFAULT_ALPHA = 0.0005
    DEFAULT_BETA = 5
    DEFAULT_GAMMA = 1
    DEFAULT_EPOCHS = 50
    WEIGHT_DECAY = 1e-5
    
    # Training settings
    GRADIENT_CLIP_NORM = 1.0
    DROPOUT_RATE = 0.1
    MASK_RATE = 0.2
    
    # Early stopping
    PATIENCE = 10
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_FACTOR = 0.5
    
    # Numerical stability
    EPS = 1e-8
    LOGSTD_MIN = -10
    LOGSTD_MAX = 10
    
    # Device settings
    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    DATA_DIR = './data'
    MODELS_DIR = './models'
    EXEC_DIR = './exec'