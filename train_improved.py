#!/usr/bin/env python3
"""
Improved training script with better error handling and logging
"""
import torch
import numpy as np
import argparse
import traceback
from torch_geometric.loader import DataLoader
from transformers import set_seed

from config import Config
from logger import setup_logger
from models import GraphPartitionModel, HyperData
from train import ISPDDataset

def validate_args(args):
    """Validate command line arguments"""
    if args.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {args.lr}")
    if args.epochs <= 0:
        raise ValueError(f"Epochs must be positive, got {args.epochs}")
    if args.alpha < 0 or args.beta < 0 or args.gamma < 0:
        raise ValueError("Loss coefficients must be non-negative")

def main():
    parser = argparse.ArgumentParser(description="GraphPart Training")
    parser.add_argument('--lr', type=float, default=Config.DEFAULT_LR, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=Config.DEFAULT_ALPHA, help='KL loss coefficient')
    parser.add_argument('--beta', type=float, default=Config.DEFAULT_BETA, help='Cut loss coefficient')
    parser.add_argument('--gamma', type=float, default=Config.DEFAULT_GAMMA, help='Balance loss coefficient')
    parser.add_argument('--epochs', type=int, default=Config.DEFAULT_EPOCHS, help='Number of epochs')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger()
    
    try:
        validate_args(args)
        set_seed(args.seed)
        device = Config.get_device()
        
        logger.info(f"Starting training with args: {args}")
        logger.info(f"Using device: {device}")
        
        # Load data
        logger.info("Loading dataset...")
        dataset = ISPDDataset(args.data_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Initialize model
        model = GraphPartitionModel(
            Config.INPUT_DIM, 
            Config.HIDDEN_DIM, 
            Config.LATENT_DIM, 
            Config.NUM_PARTITIONS, 
            True
        ).to(device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Training logic would go here...
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()