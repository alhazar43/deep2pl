#!/usr/bin/env python3
"""
Ready-to-run script for Torch style (runtime k-fold) datasets.
Usage: python run_torch_style.py [dataset] [fold]
"""

import sys
import argparse
from utils.config import Config
from train import train_model

def get_torch_config(dataset_name="assist2015", fold_idx=0):
    """Get configuration for Torch style training."""
    config = Config()
    
    # Data configuration
    config.data_style = "torch"
    config.data_dir = "./data-orig"
    config.dataset_name = dataset_name
    config.k_fold = 5
    config.fold_idx = fold_idx
    
    # Dataset-specific configurations
    if dataset_name == "assist2015":
        config.n_questions = 100
        config.seq_len = 50
        config.batch_size = 32
        config.n_epochs = 20
        config.use_discrimination = True  # 2PL model for assist2015
        config.save_dir = f"checkpoints_torch_{dataset_name}_fold{fold_idx}"
        config.log_dir = f"logs_torch_{dataset_name}_fold{fold_idx}"
    elif dataset_name == "assist2017":
        config.n_questions = 102
        config.seq_len = 50
        config.batch_size = 32
        config.n_epochs = 20
        config.use_discrimination = True
        config.save_dir = f"checkpoints_torch_{dataset_name}_fold{fold_idx}"
        config.log_dir = f"logs_torch_{dataset_name}_fold{fold_idx}"
    elif dataset_name == "kddcup2010":
        config.n_questions = 661
        config.seq_len = 50
        config.batch_size = 16  # Smaller batch for large dataset
        config.n_epochs = 15
        config.use_discrimination = False
        config.save_dir = f"checkpoints_torch_{dataset_name}_fold{fold_idx}"
        config.log_dir = f"logs_torch_{dataset_name}_fold{fold_idx}"
    elif dataset_name == "statics2011":
        config.n_questions = 1223
        config.seq_len = 50
        config.batch_size = 16  # Smaller batch for large dataset
        config.n_epochs = 15
        config.use_discrimination = False
        config.save_dir = f"checkpoints_torch_{dataset_name}_fold{fold_idx}"
        config.log_dir = f"logs_torch_{dataset_name}_fold{fold_idx}"
    elif dataset_name == "synthetic":
        config.n_questions = 50
        config.seq_len = 30
        config.batch_size = 32
        config.n_epochs = 15
        config.use_discrimination = False
        config.save_dir = f"checkpoints_torch_{dataset_name}_fold{fold_idx}"
        config.log_dir = f"logs_torch_{dataset_name}_fold{fold_idx}"
    else:
        print(f"Warning: Unknown dataset {dataset_name}, using default settings")
        config.n_questions = 100
        config.use_discrimination = False
    
    # Model configuration
    config.memory_size = 50
    config.key_memory_state_dim = 50
    config.value_memory_state_dim = 200
    config.summary_vector_dim = 50
    config.q_embed_dim = 50
    config.qa_embed_dim = 200
    config.ability_scale = 3.0
    config.dropout_rate = 0.1
    
    # Training configuration
    config.learning_rate = 0.001
    config.max_grad_norm = 5.0
    config.weight_decay = 1e-5
    config.eval_every = 5
    config.save_every = 10
    config.verbose = True
    config.tensorboard = False
    config.seed = 42
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Train Deep-IRT with Torch style datasets')
    parser.add_argument('--dataset', type=str, default='assist2015',
                        choices=['assist2015', 'assist2017', 'kddcup2010', 'statics2011', 'synthetic'],
                        help='Dataset name')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index (0-4)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides default)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides default)')
    
    args = parser.parse_args()
    
    # Validate fold index
    if args.fold < 0 or args.fold >= 5:
        print("Error: Fold index must be between 0 and 4")
        sys.exit(1)
    
    print(f"=== Training Deep-IRT with Torch Style ===")
    print(f"Dataset: {args.dataset}")
    print(f"Fold: {args.fold}")
    
    # Get configuration
    config = get_torch_config(args.dataset, args.fold)
    
    # Override with command line arguments
    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    print(f"Training for {config.n_epochs} epochs with batch size {config.batch_size}")
    print(f"Model type: {'2PL' if config.use_discrimination else '1PL'}")
    
    # Start training
    train_model(config)

if __name__ == "__main__":
    main()