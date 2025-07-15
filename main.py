#!/usr/bin/env python3
"""
Main Pipeline Script for Deep Item Response Theory Model

This module provides a unified interface for training Deep-IRT models and 
generating comprehensive visualizations. It supports both individual component 
execution and complete pipeline workflows.

Features:
- Unified training and visualization pipeline
- Support for multiple datasets and data formats
- Automatic checkpoint management
- Organized output structure with figs/dataset_name directories
- Data persistence for efficient re-visualization

Author: Deep-IRT Pipeline System
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime


def run_command(cmd, desc="Running command", show_progress=False):
    """
    Execute subprocess command with comprehensive error handling.
    
    Parameters:
        cmd (list): Command and arguments to execute
        desc (str): Description of the command being run
        show_progress (bool): Whether to show real-time output
        
    Returns:
        tuple: (success_status, output) where success_status is boolean
    """
    print(f"{desc}...")
    
    if show_progress:
        # Don't capture output to show real-time progress
        result = subprocess.run(cmd)
        return result.returncode == 0, ""
    else:
        # Capture output for error handling
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error: {desc} failed")
            if result.stderr:
                print(result.stderr)
            return False, result.stdout
        
        return True, result.stdout


def find_checkpoint_files(data_style, dataset):
    """
    Locate checkpoint and configuration files for a given dataset.
    
    Parameters:
        data_style (str): Data format style ('yeung' or 'torch')
        dataset (str): Dataset name
        
    Returns:
        tuple: (checkpoint_path, config_path) with file paths or None if not found
    """
    checkpoint_dir = f"checkpoints_{data_style}"
    
    # Search for checkpoint files in order of preference
    checkpoints = [
        f"best_model_{dataset}.pth",
        f"final_model_{dataset}.pth"
    ]
    
    checkpoint_path = None
    for ckpt in checkpoints:
        path = os.path.join(checkpoint_dir, ckpt)
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    config_path = os.path.join(checkpoint_dir, f"config_{dataset}.json")
    
    return checkpoint_path, config_path


def train_model(args):
    """
    Execute model training with specified configuration.
    
    Parameters:
        args (Namespace): Parsed command-line arguments containing training parameters
        
    Returns:
        tuple: (success_status, checkpoint_path, config_path)
    """
    cmd = [
        sys.executable, "train.py",
        "--dataset", args.dataset,
        "--data_style", args.data_style,
        "--fold", str(args.fold)
    ]
    
    # Append optional training parameters if specified
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.learning_rate:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    
    success, output = run_command(cmd, f"Training {args.dataset} ({args.data_style})", show_progress=True)
    
    if success:
        checkpoint_path, config_path = find_checkpoint_files(args.data_style, args.dataset)
        if checkpoint_path and os.path.exists(config_path):
            print("Training completed successfully")
            return True, checkpoint_path, config_path
    
    print("Training failed or files not found")
    return False, None, None


def create_visualizations(checkpoint_path, config_path, output_dir, student_idx=0, save_data_path=None):
    """
    Generate comprehensive visualizations from trained model.
    
    Parameters:
        checkpoint_path (str): Path to model checkpoint
        config_path (str): Path to model configuration file
        output_dir (str): Directory for saving visualizations
        student_idx (int): Student index for per-KC visualizations
        save_data_path (str, optional): Path to save extracted data
        
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [
        sys.executable, "visualize.py",
        "--checkpoint", checkpoint_path,
        "--config", config_path,
        "--output_dir", output_dir,
        "--student_idx", str(student_idx)
    ]
    
    if save_data_path:
        cmd.extend(["--save_data", save_data_path])
    
    success, output = run_command(cmd, "Generating visualizations")
    
    if success:
        print(output)
    
    return success


def create_visualizations_from_data(data_path, output_dir, student_idx=0):
    """
    Generate visualizations from previously saved data without model inference.
    
    Parameters:
        data_path (str): Path to saved data pickle file
        output_dir (str): Directory for saving visualizations
        student_idx (int): Student index for per-KC visualizations
        
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [
        sys.executable, "visualize.py",
        "--load_data", data_path,
        "--output_dir", output_dir,
        "--student_idx", str(student_idx)
    ]
    
    success, output = run_command(cmd, "Generating visualizations from saved data")
    
    if success:
        print(output)
    
    return success


def get_default_epochs(dataset, data_style):
    """
    Determine appropriate default epoch count based on dataset and data style.
    
    Parameters:
        dataset (str): Dataset name
        data_style (str): Data format style
        
    Returns:
        int: Recommended number of training epochs
    """
    if dataset == "STATICS" and data_style == "yeung":
        return 20
    elif data_style == "torch":
        return 25
    else:
        return 20


def main():
    """
    Main pipeline function coordinating training and visualization workflows.
    
    Handles command-line argument parsing, validates inputs, and orchestrates
    the complete Deep-IRT pipeline including training, visualization, and
    data persistence.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description='Deep-IRT Training + Visualization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick STATICS training with per-KC mode
  python main.py --dataset STATICS --data_style yeung --epochs 5

  # Standard assist2015 training
  python main.py --dataset assist2015 --data_style torch --epochs 25

  # Visualization only from saved data
  python main.py --viz_only --load_data saved_data.pkl
        """
    )
    
    # Define required command-line arguments
    parser.add_argument('--dataset', type=str, 
                        choices=['assist2009_updated', 'assist2015', 'STATICS', 'assist2009', 'fsaif1tof3', 'synthetic'],
                        help='Dataset name')
    parser.add_argument('--data_style', type=str, 
                        choices=['yeung', 'torch'],
                        help='Data format style')
    
    # Define training configuration arguments
    parser.add_argument('--fold', type=int, default=0, help='Fold index (0-4)')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    
    # Define visualization configuration arguments
    parser.add_argument('--student_idx', type=int, default=0, help='Student index for per-KC plots')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--save_data', type=str, help='Save extracted data file')
    parser.add_argument('--load_data', type=str, help='Load saved data file')
    
    # Define pipeline control arguments
    parser.add_argument('--skip_training', action='store_true', help='Skip training')
    parser.add_argument('--skip_visualization', action='store_true', help='Skip visualization')
    parser.add_argument('--viz_only', action='store_true', help='Only run visualization from saved data')
    
    args = parser.parse_args()
    
    # Validate command-line arguments
    if args.fold < 0 or args.fold >= 5:
        print("Error: Fold index must be between 0 and 4")
        return 1
    
    # Handle visualization-only workflow
    if args.viz_only:
        if not args.load_data:
            print("Error: --viz_only requires --load_data")
            return 1
        
        output_dir = args.output_dir or os.path.join('figs', 'saved_data_viz')
        success = create_visualizations_from_data(args.load_data, output_dir, args.student_idx)
        return 0 if success else 1
    
    # Handle standard pipeline workflow
    if not args.dataset or not args.data_style:
        print("Error: --dataset and --data_style are required")
        return 1
    
    # Configure default parameters
    if not args.epochs:
        args.epochs = get_default_epochs(args.dataset, args.data_style)
    
    if not args.output_dir:
        args.output_dir = os.path.join('figs', args.dataset)
    
    if not args.save_data and not args.skip_visualization:
        args.save_data = f"saved_data_{args.dataset}_{args.data_style}.pkl"
    
    # Display pipeline configuration
    print(f"Deep-IRT Pipeline: {args.dataset} ({args.data_style})")
    print(f"Config: {args.epochs} epochs, fold {args.fold}")
    print(f"Output: {args.output_dir}")
    
    training_success = True
    checkpoint_path = None
    config_path = None
    
    # Execute training phase
    if not args.skip_training:
        training_success, checkpoint_path, config_path = train_model(args)
        if not training_success:
            print("Pipeline failed at training stage")
            return 1
    else:
        # Locate existing checkpoint files
        checkpoint_path, config_path = find_checkpoint_files(args.data_style, args.dataset)
        if not checkpoint_path or not os.path.exists(config_path):
            print(f"No existing checkpoint found for {args.dataset} ({args.data_style})")
            return 1
    
    # Execute visualization phase
    viz_success = True
    if not args.skip_visualization:
        os.makedirs(args.output_dir, exist_ok=True)
        viz_success = create_visualizations(checkpoint_path, config_path, args.output_dir, 
                                          args.student_idx, args.save_data)
    
    # Report final pipeline status
    if training_success and viz_success:
        print(f"\n{'='*50}")
        print("Pipeline completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*50}")
        return 0
    else:
        print(f"\n{'='*50}")
        print("Pipeline failed!")
        print(f"{'='*50}")
        return 1


if __name__ == "__main__":
    sys.exit(main())