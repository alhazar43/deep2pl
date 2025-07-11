#!/usr/bin/env python3
"""
Main Script for Deep-IRT: Training + Visualization
Combines training and automatic visualization generation in one pipeline
"""

import os
import sys
import argparse
import json
import subprocess
from datetime import datetime


def run_training(args):
    """Run training using train.py with specified arguments."""
    print(f"Training {args.dataset} ({args.data_style}) - {args.epochs} epochs")
    
    train_cmd = [
        sys.executable, "train.py",
        "--dataset", args.dataset,
        "--data_style", args.data_style,
        "--fold", str(args.fold)
    ]
    
    # Add optional parameters
    if args.epochs is not None:
        train_cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        train_cmd.extend(["--batch_size", str(args.batch_size)])
    if args.learning_rate is not None:
        train_cmd.extend(["--learning_rate", str(args.learning_rate)])
    
    # Run training
    result = subprocess.run(train_cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Training failed")
        return False, None, None
    
    # Find the generated checkpoint and config
    checkpoint_dir = f"checkpoints_{args.data_style}"
    final_checkpoint = os.path.join(checkpoint_dir, f"final_model_{args.dataset}.pth")
    best_checkpoint = os.path.join(checkpoint_dir, f"best_model_{args.dataset}.pth")
    config_file = os.path.join(checkpoint_dir, f"config_{args.dataset}.json")
    
    # Use best checkpoint if available, otherwise final
    if os.path.exists(best_checkpoint):
        checkpoint_path = best_checkpoint
    elif os.path.exists(final_checkpoint):
        checkpoint_path = final_checkpoint
    else:
        print(f"No checkpoint found in {checkpoint_dir}")
        return False, None, None
    
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        return False, None, None
    
    print(f"Training completed")
    return True, checkpoint_path, config_file


def run_visualization(checkpoint_path, config_file, output_dir, student_idx=0, save_data_path=None):
    """Run visualization using visualize.py."""
    print(f"Generating visualizations...")
    
    viz_cmd = [
        sys.executable, "visualize.py",
        "--checkpoint", checkpoint_path,
        "--config", config_file,
        "--output_dir", output_dir,
        "--student_idx", str(student_idx)
    ]
    
    # Add save data option if specified
    if save_data_path:
        viz_cmd.extend(["--save_data", save_data_path])
    
    # Run visualization
    result = subprocess.run(viz_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Visualization failed:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def run_visualization_from_saved_data(data_path, output_dir, student_idx=0):
    """Run visualization from saved data without model inference."""
    print(f"Generating visualizations from saved data...")
    
    viz_cmd = [
        sys.executable, "visualize.py",
        "--load_data", data_path,
        "--output_dir", output_dir,
        "--student_idx", str(student_idx)
    ]
    
    # Run visualization
    result = subprocess.run(viz_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Visualization from saved data failed:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def print_summary(args, checkpoint_path, output_dir, training_success, viz_success):
    """Print final summary of the pipeline."""
    print(f"\n{'='*50}")
    status = "SUCCESS" if training_success and viz_success else "FAILED"
    print(f"Pipeline {status}")
    print(f"{'='*50}")
    if training_success and viz_success:
        print(f"Results: {output_dir}/")
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
            print(f"Generated {len(files)} visualizations")
    else:
        if not training_success:
            print(f"Training failed")
        if not viz_success:
            print(f"Visualization failed")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description='Deep-IRT Training + Visualization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick STATICS training with per-KC mode
  python main.py --dataset STATICS --data_style yeung --epochs 5

  # Large-scale STATICS training
  python main.py --dataset STATICS --data_style yeung --epochs 50 --batch_size 32

  # Standard assist2015 training
  python main.py --dataset assist2015 --data_style torch --epochs 25

  # Custom configuration
  python main.py --dataset STATICS --data_style yeung --epochs 30 --learning_rate 0.0005 --student_idx 5
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['assist2009_updated', 'assist2015', 'STATICS', 'assist2009'],
                        help='Dataset name')
    parser.add_argument('--data_style', type=str, required=True,
                        choices=['yeung', 'torch'],
                        help='Data format style')
    
    # Training arguments
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index for cross-validation (0-4)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: dataset-specific)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: dataset-specific)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (default: 0.001)')
    
    # Visualization arguments
    parser.add_argument('--student_idx', type=int, default=0,
                        help='Student index for per-KC visualization (default: 0)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for visualizations (default: results_DATASET_TIMESTAMP)')
    parser.add_argument('--save_data', type=str, default=None,
                        help='Path to save extracted theta/beta data for reuse (default: saved_data_DATASET.pkl)')
    parser.add_argument('--load_data', type=str, default=None,
                        help='Path to load saved data for visualization without training/inference')
    
    # Pipeline control
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only run visualization')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip visualization and only run training')
    parser.add_argument('--viz_only', action='store_true',
                        help='Only run visualization from saved data (requires --load_data)')
    
    args = parser.parse_args()
    
    # Validation
    if args.fold < 0 or args.fold >= 5:
        print("Error: Fold index must be between 0 and 4")
        return
    
    # Check if this is visualization-only mode
    if args.viz_only:
        if not args.load_data:
            print("Error: --viz_only requires --load_data")
            return
        if args.output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output_dir = f"viz_results_{timestamp}"
        
        success = run_visualization_from_saved_data(args.load_data, args.output_dir, args.student_idx)
        print(f"Visualization from saved data: {'SUCCESS' if success else 'FAILED'}")
        return 0 if success else 1
    
    # Set default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"results_{args.dataset}_{timestamp}"
    
    # Set default save data path
    if args.save_data is None and not args.skip_visualization:
        args.save_data = f"saved_data_{args.dataset}_{args.data_style}.pkl"
    
    # Set default epochs if not specified
    if args.epochs is None:
        if args.dataset == "STATICS" and args.data_style == "yeung":
            args.epochs = 20  # Default for per-KC mode
        elif args.data_style == "torch":
            args.epochs = 25  # Default for global mode
        else:
            args.epochs = 20  # General default
    
    print(f"Deep-IRT Pipeline: {args.dataset} ({args.data_style})")
    config_str = f"{args.epochs}ep"
    if args.batch_size:
        config_str += f", bs{args.batch_size}"
    if args.learning_rate:
        config_str += f", lr{args.learning_rate}"
    print(f"Config: {config_str}, fold{args.fold} -> {args.output_dir}")
    
    training_success = True
    checkpoint_path = None
    config_file = None
    
    # Training phase
    if not args.skip_training:
        training_success, checkpoint_path, config_file = run_training(args)
        
        if not training_success:
            print("Training failed. Exiting pipeline.")
            return
    else:
        # Find existing checkpoint
        checkpoint_dir = f"checkpoints_{args.data_style}"
        best_checkpoint = os.path.join(checkpoint_dir, f"best_model_{args.dataset}.pth")
        final_checkpoint = os.path.join(checkpoint_dir, f"final_model_{args.dataset}.pth")
        config_file = os.path.join(checkpoint_dir, f"config_{args.dataset}.json")
        
        if os.path.exists(best_checkpoint):
            checkpoint_path = best_checkpoint
        elif os.path.exists(final_checkpoint):
            checkpoint_path = final_checkpoint
        else:
            print(f"No existing checkpoint found for {args.dataset} ({args.data_style})")
            return
        
        if not os.path.exists(config_file):
            print(f"Config file not found: {config_file}")
            return
    
    # Visualization phase
    viz_success = True
    if not args.skip_visualization:
        os.makedirs(args.output_dir, exist_ok=True)
        viz_success = run_visualization(checkpoint_path, config_file, args.output_dir, 
                                      args.student_idx, args.save_data)
    
    # Summary
    print_summary(args, checkpoint_path, args.output_dir, training_success, viz_success)
    
    # Success indicators for automation
    return 0 if (training_success and viz_success) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)