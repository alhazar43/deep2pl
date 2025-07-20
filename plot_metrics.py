#!/usr/bin/env python3
"""
Training Metrics Plotting Utility for Deep-IRT Model

Supports:
- Plotting training/validation loss, accuracy, AUC over epochs
- Loading metrics from training results
- Saving plots to results directory
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime


def load_training_metrics(metrics_path):
    """
    Load training metrics from JSON file.
    
    Args:
        metrics_path: Path to metrics JSON file
        
    Returns:
        dict: Training metrics data
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def plot_training_curves(metrics, save_dir=None, dataset_name=None, fold_idx=None):
    """
    Plot training curves for loss, accuracy, and AUC.
    
    Args:
        metrics: Dictionary containing training metrics
        save_dir: Directory to save plots
        dataset_name: Name of dataset for titles
        fold_idx: Fold index for titles
    """
    train_losses = metrics['train_losses']
    valid_losses = [v for v in metrics['valid_losses'] if v is not None]
    valid_aucs = [v for v in metrics['valid_aucs'] if v is not None]
    valid_accs = [v for v in metrics['valid_accs'] if v is not None]
    
    # Create epochs arrays
    train_epochs = list(range(1, len(train_losses) + 1))
    valid_epochs = []
    eval_every = metrics['config'].get('eval_every', 5)
    
    for i in range(1, len(train_losses) + 1):
        if i % eval_every == 0:
            valid_epochs.append(i)
    
    # Ensure we have the right number of validation points
    valid_epochs = valid_epochs[:len(valid_losses)]
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Training Metrics - {dataset_name or "Dataset"}' + 
                 (f' (Fold {fold_idx})' if fold_idx is not None else ''), 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(train_epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    if valid_losses:
        ax1.plot(valid_epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation AUC
    ax2 = axes[0, 1]
    if valid_aucs:
        ax2.plot(valid_epochs, valid_aucs, 'g-', label='Validation AUC', linewidth=2, alpha=0.8)
        ax2.axhline(y=metrics['best_valid_auc'], color='g', linestyle='--', alpha=0.7, 
                   label=f'Best AUC: {metrics["best_valid_auc"]:.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Validation AUC', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Validation Accuracy
    ax3 = axes[1, 0]
    if valid_accs:
        ax3.plot(valid_epochs, valid_accs, 'm-', label='Validation Accuracy', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Validation Accuracy', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Combined Loss (zoomed)
    ax4 = axes[1, 1]
    ax4.plot(train_epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    if valid_losses:
        ax4.plot(valid_epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    
    # Zoom to show convergence better
    if len(train_losses) > 20:
        start_epoch = len(train_losses) // 4  # Start from 25% of training
        ax4.set_xlim(start_epoch, len(train_losses))
        min_loss = min(train_losses[start_epoch:])
        max_loss = max(train_losses[start_epoch:])
        if valid_losses:
            min_loss = min(min_loss, min(valid_losses))
            max_loss = max(max_loss, max(valid_losses))
        loss_range = max_loss - min_loss
        ax4.set_ylim(min_loss - 0.1 * loss_range, max_loss + 0.1 * loss_range)
        ax4.set_title('Loss (Convergence View)', fontweight='bold')
    else:
        ax4.set_title('Loss (Full View)', fontweight='bold')
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
        plot_path = os.path.join(save_dir, f"training_curves_{dataset_name}{fold_suffix}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {plot_path}")
    
    plt.close()
    
    return fig


def plot_cross_validation_summary(cv_summary_path, save_dir=None):
    """
    Plot cross-validation summary results.
    
    Args:
        cv_summary_path: Path to cross-validation summary JSON
        save_dir: Directory to save plots
    """
    with open(cv_summary_path, 'r') as f:
        cv_summary = cv_summary_data = json.load(f)
    
    fold_results = cv_summary['fold_results']
    folds = [r['fold'] for r in fold_results]
    valid_aucs = [r['best_valid_auc'] for r in fold_results]
    test_aucs = [r['test_auc'] for r in fold_results]
    test_accs = [r['test_acc'] for r in fold_results]
    
    # Set up plotting
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'5-Fold Cross Validation Results - {cv_summary["dataset"]}', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Validation AUC per fold
    ax1 = axes[0]
    bars1 = ax1.bar(folds, valid_aucs, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.axhline(y=cv_summary['mean_valid_auc'], color='red', linestyle='--', linewidth=2,
               label=f"Mean: {cv_summary['mean_valid_auc']:.4f}")
    ax1.fill_between([-0.5, 4.5], 
                     cv_summary['mean_valid_auc'] - cv_summary['std_valid_auc'],
                     cv_summary['mean_valid_auc'] + cv_summary['std_valid_auc'],
                     alpha=0.2, color='red', label=f"±1 STD: {cv_summary['std_valid_auc']:.4f}")
    
    # Add value labels on bars
    for bar, val in zip(bars1, valid_aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Validation AUC')
    ax1.set_title('Best Validation AUC per Fold', fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test AUC per fold
    ax2 = axes[1]
    bars2 = ax2.bar(folds, test_aucs, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    ax2.axhline(y=cv_summary['mean_test_auc'], color='red', linestyle='--', linewidth=2,
               label=f"Mean: {cv_summary['mean_test_auc']:.4f}")
    ax2.fill_between([-0.5, 4.5], 
                     cv_summary['mean_test_auc'] - cv_summary['std_test_auc'],
                     cv_summary['mean_test_auc'] + cv_summary['std_test_auc'],
                     alpha=0.2, color='red', label=f"±1 STD: {cv_summary['std_test_auc']:.4f}")
    
    # Add value labels on bars
    for bar, val in zip(bars2, test_aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Test AUC')
    ax2.set_title('Test AUC per Fold', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test Accuracy per fold
    ax3 = axes[2]
    bars3 = ax3.bar(folds, test_accs, alpha=0.7, color='orange', edgecolor='darkorange')
    ax3.axhline(y=cv_summary['mean_test_acc'], color='red', linestyle='--', linewidth=2,
               label=f"Mean: {cv_summary['mean_test_acc']:.4f}")
    ax3.fill_between([-0.5, 4.5], 
                     cv_summary['mean_test_acc'] - cv_summary['std_test_acc'],
                     cv_summary['mean_test_acc'] + cv_summary['std_test_acc'],
                     alpha=0.2, color='red', label=f"±1 STD: {cv_summary['std_test_acc']:.4f}")
    
    # Add value labels on bars
    for bar, val in zip(bars3, test_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Test Accuracy per Fold', fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"cv_summary_{cv_summary['dataset']}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Cross-validation summary saved to: {plot_path}")
    
    plt.close()
    
    return fig


def plot_roc_curves(eval_results_paths, save_dir=None, dataset_name=None):
    """
    Plot ROC curves from evaluation results.
    
    Args:
        eval_results_paths: List of paths to evaluation result JSON files
        save_dir: Directory to save plots
        dataset_name: Dataset name for title
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(eval_results_paths)))
    
    for i, eval_path in enumerate(eval_results_paths):
        with open(eval_path, 'r') as f:
            eval_results = json.load(f)
        
        if 'roc_curve' in eval_results:
            fpr = eval_results['roc_curve']['fpr']
            tpr = eval_results['roc_curve']['tpr']
            auc = eval_results['auc']
            
            # Extract model info from filename
            filename = os.path.basename(eval_path)
            if 'fold' in filename:
                fold_match = filename.split('fold')[1].split('_')[0]
                label = f"Fold {fold_match} (AUC = {auc:.4f})"
            else:
                label = f"Model (AUC = {auc:.4f})"
            
            ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=label)
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves - {dataset_name or "Dataset"}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"roc_curves_{dataset_name or 'dataset'}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {plot_path}")
    
    plt.close()
    
    return fig


def main():
    """Main function for plotting metrics."""
    parser = argparse.ArgumentParser(description='Plot Training Metrics for Deep-IRT Model')
    parser.add_argument('--metrics_path', type=str,
                        help='Path to training metrics JSON file')
    parser.add_argument('--cv_summary_path', type=str,
                        help='Path to cross-validation summary JSON file')
    parser.add_argument('--eval_results_dir', type=str,
                        help='Directory containing evaluation result JSON files')
    parser.add_argument('--results_dir', type=str, default='results/train',
                        help='Results directory containing metrics files')
    parser.add_argument('--output_dir', type=str, default='results/plots',
                        help='Output directory for plots')
    parser.add_argument('--dataset', type=str,
                        help='Dataset name to plot (will search for files)')
    parser.add_argument('--all', action='store_true',
                        help='Plot metrics for all datasets found in results_dir')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Single metrics file
    if args.metrics_path:
        print(f"Plotting metrics from {args.metrics_path}")
        metrics = load_training_metrics(args.metrics_path)
        dataset_name = metrics['config'].get('dataset_name', 'unknown')
        fold_idx = None
        if 'fold' in os.path.basename(args.metrics_path):
            try:
                fold_idx = int(os.path.basename(args.metrics_path).split('fold')[1].split('.')[0])
            except:
                pass
        plot_training_curves(metrics, args.output_dir, dataset_name, fold_idx)
    
    # Cross-validation summary
    if args.cv_summary_path:
        print(f"Plotting cross-validation summary from {args.cv_summary_path}")
        plot_cross_validation_summary(args.cv_summary_path, args.output_dir)
    
    # Single dataset
    if args.dataset:
        print(f"Plotting metrics for dataset: {args.dataset}")
        
        # Look for individual fold metrics
        for fold in range(5):
            metrics_path = os.path.join(args.results_dir, f"metrics_{args.dataset}_fold{fold}.json")
            if os.path.exists(metrics_path):
                metrics = load_training_metrics(metrics_path)
                plot_training_curves(metrics, args.output_dir, args.dataset, fold)
        
        # Look for cross-validation summary
        cv_path = os.path.join(args.results_dir, f"cv_summary_{args.dataset}.json")
        if os.path.exists(cv_path):
            plot_cross_validation_summary(cv_path, args.output_dir)
        
        # Look for evaluation results
        if args.eval_results_dir:
            eval_files = []
            for filename in os.listdir(args.eval_results_dir):
                if filename.startswith(f"eval_") and args.dataset in filename and filename.endswith('.json'):
                    eval_files.append(os.path.join(args.eval_results_dir, filename))
            
            if eval_files:
                plot_roc_curves(eval_files, args.output_dir, args.dataset)
    
    # All datasets
    if args.all:
        print(f"Plotting metrics for all datasets in {args.results_dir}")
        
        # Find all datasets
        datasets = set()
        for filename in os.listdir(args.results_dir):
            if filename.startswith('metrics_') and filename.endswith('.json'):
                dataset = filename.replace('metrics_', '').replace('.json', '')
                if '_fold' in dataset:
                    dataset = dataset.split('_fold')[0]
                datasets.add(dataset)
            elif filename.startswith('cv_summary_') and filename.endswith('.json'):
                dataset = filename.replace('cv_summary_', '').replace('.json', '')
                datasets.add(dataset)
        
        for dataset in datasets:
            print(f"Processing dataset: {dataset}")
            
            # Plot individual fold metrics
            for fold in range(5):
                metrics_path = os.path.join(args.results_dir, f"metrics_{dataset}_fold{fold}.json")
                if os.path.exists(metrics_path):
                    metrics = load_training_metrics(metrics_path)
                    plot_training_curves(metrics, args.output_dir, dataset, fold)
            
            # Plot cross-validation summary
            cv_path = os.path.join(args.results_dir, f"cv_summary_{dataset}.json")
            if os.path.exists(cv_path):
                plot_cross_validation_summary(cv_path, args.output_dir)
    
    print(f"All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()