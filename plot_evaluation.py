#!/usr/bin/env python3
"""
Evaluation Visualization Script for Deep-2PL Model

This script generates ROC curves, Precision-Recall curves, and other 
evaluation visualizations from evaluation results.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_evaluation_results(results_dir, dataset=None):
    """Load evaluation results from JSON files."""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find evaluation files
    eval_files = list(results_dir.glob("eval_*.json"))
    
    if dataset:
        eval_files = [f for f in eval_files if dataset in f.name]
    
    if not eval_files:
        raise FileNotFoundError(f"No evaluation files found in {results_dir}")
    
    results = {}
    for eval_file in eval_files:
        with open(eval_file, 'r') as f:
            data = json.load(f)
            
        # Extract dataset and model info from filename
        filename = eval_file.stem
        parts = filename.replace("eval_", "").split("_")
        
        model_name = "_".join(parts[:-1])  # Everything except last part (split)
        split = parts[-1]
        
        if model_name not in results:
            results[model_name] = {}
        
        results[model_name][split] = data
    
    return results

def plot_roc_curves(results, output_dir, dataset_name=None):
    """Plot ROC curves for all models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (model_name, model_results) in enumerate(results.items()):
        if 'test' in model_results and 'roc_curve' in model_results['test']:
            roc_data = model_results['test']['roc_curve']
            fpr = np.array(roc_data['fpr'])
            tpr = np.array(roc_data['tpr'])
            auc = model_results['test']['auc']
            
            # Extract dataset from model name for label
            if 'fold' in model_name:
                dataset = model_name.split('_fold')[0].replace('best_model_', '').replace('final_model_', '')
                fold = model_name.split('_fold')[1]
                label = f"{dataset} (fold {fold}, AUC={auc:.3f})"
            else:
                dataset = model_name.replace('best_model_', '').replace('final_model_', '')
                label = f"{dataset} (AUC={auc:.3f})"
            
            ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=label)
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Save plot
    output_file = output_dir / f"roc_curves_{dataset_name if dataset_name else 'all'}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to: {output_file}")
    return output_file

def plot_precision_recall_curves(results, output_dir, dataset_name=None):
    """Plot Precision-Recall curves for all models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (model_name, model_results) in enumerate(results.items()):
        if 'test' in model_results and 'precision_recall_curve' in model_results['test']:
            pr_data = model_results['test']['precision_recall_curve']
            precision = np.array(pr_data['precision'])
            recall = np.array(pr_data['recall'])
            
            # Calculate average precision
            avg_precision = np.trapz(precision, recall)
            
            # Extract dataset from model name for label
            if 'fold' in model_name:
                dataset = model_name.split('_fold')[0].replace('best_model_', '').replace('final_model_', '')
                fold = model_name.split('_fold')[1]
                label = f"{dataset} (fold {fold}, AP={avg_precision:.3f})"
            else:
                dataset = model_name.replace('best_model_', '').replace('final_model_', '')
                label = f"{dataset} (AP={avg_precision:.3f})"
            
            ax.plot(recall, precision, color=colors[i], linewidth=2, label=label)
    
    # Calculate baseline (random classifier performance)
    baseline = results[list(results.keys())[0]]['test']['n_samples']
    if 'metadata' in results[list(results.keys())[0]]['test']:
        # Try to calculate positive rate from results
        accuracy = results[list(results.keys())[0]]['test']['accuracy']
        baseline_precision = accuracy  # Rough approximation
    else:
        baseline_precision = 0.5  # Default assumption
    
    ax.axhline(y=baseline_precision, color='k', linestyle='--', alpha=0.5, 
               label=f'Baseline (P={baseline_precision:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Save plot
    output_file = output_dir / f"precision_recall_curves_{dataset_name if dataset_name else 'all'}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Precision-Recall curves saved to: {output_file}")
    return output_file

def plot_performance_summary(results, output_dir, dataset_name=None):
    """Plot performance summary with AUC and Accuracy comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract performance metrics
    model_names = []
    aucs = []
    accuracies = []
    
    for model_name, model_results in results.items():
        if 'test' in model_results:
            # Clean model name for display
            clean_name = model_name.replace('best_model_', '').replace('final_model_', '')
            if 'fold' in clean_name:
                dataset_part, fold_part = clean_name.split('_fold')
                clean_name = f"{dataset_part}\n(fold {fold_part})"
            
            model_names.append(clean_name)
            aucs.append(model_results['test']['auc'])
            accuracies.append(model_results['test']['accuracy'])
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # AUC comparison
    bars1 = ax1.bar(range(len(model_names)), aucs, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_ylabel('AUC Score', fontsize=12)
    ax1.set_title('Model Performance - AUC Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, auc in zip(bars1, aucs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy comparison
    bars2 = ax2.bar(range(len(model_names)), accuracies, color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.set_xlabel('Models', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Performance - Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, acc in zip(bars2, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Save plot
    output_file = output_dir / f"performance_summary_{dataset_name if dataset_name else 'all'}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance summary saved to: {output_file}")
    return output_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate evaluation visualizations')
    parser.add_argument('--results_dir', type=str, default='results/test',
                        help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, default='results/plots',
                        help='Output directory for plots')
    parser.add_argument('--dataset', type=str,
                        help='Filter results for specific dataset')
    
    args = parser.parse_args()
    
    print(f"Loading evaluation results from: {args.results_dir}")
    
    try:
        results = load_evaluation_results(args.results_dir, args.dataset)
        print(f"Found results for {len(results)} models")
        
        if not results:
            print("No evaluation results found!")
            return
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        # ROC curves
        plot_roc_curves(results, args.output_dir, args.dataset)
        
        # Precision-Recall curves
        plot_precision_recall_curves(results, args.output_dir, args.dataset)
        
        # Performance summary
        plot_performance_summary(results, args.output_dir, args.dataset)
        
        print(f"\nAll visualizations saved to: {args.output_dir}")
        
        # Print summary
        print("\nPerformance Summary:")
        print("-" * 60)
        for model_name, model_results in results.items():
            if 'test' in model_results:
                clean_name = model_name.replace('best_model_', '').replace('final_model_', '')
                auc = model_results['test']['auc']
                acc = model_results['test']['accuracy']
                loss = model_results['test']['loss']
                n_samples = model_results['test']['n_samples']
                
                print(f"{clean_name:<25} | AUC: {auc:.4f} | Acc: {acc:.4f} | Loss: {loss:.4f} | Samples: {n_samples}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())