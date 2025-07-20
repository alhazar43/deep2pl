#!/usr/bin/env python3
"""
Comprehensive Deep-2PL Pipeline Script

This script handles training, testing, evaluation, and visualization for all datasets.
It performs 5-fold cross validation, evaluates all models, extracts comprehensive
statistics, and generates visualizations.

Features:
- Complete 5-fold cross validation training for all datasets
- Comprehensive model evaluation on test sets
- Statistical extraction (theta, alpha, beta parameters)
- Training metrics visualization
- Results organization in structured directories
"""

import os
import sys
import json
import argparse
import subprocess
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path


def run_command(cmd, desc="Running command", show_progress=True):
    """Execute subprocess command with error handling."""
    print(f"\n{desc}...")
    
    if show_progress:
        result = subprocess.run(cmd)
        return result.returncode == 0, ""
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {desc} failed")
            if result.stderr:
                print(result.stderr)
            return False, result.stdout
        return True, result.stdout


def get_dataset_configs():
    """Get all available dataset configurations."""
    return {
        'assist2009_updated': {'epochs': 20, 'batch_size': 32},
        'STATICS': {'epochs': 20, 'batch_size': 16},
        'assist2015': {'epochs': 20, 'batch_size': 32},
        'fsaif1tof3': {'epochs': 20, 'batch_size': 32},
        'synthetic': {'epochs': 20, 'batch_size': 32},
        'assist2009': {'epochs': 20, 'batch_size': 32},
        'assist2017': {'epochs': 20, 'batch_size': 32},
        'statics2011': {'epochs': 20, 'batch_size': 32},
        'kddcup2010': {'epochs': 20, 'batch_size': 32}
    }


def train_all_models(datasets=None, single_fold=False, fold_idx=0):
    """
    Train models for specified datasets.
    
    Args:
        datasets: List of dataset names (None for all)
        single_fold: If True, train only specified fold
        fold_idx: Fold index to train (0-4)
    
    Returns:
        dict: Training results summary
    """
    configs = get_dataset_configs()
    results = {}
    
    # Filter datasets
    if datasets is None:
        datasets = list(configs.keys())
    
    for dataset in datasets:
        if dataset not in configs:
            continue
            
        print(f"\n{'='*60}")
        print(f"Training {dataset}")
        print(f"{'='*60}")
        
        config = configs[dataset]
        
        # Build training command
        cmd = [
            sys.executable, "train.py",
            "--dataset", dataset,
            "--epochs", str(config['epochs']),
            "--batch_size", str(config['batch_size'])
        ]
        
        if single_fold:
            cmd.extend(["--fold", str(fold_idx), "--single_fold"])
        
        # Execute training
        success, output = run_command(
            cmd, 
            f"Training {dataset}" + 
            (f" fold {fold_idx}" if single_fold else " all folds"),
            show_progress=True
        )
        
        results[dataset] = {
            'dataset': dataset,
            'success': success,
            'single_fold': single_fold,
            'fold_idx': fold_idx if single_fold else None,
            'config': config
        }
        
        if success:
            print(f"✓ Training completed for {dataset}")
        else:
            print(f"✗ Training failed for {dataset}")
    
    return results


def evaluate_all_models(datasets=None, splits=['test']):
    """
    Evaluate all trained models.
    
    Args:
        datasets: List of dataset names (None for all)
        splits: List of splits to evaluate on
    
    Returns:
        dict: Evaluation results
    """
    save_models_dir = Path("save_models")
    if not save_models_dir.exists():
        print("No save_models directory found. Run training first.")
        return {}
    
    # Find all available models
    model_files = list(save_models_dir.glob("best_model_*.pth"))
    results = {}
    
    for model_file in model_files:
        # Parse model info from filename
        filename = model_file.stem
        if "fold" in filename:
            parts = filename.replace("best_model_", "").split("_fold")
            dataset = parts[0]
            fold = int(parts[1])
        else:
            dataset = filename.replace("best_model_", "")
            fold = None
        
        # Skip if not in requested datasets
        if datasets and dataset not in datasets:
            continue
        
        # Determine data style from config file
        config_pattern = f"config_{dataset}" + (f"_fold{fold}" if fold is not None else "") + ".json"
        config_file = save_models_dir / config_pattern
        
        if not config_file.exists():
            print(f"Config file not found for {model_file}")
            continue
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"\nEvaluating {dataset}" + 
              (f" fold {fold}" if fold is not None else ""))
        
        for split in splits:
            cmd = [
                sys.executable, "evaluate.py",
                "--model_path", str(model_file),
                "--split", split
            ]
            
            if fold is not None:
                cmd.extend(["--fold", str(fold)])
            
            success, output = run_command(
                cmd,
                f"Evaluating {dataset} on {split}",
                show_progress=False
            )
            
            key = f"{dataset}" + (f"_fold{fold}" if fold is not None else "")
            if key not in results:
                results[key] = {}
            
            results[key][split] = {
                'success': success,
                'model_file': str(model_file),
                'config_file': str(config_file)
            }
    
    return results


def extract_model_statistics(datasets=None):
    """
    Extract comprehensive statistics from trained models.
    
    Args:
        datasets: List of dataset names (None for all)
    
    Returns:
        dict: Extracted statistics
    """
    import torch
    from models.model import DeepIRTModel
    from data.dataloader import create_datasets, create_dataloader
    
    save_models_dir = Path("save_models")
    stats_dir = Path("stats")
    stats_dir.mkdir(exist_ok=True)
    
    model_files = list(save_models_dir.glob("best_model_*.pth"))
    all_stats = {}
    
    for model_file in model_files:
        # Parse model info
        filename = model_file.stem
        if "fold" in filename:
            parts = filename.replace("best_model_", "").split("_fold")
            dataset = parts[0]
            fold = int(parts[1])
        else:
            dataset = filename.replace("best_model_", "")
            fold = None
        
        if datasets and dataset not in datasets:
            continue
        
        # Load config
        config_pattern = f"config_{dataset}" + (f"_fold{fold}" if fold is not None else "") + ".json"
        config_file = save_models_dir / config_pattern
        
        if not config_file.exists():
            continue
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"\nExtracting statistics for {dataset}" +
              (f" fold {fold}" if fold is not None else ""))
        
        try:
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_file, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Get actual n_questions from checkpoint
            actual_n_questions = state_dict['q_embed.weight'].shape[0] - 1
            
            model = DeepIRTModel(
                n_questions=actual_n_questions,
                memory_size=config['memory_size'],
                key_dim=config.get('key_dim', 50),
                value_dim=config.get('value_dim', 200),
                summary_dim=config.get('summary_dim', 50),
                q_embed_dim=config['q_embed_dim'],
                qa_embed_dim=config['qa_embed_dim'],
                ability_scale=config['ability_scale'],
                use_discrimination=config['use_discrimination'],
                dropout_rate=config['dropout_rate'],
                q_matrix_path=config.get('q_matrix_path'),
                skill_mapping_path=config.get('skill_mapping_path')
            ).to(device)
            
            model.load_state_dict(state_dict)
            model.eval()
            
            # Create test dataset
            _, _, test_dataset = create_datasets(
                data_dir=config['data_dir'],
                dataset_name=config['dataset_name'],
                seq_len=config['seq_len'],
                n_questions=config['n_questions'],
                k_fold=config['k_fold'],
                fold_idx=fold or 0
            )
            
            test_loader = create_dataloader(test_dataset, batch_size=1, shuffle=False)
            
            # Extract statistics
            stats = {
                'dataset': dataset,
                'fold': fold,
                'model_info': {
                    'n_questions': actual_n_questions,
                    'n_parameters': sum(p.numel() for p in model.parameters()),
                    'per_kc_mode': model.per_kc_mode,
                    'use_discrimination': config['use_discrimination'],
                    'n_kcs': getattr(model, 'n_kcs', None)
                },
                'theta_stats': [],
                'beta_stats': [],
                'alpha_stats': [],
                'kc_theta_stats': [] if model.per_kc_mode else None
            }
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if batch_idx >= 50:  # Limit to 50 students for statistics
                        break
                    
                    q_data = batch['q_data'].to(device)
                    qa_data = batch['qa_data'].to(device)
                    
                    predictions, student_abilities, item_difficulties, z_values, kc_info = model(q_data, qa_data)
                    
                    # Extract theta (student abilities)
                    valid_mask = q_data > 0
                    if valid_mask.any():
                        valid_thetas = student_abilities[valid_mask].cpu().numpy()
                        stats['theta_stats'].extend(valid_thetas.tolist())
                        
                        # Extract beta (item difficulties)
                        valid_betas = item_difficulties[valid_mask].cpu().numpy()
                        stats['beta_stats'].extend(valid_betas.tolist())
                        
                        # Extract alpha (discrimination) if available
                        if hasattr(model, 'discrimination') and model.discrimination is not None:
                            alpha_val = model.discrimination.item()
                            stats['alpha_stats'].extend([alpha_val] * len(valid_thetas))
                        else:
                            stats['alpha_stats'].extend([1.0] * len(valid_thetas))
                    
                    # Extract per-KC statistics if available
                    if model.per_kc_mode and 'all_kc_thetas' in kc_info:
                        kc_thetas = kc_info['all_kc_thetas'][0].cpu().numpy()  # First student
                        seq_len, n_kcs = kc_thetas.shape
                        for t in range(seq_len):
                            if q_data[0, t].item() > 0:  # Valid timestep
                                stats['kc_theta_stats'].append(kc_thetas[t].tolist())
            
            # Calculate summary statistics
            if stats['theta_stats']:
                stats['theta_summary'] = {
                    'mean': float(np.mean(stats['theta_stats'])),
                    'std': float(np.std(stats['theta_stats'])),
                    'min': float(np.min(stats['theta_stats'])),
                    'max': float(np.max(stats['theta_stats']))
                }
            
            if stats['beta_stats']:
                stats['beta_summary'] = {
                    'mean': float(np.mean(stats['beta_stats'])),
                    'std': float(np.std(stats['beta_stats'])),
                    'min': float(np.min(stats['beta_stats'])),
                    'max': float(np.max(stats['beta_stats']))
                }
            
            if stats['alpha_stats']:
                unique_alphas = list(set(stats['alpha_stats']))
                stats['alpha_summary'] = {
                    'mean': float(np.mean(stats['alpha_stats'])),
                    'std': float(np.std(stats['alpha_stats'])),
                    'unique_values': unique_alphas,
                    'is_constant': len(unique_alphas) == 1
                }
            
            # Save statistics
            stats_filename = f"saved_stats_{dataset}" + (f"_fold{fold}" if fold is not None else "") + ".pkl"
            stats_path = stats_dir / stats_filename
            
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)
            
            print(f"Statistics saved to {stats_path}")
            
            key = f"{dataset}" + (f"_fold{fold}" if fold is not None else "")
            all_stats[key] = stats
            
        except Exception as e:
            print(f"Error extracting statistics for {dataset}: {e}")
            continue
    
    return all_stats


def generate_all_visualizations(datasets=None):
    """Generate training metrics and evaluation visualizations."""
    
    # Generate training metrics plots
    print("\nGenerating training metrics visualizations...")
    cmd = [sys.executable, "plot_metrics.py", "--all", "--results_dir", "results/train", "--output_dir", "results/plots"]
    run_command(cmd, "Generating training metrics plots", show_progress=False)
    
    # Generate evaluation visualizations for ROC curves
    print("\nGenerating evaluation visualizations...")
    eval_results_dir = Path("results/test")
    if eval_results_dir.exists():
        eval_files = list(eval_results_dir.glob("eval_*.json"))
        if eval_files:
            datasets_found = set()
            for eval_file in eval_files:
                for dataset in ['STATICS', 'assist2015', 'assist2009_updated', 'fsaif1tof3', 'synthetic', 'assist2009']:
                    if dataset in eval_file.name:
                        datasets_found.add(dataset)
            
            for dataset in datasets_found:
                dataset_files = [str(f) for f in eval_files if dataset in f.name]
                if dataset_files:
                    cmd = [
                        sys.executable, "plot_metrics.py",
                        "--eval_results_dir", str(eval_results_dir),
                        "--dataset", dataset,
                        "--output_dir", "results/plots"
                    ]
                    run_command(cmd, f"Generating ROC curves for {dataset}", show_progress=False)


def create_summary_report(training_results, evaluation_results, statistics):
    """Create comprehensive summary report."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'training_summary': {
            'total_models': len(training_results),
            'successful_trainings': sum(1 for r in training_results.values() if r['success']),
            'failed_trainings': sum(1 for r in training_results.values() if not r['success']),
            'details': training_results
        },
        'evaluation_summary': {
            'total_evaluations': len(evaluation_results),
            'details': evaluation_results
        },
        'statistics_summary': {
            'total_models_analyzed': len(statistics),
            'details': {}
        }
    }
    
    # Add statistics summary
    for key, stats in statistics.items():
        summary['statistics_summary']['details'][key] = {
            'dataset': stats['dataset'],
            'fold': stats['fold'],
            'model_info': stats['model_info'],
            'theta_summary': stats.get('theta_summary', {}),
            'beta_summary': stats.get('beta_summary', {}),
            'alpha_summary': stats.get('alpha_summary', {})
        }
    
    # Save summary report
    summary_path = results_dir / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary report saved to {summary_path}")
    
    # Print brief summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Training: {summary['training_summary']['successful_trainings']}/{summary['training_summary']['total_models']} successful")
    print(f"Evaluation: {len(evaluation_results)} models evaluated")
    print(f"Statistics: {len(statistics)} models analyzed")
    print(f"Results saved in: results/")
    print(f"{'='*60}")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Deep-2PL Pipeline: Training, Testing, and Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for all datasets
  python main.py --all

  # Run pipeline for specific datasets
  python main.py --datasets STATICS assist2015

  # Train and evaluate single fold
  python main.py --datasets STATICS --single_fold --fold 0

  # Skip training, only evaluate and visualize
  python main.py --skip_training --datasets STATICS
        """
    )
    
    # Dataset selection
    parser.add_argument('--all', action='store_true', help='Run pipeline for all available datasets')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['assist2009_updated', 'assist2015', 'STATICS', 'assist2009', 'fsaif1tof3', 'synthetic',
                               'assist2017', 'statics2011', 'kddcup2010'],
                       help='Specific datasets to process')
    
    # Training options
    parser.add_argument('--single_fold', action='store_true', help='Train only single fold instead of 5-fold CV')
    parser.add_argument('--fold', type=int, default=0, help='Fold index for single fold training (0-4)')
    
    # Pipeline control
    parser.add_argument('--skip_training', action='store_true', help='Skip training phase')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation phase')
    parser.add_argument('--skip_statistics', action='store_true', help='Skip statistics extraction')
    parser.add_argument('--skip_visualization', action='store_true', help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.fold < 0 or args.fold >= 5:
        print("Error: Fold index must be between 0 and 4")
        return 1
    
    # Determine datasets to process
    datasets = None
    
    if args.all:
        datasets = None  # Process all available
    else:
        datasets = args.datasets
    
    if not args.all and not datasets:
        print("Error: Must specify --all or --datasets")
        return 1
    
    print(f"{'='*60}")
    print("COMPREHENSIVE DEEP-2PL PIPELINE")
    print(f"{'='*60}")
    print(f"Datasets: {datasets or 'all available'}")
    print(f"Training mode: {'Single fold' if args.single_fold else '5-fold CV'}")
    if args.single_fold:
        print(f"Fold: {args.fold}")
    print(f"{'='*60}")
    
    # Initialize results
    training_results = {}
    evaluation_results = {}
    statistics = {}
    
    # Phase 1: Training
    if not args.skip_training:
        print("\n" + "="*20 + " PHASE 1: TRAINING " + "="*20)
        training_results = train_all_models(datasets, args.single_fold, args.fold)
    
    # Phase 2: Evaluation
    if not args.skip_evaluation:
        print("\n" + "="*20 + " PHASE 2: EVALUATION " + "="*19)
        evaluation_results = evaluate_all_models(datasets, ['test'])
    
    # Phase 3: Statistics Extraction
    if not args.skip_statistics:
        print("\n" + "="*18 + " PHASE 3: STATISTICS " + "="*18)
        statistics = extract_model_statistics(datasets)
    
    # Phase 4: Visualization
    if not args.skip_visualization:
        print("\n" + "="*18 + " PHASE 4: VISUALIZATION " + "="*16)
        generate_all_visualizations(datasets)
    
    # Phase 5: Summary Report
    print("\n" + "="*20 + " PHASE 5: SUMMARY " + "="*20)
    create_summary_report(training_results, evaluation_results, statistics)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())