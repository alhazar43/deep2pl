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
from tqdm import tqdm
import time
import logging


def load_irt_stats(dataset_name, fold_idx=None, stats_type='fast'):
    """
    Load IRT statistics saved during training.
    
    Args:
        dataset_name: Name of dataset
        fold_idx: Fold index (None for all folds)  
        stats_type: 'fast' (weight-based) or 'full' (data-based)
        
    Returns:
        dict: IRT parameters (alpha, beta, theta)
    """
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    irt_dir = "irt_stats"
    
    if stats_type == 'fast':
        irt_path = os.path.join(irt_dir, f"fast_irt_{dataset_name}{fold_suffix}.pkl")
    else:
        irt_path = os.path.join(irt_dir, f"full_irt_{dataset_name}{fold_suffix}.pkl")
    
    try:
        with open(irt_path, 'rb') as f:
            irt_stats = pickle.load(f)
        
        print(f"\n=== IRT Statistics: {dataset_name} ({stats_type}) ===")
        if 'alpha_estimates' in irt_stats:
            alpha = irt_stats['alpha_estimates'] 
            print(f"Alpha (discrimination): mean={np.mean(alpha):.3f}±{np.std(alpha):.3f} [{len(alpha)} questions]")
        if 'beta_estimates' in irt_stats:
            beta = irt_stats['beta_estimates']
            print(f"Beta (difficulty): mean={np.mean(beta):.3f}±{np.std(beta):.3f} [{len(beta)} questions]")
        if 'theta_estimates' in irt_stats:
            theta = irt_stats['theta_estimates'] 
            print(f"Theta (student abilities): mean={np.mean(theta):.3f}±{np.std(theta):.3f} [{len(theta)} students]")
        elif 'student_abilities' in irt_stats:
            theta = irt_stats['student_abilities']
            print(f"Theta (student abilities): mean={np.mean(theta):.3f}±{np.std(theta):.3f} [{theta.shape}]")
        
        # Show extraction method and coverage info
        if 'extraction_method' in irt_stats:
            method_desc = {
                'dynamic_data_based': 'Dynamic data-based (fully adaptive)',
                'proper_data_based': 'Data-based extraction',
                'placeholder_no_data': 'Placeholder (no data provided)'
            }.get(irt_stats['extraction_method'], irt_stats['extraction_method'])
            print(f"Extraction method: {method_desc}")
        
        # Enhanced coverage information
        if 'questions_observed' in irt_stats and 'students_observed' in irt_stats:
            print(f"Data coverage: {irt_stats['questions_observed']} questions, {irt_stats['students_observed']} students")
            
            # Show question ID range if available (dynamic extraction)
            if 'question_id_range' in irt_stats:
                q_min, q_max = irt_stats['question_id_range']
                print(f"Question ID range: {q_min}-{q_max}")
            
            # Show extraction quality if available
            if 'extraction_quality' in irt_stats:
                quality = irt_stats['extraction_quality']
                coverage_pct = quality['coverage_ratio'] * 100
                print(f"Quality: {coverage_pct:.0f}% coverage ({quality['questions_with_data']}/{quality['questions_total']} questions)")
                
                if 'dataset_characteristics' in irt_stats:
                    chars = irt_stats['dataset_characteristics']
                    if 'total_interactions' in chars:
                        print(f"Total interactions: {chars['total_interactions']}")
        
        if 'per_kc_mode' in irt_stats and irt_stats['per_kc_mode']:
            print(f"Per-KC mode: {irt_stats['n_kcs']} knowledge components")
        
        return irt_stats
        
    except FileNotFoundError:
        print(f"❌ IRT statistics not found: {irt_path}")
        print(f"   Train the model first: python train.py --dataset {dataset_name}")
        return None
    except Exception as e:
        print(f"❌ Error loading IRT statistics: {e}")
        return None


def setup_logging():
    """Setup comprehensive logging for the pipeline."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Main pipeline log
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("Deep2PL_Pipeline")
    logger.info(f"Pipeline logging initialized: {log_file}")
    return logger


def run_command_with_progress(cmd, desc="Running command", dataset=None, expected_duration=None):
    """Execute subprocess command with real-time progress monitoring and logging."""
    logger = logging.getLogger("Deep2PL_Pipeline")
    
    logger.info(f"Starting: {desc}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    print(f"\n🚀 {desc}")
    
    # Start subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Setup progress tracking
    if expected_duration:
        progress_bar = tqdm(
            total=expected_duration,
            desc=f"{dataset or 'Task'}" if dataset else desc,
            unit="s",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        start_time = time.time()
        last_update = start_time
        
        # Monitor progress
        output_lines = []
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                logger.info(f"[{dataset or 'STDOUT'}] {line.strip()}")
                
                # Update progress bar based on time elapsed
                current_time = time.time()
                if current_time - last_update >= 1.0:  # Update every second
                    elapsed = int(current_time - start_time)
                    if elapsed <= expected_duration:
                        progress_bar.n = elapsed
                        progress_bar.refresh()
                    last_update = current_time
            else:
                time.sleep(0.1)
        
        # Final progress update
        progress_bar.n = min(int(time.time() - start_time), expected_duration)
        progress_bar.refresh()
        progress_bar.close()
        
    else:
        # Simple spinner for unknown duration
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        spinner_idx = 0
        
        output_lines = []
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                logger.info(f"[{dataset or 'STDOUT'}] {line.strip()}")
                
                # Update spinner
                print(f"\r{spinner_chars[spinner_idx % len(spinner_chars)]} {desc}", end='', flush=True)
                spinner_idx += 1
            else:
                time.sleep(0.1)
        
        print("\r✅", end='')
    
    # Get remaining output
    remaining_output, _ = process.communicate()
    if remaining_output:
        for line in remaining_output.strip().split('\n'):
            if line:
                output_lines.append(line)
                logger.info(f"[{dataset or 'STDOUT'}] {line}")
    
    success = process.returncode == 0
    
    if success:
        print(f" ✅ {desc} completed successfully")
        logger.info(f"Completed successfully: {desc}")
    else:
        print(f" ❌ {desc} failed (exit code: {process.returncode})")
        logger.error(f"Failed: {desc} (exit code: {process.returncode})")
    
    return success, '\n'.join(output_lines)


def run_command(cmd, desc="Running command", show_progress=True):
    """Backward compatibility wrapper."""
    return run_command_with_progress(cmd, desc)


def get_available_datasets(data_dir="./data"):
    """
    Detect which datasets are actually available in the data directory.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        list: Names of available datasets
    """
    available = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return available
    
    # Check for dataset directories
    for dataset_dir in data_path.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            
            # Look for training data files (multiple possible formats)
            train_files = [
                dataset_dir / f"{dataset_name}_train.txt",
                dataset_dir / f"{dataset_name}.train",
                dataset_dir / "train.txt"
            ]
            
            # Check if any training file exists
            if any(f.exists() for f in train_files):
                available.append(dataset_name)
                
    return sorted(available)


def get_dataset_categories():
    """Get dataset categories for filtering (like run_all_datasets.sh)."""
    return {
        'small': ['synthetic', 'assist2009_updated', 'assist2015', 'assist2017', 'assist2009'],
        'medium': ['kddcup2010', 'STATICS'], 
        'large': ['statics2011', 'fsaif1tof3']
    }

def estimate_training_time(datasets, epochs=20, single_fold=False):
    """Estimate training time for datasets (based on shell script estimates)."""
    # Time estimates in hours (for 30 epochs, 5-fold CV)
    time_estimates = {
        'synthetic': 0.075,
        'assist2009_updated': 0.1,
        'assist2015': 0.14,
        'assist2017': 0.15,
        'assist2009': 0.19,
        'kddcup2010': 0.98,
        'STATICS': 1.43,
        'statics2011': 1.8,
        'fsaif1tof3': 2.55
    }
    
    total_hours = 0
    print("⏱️  Training time estimates:")
    
    for dataset in datasets:
        if dataset in time_estimates:
            dataset_time = time_estimates[dataset]
            # Scale by epochs
            dataset_time = dataset_time * epochs / 30
            # Scale for single fold
            if single_fold:
                dataset_time = dataset_time / 5
            
            total_hours += dataset_time
            print(f"   {dataset}: {dataset_time:.1f}h")
    
    print(f"   Total estimated: {total_hours:.1f}h")
    
    if total_hours > 8:
        print("⚠️  Long training time. Consider --exclude large or --only small")
    
    return total_hours

def filter_datasets_by_category(available_datasets, exclude=None, only=None):
    """Filter datasets by category."""
    categories = get_dataset_categories()
    
    if only:
        if only in categories:
            return [d for d in available_datasets if d in categories[only]]
        else:
            return available_datasets
    
    if exclude:
        if exclude in categories:
            exclude_list = categories[exclude]
            return [d for d in available_datasets if d not in exclude_list]
        else:
            return available_datasets
    
    return available_datasets

def generate_comprehensive_comparison():
    """Generate comprehensive results comparison (from run_all_datasets.sh)."""
    import glob
    
    print("📊 Generating comprehensive results comparison...")
    
    # Find all test evaluation files
    test_files = glob.glob("results/test/eval_*_test.json")
    
    if not test_files:
        print("❌ No evaluation results found")
        return False
    
    results = []
    for file_path in test_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract dataset name from filename
            filename = os.path.basename(file_path)
            parts = filename.replace('.json', '').split('_')
            dataset = '_'.join(parts[2:-2])  # Remove eval_ prefix and test suffix
            
            results.append({
                'dataset': dataset,
                'auc': data.get('auc', 0),
                'accuracy': data.get('accuracy', 0),
                'loss': data.get('loss', float('inf')),
                'precision': data.get('precision', 0),
                'recall': data.get('recall', 0),
                'f1': data.get('f1', 0)
            })
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
    
    if not results:
        print("❌ No valid results to compare")
        return False
    
    # Display comparison
    print(f"\n{'='*80}")
    print("📋 COMPREHENSIVE RESULTS COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Dataset':<18} {'AUC':<8} {'Accuracy':<10} {'F1':<8} {'Loss':<8}")
    print("-" * 60)
    
    results.sort(key=lambda x: x['auc'], reverse=True)
    for result in results:
        print(f"{result['dataset']:<18} {result['auc']:.4f}   {result['accuracy']:.4f}    {result['f1']:.4f}   {result['loss']:.4f}")
    
    # Save to CSV
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv('results/comprehensive_comparison.csv', index=False)
        print(f"\n💾 Detailed results saved to: results/comprehensive_comparison.csv")
    except ImportError:
        print(f"\n💾 Install pandas to save CSV results")
    
    return True

def get_dataset_configs():
    """Get dataset configurations for available datasets only."""
    # Default configurations for all possible datasets
    all_configs = {
        # Small datasets (fast training)
        'synthetic': {'epochs': 20, 'batch_size': 32, 'est_time_single': 180, 'est_time_cv': 900},
        'assist2015': {'epochs': 20, 'batch_size': 32, 'est_time_single': 240, 'est_time_cv': 1200},
        'assist2009': {'epochs': 20, 'batch_size': 32, 'est_time_single': 300, 'est_time_cv': 1500},
        'assist2017': {'epochs': 20, 'batch_size': 32, 'est_time_single': 240, 'est_time_cv': 1200},
        
        # Medium datasets
        'assist2009_updated': {'epochs': 20, 'batch_size': 32, 'est_time_single': 600, 'est_time_cv': 3000},
        'kddcup2010': {'epochs': 20, 'batch_size': 32, 'est_time_single': 3600, 'est_time_cv': 18000},
        'STATICS': {'epochs': 20, 'batch_size': 16, 'est_time_single': 3600, 'est_time_cv': 18000},
        
        # Large datasets (slow training)
        'statics2011': {'epochs': 20, 'batch_size': 32, 'est_time_single': 6000, 'est_time_cv': 30000},
        'fsaif1tof3': {'epochs': 20, 'batch_size': 32, 'est_time_single': 7200, 'est_time_cv': 36000}
    }
    
    # Get actually available datasets
    available_datasets = get_available_datasets()
    
    # Return configs only for available datasets, with defaults for unknown ones
    configs = {}
    for dataset in available_datasets:
        if dataset in all_configs:
            configs[dataset] = all_configs[dataset]
        else:
            # Default config for unknown datasets
            configs[dataset] = {'epochs': 20, 'batch_size': 32, 'est_time_single': 600, 'est_time_cv': 3000}
    
    return configs


def train_all_models(datasets=None, single_fold=False, fold_idx=0, epochs=None, batch_size=None, learning_rate=None, model_type='optimized'):
    """
    Train models for specified datasets with progress tracking.
    
    Args:
        datasets: List of dataset names (None for all)
        single_fold: If True, train only specified fold
        fold_idx: Fold index to train (0-4)
        epochs: Number of epochs (overrides defaults)
        batch_size: Batch size (overrides defaults) 
        learning_rate: Learning rate (overrides default)
        model_type: Model architecture type
    
    Returns:
        dict: Training results summary
    """
    logger = logging.getLogger("Deep2PL_Pipeline")
    configs = get_dataset_configs()
    results = {}
    
    # Filter datasets - only use available ones
    if datasets is None:
        datasets = list(configs.keys())  # All available datasets
        logger.info(f"Auto-detected {len(datasets)} available datasets: {datasets}")
    else:
        # Validate requested datasets exist
        available = list(configs.keys())
        missing = [d for d in datasets if d not in available]
        if missing:
            logger.warning(f"Requested datasets not found in data/: {missing}")
            logger.info(f"Available datasets: {available}")
            datasets = [d for d in datasets if d in available]
        
    if not datasets:
        logger.error("No datasets available for training!")
        return {}
        
    logger.info(f"Training pipeline: {len(datasets)} datasets, {'single fold' if single_fold else '5-fold CV'}")
    
    # Overall progress tracking
    total_datasets = len([d for d in datasets if d in configs])
    dataset_progress = tqdm(total=total_datasets, desc="📊 Datasets", unit="dataset", position=0)
    
    for dataset in datasets:
        if dataset not in configs:
            logger.warning(f"Unknown dataset: {dataset}")
            continue
            
        print(f"\n{'='*60}")
        print(f"🎯 Training {dataset}")
        print(f"{'='*60}")
        
        config = configs[dataset]
        
        # Build training command with parameter overrides
        cmd = [
            sys.executable, "train.py",
            "--dataset", dataset,
            "--epochs", str(epochs if epochs is not None else config['epochs']),
            "--batch_size", str(batch_size if batch_size is not None else config['batch_size']),
            "--model_type", model_type
        ]
        
        if learning_rate is not None:
            cmd.extend(["--learning_rate", str(learning_rate)])
            
        if single_fold:
            cmd.extend(["--fold", str(fold_idx), "--single_fold"])
        
        # Get expected training time (adjust for custom epochs)
        base_epochs = config['epochs']
        actual_epochs = epochs if epochs is not None else base_epochs
        time_multiplier = actual_epochs / base_epochs if base_epochs > 0 else 1.0
        
        expected_time = int(config.get('est_time_single' if single_fold else 'est_time_cv', 600) * time_multiplier)
        
        # Execute training with progress bar
        success, output = run_command_with_progress(
            cmd, 
            f"Training {dataset}" + (f" fold {fold_idx}" if single_fold else " (5-fold CV)"),
            dataset=dataset,
            expected_duration=expected_time
        )
        
        results[dataset] = {
            'dataset': dataset,
            'success': success,
            'single_fold': single_fold,
            'fold_idx': fold_idx if single_fold else None,
            'config': config
        }
        
        if success:
            print(f"✅ Training completed for {dataset}")
            logger.info(f"Training completed successfully: {dataset}")
        else:
            print(f"❌ Training failed for {dataset}")
            logger.error(f"Training failed: {dataset}")
        
        # Update dataset progress
        dataset_progress.update(1)
        
    dataset_progress.close()
    
    # Summary
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    print(f"\n📊 Training Summary: {successful}/{total} datasets completed successfully")
    logger.info(f"Training pipeline completed: {successful}/{total} datasets successful")
    
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
    from models.model_selector import create_model
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
            
            # Default to optimized model for better performance
            model = create_model(
                model_type='optimized',
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
                    
                    predictions, student_abilities, item_difficulties, discrimination_params, z_values, kc_info = model(q_data, qa_data)
                    
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

  # Custom training parameters
  python main.py --datasets synthetic --epochs 50 --batch_size 64 --learning_rate 0.01

  # Train and evaluate single fold with custom epochs
  python main.py --datasets STATICS --single_fold --fold 0 --epochs 10

  # All datasets with original model architecture
  python main.py --all --model_type original --epochs 30

  # Skip training, only evaluate and visualize
  python main.py --skip_training --datasets STATICS
  
  # Quick IRT statistics display (fast extraction from weights)
  python main.py --show_irt --datasets synthetic STATICS --skip_training --skip_evaluation --skip_statistics --skip_visualization
  
  # Show full IRT statistics (requires data processing)
  python main.py --show_irt --irt_type full --datasets synthetic --single_fold --fold 0 --skip_training --skip_evaluation --skip_statistics --skip_visualization
        """
    )
    
    # Dataset selection  
    parser.add_argument('--all', action='store_true', help='Run pipeline for all available datasets')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to process')
    
    # Dataset filtering (like run_all_datasets.sh)
    parser.add_argument('--exclude', choices=['large', 'medium', 'small'], help='Exclude dataset category')
    parser.add_argument('--only', choices=['large', 'medium', 'small'], help='Include only dataset category')
    
    # Training options
    parser.add_argument('--single_fold', action='store_true', help='Train only single fold instead of 5-fold CV')
    parser.add_argument('--fold', type=int, default=0, help='Fold index for single fold training (0-4)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (overrides dataset defaults)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides dataset defaults)')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate (overrides default 0.001)')
    parser.add_argument('--model_type', choices=['original', 'optimized'], default='optimized', help='Model architecture type')
    
    # Pipeline control  
    parser.add_argument('--skip_training', action='store_true', help='Skip training phase')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation phase')
    parser.add_argument('--skip_statistics', action='store_true', help='Skip statistics extraction')
    parser.add_argument('--skip_visualization', action='store_true', help='Skip visualization generation')
    
    # Advanced options (from shell scripts)
    parser.add_argument('--quick', action='store_true', help='Quick mode: 1 epoch, single fold (for testing)')
    parser.add_argument('--time_estimate', action='store_true', help='Show time estimates and exit')
    parser.add_argument('--comparison', action='store_true', help='Generate comprehensive results comparison')
    
    # IRT statistics
    parser.add_argument('--show_irt', action='store_true', help='Display IRT statistics for specified datasets')
    parser.add_argument('--irt_type', choices=['fast', 'full'], default='fast', help='Type of IRT statistics to show')
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        args.epochs = 1
        args.single_fold = True
        print("🚀 Quick mode enabled: 1 epoch, single fold")
    
    # Validate arguments
    if args.fold < 0 or args.fold >= 5:
        print("Error: Fold index must be between 0 and 4")
        return 1
    
    # Get available datasets
    available_datasets = get_available_datasets()
    
    # Determine datasets to process
    if args.all:
        datasets = available_datasets
    elif args.datasets:
        # Validate requested datasets exist
        missing = [d for d in args.datasets if d not in available_datasets]
        if missing:
            print(f"❌ Datasets not found in data/: {missing}")
            print(f"Available datasets: {available_datasets}")
            return 1
        datasets = args.datasets
    else:
        print("Error: Must specify --all or --datasets")
        print(f"Available datasets: {available_datasets}")
        return 1
    
    # Apply category filtering
    datasets = filter_datasets_by_category(datasets, exclude=args.exclude, only=args.only)
    
    if not datasets:
        print("❌ No datasets available after filtering")
        return 1
    
    # Show time estimates if requested
    if args.time_estimate:
        estimate_training_time(datasets, args.epochs or 20, args.single_fold)
        return 0
    
    print(f"📁 Processing {len(datasets)} datasets: {datasets}")
    
    # Initialize comprehensive logging
    logger = setup_logging()
    
    print(f"{'='*60}")
    print("🚀 COMPREHENSIVE DEEP-2PL PIPELINE")
    print(f"{'='*60}")
    print(f"Datasets: {datasets or 'all available'}")
    print(f"Training mode: {'Single fold' if args.single_fold else '5-fold CV'}")
    if args.single_fold:
        print(f"Fold: {args.fold}")
    print(f"{'='*60}")
    
    logger.info("Pipeline started")
    logger.info(f"Arguments: datasets={datasets}, single_fold={args.single_fold}, fold={args.fold}")
    
    # Quick IRT statistics display (if requested)
    if args.show_irt:
        print("\n" + "="*20 + " IRT STATISTICS " + "="*21)
        selected_datasets = args.datasets or ['synthetic', 'assist2015', 'STATICS']
        for dataset in selected_datasets:
            if args.single_fold:
                load_irt_stats(dataset, fold_idx=args.fold, stats_type=args.irt_type)
            else:
                # Show stats for all folds
                for fold in range(5):
                    load_irt_stats(dataset, fold_idx=fold, stats_type=args.irt_type)
        
        if all([args.skip_training, args.skip_evaluation, args.skip_statistics, args.skip_visualization]):
            print("\nIRT statistics display completed.")
            return 0
    
    # Initialize results
    training_results = {}
    evaluation_results = {}
    statistics = {}
    
    # Phase 1: Training
    if not args.skip_training:
        print("\n" + "="*20 + " PHASE 1: TRAINING " + "="*20)
        training_results = train_all_models(
            datasets=datasets, 
            single_fold=args.single_fold, 
            fold_idx=args.fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_type=args.model_type
        )
    
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
    
    # Phase 6: Comprehensive Comparison (if requested or multiple datasets)
    if args.comparison or len(datasets) > 1:
        print("\n" + "="*18 + " PHASE 6: COMPARISON " + "="*18)
        generate_comprehensive_comparison()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())