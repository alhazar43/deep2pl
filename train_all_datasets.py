#!/usr/bin/env python3
"""
Train the fixed Deep-IRT model on all datasets for 30 epochs.
Comprehensive benchmark across all available datasets.
"""

import os
import sys
import json
import time
from datetime import datetime
import logging
import argparse

# Setup logging
def setup_logging():
    """Setup logging configuration."""
    log_filename = f"logs/train_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_training(dataset, epochs=30, logger=None):
    """Run training for a specific dataset."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training for dataset: {dataset}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"{'='*60}")
    
    # Run training command
    cmd = f"python train.py --dataset {dataset} --epochs {epochs}"
    
    start_time = time.time()
    
    try:
        # Execute training
        import subprocess
        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per dataset
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"✓ {dataset} completed successfully in {duration:.1f}s")
            
            # Parse results from output
            output_lines = result.stdout.split('\n')
            summary_data = {
                'dataset': dataset,
                'status': 'success',
                'duration_seconds': duration,
                'epochs': epochs
            }
            
            # Extract final metrics from output
            for line in output_lines[-20:]:  # Check last 20 lines
                if 'Validation AUC:' in line:
                    try:
                        val_auc = float(line.split('Validation AUC:')[1].split('±')[0].strip())
                        summary_data['val_auc_mean'] = val_auc
                    except:
                        pass
                if 'Test AUC:' in line:
                    try:
                        test_auc = float(line.split('Test AUC:')[1].split('±')[0].strip())
                        summary_data['test_auc_mean'] = test_auc
                    except:
                        pass
                if 'Test Accuracy:' in line:
                    try:
                        test_acc = float(line.split('Test Accuracy:')[1].split('±')[0].strip())
                        summary_data['test_acc_mean'] = test_acc
                    except:
                        pass
            
            return summary_data
            
        else:
            logger.error(f"✗ {dataset} failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return {
                'dataset': dataset,
                'status': 'failed',
                'duration_seconds': duration,
                'error': result.stderr,
                'epochs': epochs
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {dataset} timed out after 2 hours")
        return {
            'dataset': dataset,
            'status': 'timeout',
            'duration_seconds': 7200,
            'epochs': epochs
        }
    except Exception as e:
        logger.error(f"✗ {dataset} failed with exception: {str(e)}")
        return {
            'dataset': dataset,
            'status': 'error',
            'error': str(e),
            'epochs': epochs
        }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train all datasets')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs (default: 30)')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to train (optional)')
    parser.add_argument('--exclude', nargs='+', help='Datasets to exclude from training')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # All available datasets (auto-detected by the system)
    all_datasets = [
        'STATICS',           # Pre-split, 956 questions, 98 KCs
        'assist2009_updated', # Pre-split, 70 questions
        'fsaif1tof3',        # Pre-split, 1722 questions
        'assist2015',        # Single-file, 96 questions
        'assist2017',        # Single-file, 102 questions
        'synthetic',         # Single-file, 50 questions
        'assist2009',        # Single-file, 124 questions
        'statics2011',       # Single-file, 1221 questions
        'kddcup2010'         # Single-file, 649 questions
    ]
    
    # Filter datasets based on arguments
    if args.datasets:
        datasets_to_train = [d for d in args.datasets if d in all_datasets]
        logger.info(f"Training specific datasets: {datasets_to_train}")
    else:
        datasets_to_train = all_datasets
        logger.info(f"Training all {len(datasets_to_train)} datasets")
    
    if args.exclude:
        datasets_to_train = [d for d in datasets_to_train if d not in args.exclude]
        logger.info(f"Excluding datasets: {args.exclude}")
        logger.info(f"Final dataset list: {datasets_to_train}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPREHENSIVE DEEP-IRT TRAINING")
    logger.info(f"{'='*80}")
    logger.info(f"Total datasets: {len(datasets_to_train)}")
    logger.info(f"Epochs per dataset: {args.epochs}")
    logger.info(f"Expected total time: ~{len(datasets_to_train) * 20} minutes (estimate)")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track overall progress
    overall_start = time.time()
    results = []
    
    # Train each dataset
    for i, dataset in enumerate(datasets_to_train, 1):
        logger.info(f"\n[{i}/{len(datasets_to_train)}] Processing {dataset}...")
        
        result = run_training(dataset, args.epochs, logger)
        results.append(result)
        
        # Save intermediate results
        results_file = f"results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('results', exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Final summary
    overall_duration = time.time() - overall_start
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total time: {overall_duration/3600:.1f} hours")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary statistics
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    logger.info(f"\nSUMMARY:")
    logger.info(f"✓ Successful: {len(successful)}/{len(results)}")
    logger.info(f"✗ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        logger.info(f"\nSUCCESSFUL RESULTS:")
        logger.info(f"{'Dataset':<15} {'Test AUC':<10} {'Test Acc':<10} {'Duration':<10}")
        logger.info(f"{'-'*50}")
        
        for result in successful:
            dataset = result['dataset']
            test_auc = result.get('test_auc_mean', 'N/A')
            test_acc = result.get('test_acc_mean', 'N/A')
            duration = f"{result['duration_seconds']/60:.1f}m"
            
            auc_str = f"{test_auc:.4f}" if isinstance(test_auc, float) else str(test_auc)
            acc_str = f"{test_acc:.4f}" if isinstance(test_acc, float) else str(test_acc)
            
            logger.info(f"{dataset:<15} {auc_str:<10} {acc_str:<10} {duration:<10}")
        
        # Best performing dataset
        if any('test_auc_mean' in r for r in successful):
            best_result = max(
                [r for r in successful if 'test_auc_mean' in r], 
                key=lambda x: x['test_auc_mean']
            )
            logger.info(f"\n🏆 Best performing: {best_result['dataset']} "
                       f"(AUC: {best_result['test_auc_mean']:.4f})")
    
    if failed:
        logger.info(f"\nFAILED DATASETS:")
        for result in failed:
            logger.info(f"✗ {result['dataset']}: {result['status']}")
    
    # Save final results
    final_results = {
        'training_summary': {
            'total_datasets': len(datasets_to_train),
            'successful': len(successful),
            'failed': len(failed),
            'total_duration_hours': overall_duration / 3600,
            'epochs_per_dataset': args.epochs,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'individual_results': results
    }
    
    final_file = f"results/final_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {final_file}")
    logger.info(f"Logs saved to: {log_filename}")


if __name__ == "__main__":
    main()