#!/usr/bin/env python3
"""
Evaluation Script for Deep-IRT Model

Supports:
- Loading trained models from save_models directory
- Evaluating on test sets
- Saving detailed evaluation results
"""

import os
import torch
import numpy as np
import json
import argparse
import logging
from datetime import datetime
# tqdm removed for performance
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, roc_curve

from models.model_selector import create_model
from data.dataloader import create_datasets, create_dataloader
from utils.config import Config


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def load_trained_model(model_path, device='cpu'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        model: Loaded DeepIRT model (original or optimized)
        config: Model configuration
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract dataset name and find corresponding config
    if 'dataset_name' in checkpoint:
        dataset_name = checkpoint['dataset_name']
    else:
        # Extract from filename
        filename = os.path.basename(model_path)
        if 'assist2009_updated' in filename:
            dataset_name = 'assist2009_updated'
        elif 'assist2015' in filename:
            dataset_name = 'assist2015'
        elif 'STATICS' in filename:
            dataset_name = 'STATICS'
        elif 'fsaif1tof3' in filename:
            dataset_name = 'fsaif1tof3'
        elif 'synthetic' in filename:
            dataset_name = 'synthetic'
        else:
            raise ValueError(f"Cannot determine dataset from filename: {filename}")
    
    # Look for corresponding config file
    config_dir = os.path.dirname(model_path)
    config_files = [
        os.path.join(config_dir, f"config_{dataset_name}.json"),
        os.path.join(config_dir, f"config_{dataset_name}_fold0.json"),
        os.path.join(config_dir, f"config_{dataset_name}_fold1.json"),
        os.path.join(config_dir, f"config_{dataset_name}_fold2.json"),
        os.path.join(config_dir, f"config_{dataset_name}_fold3.json"),
        os.path.join(config_dir, f"config_{dataset_name}_fold4.json"),
    ]
    
    config_path = None
    for cf in config_files:
        if os.path.exists(cf):
            config_path = cf
            break
    
    if config_path is None:
        logger.warning(f"No config file found for {dataset_name}, using default configuration")
        # Create default config based on dataset name
        if dataset_name == 'assist2009_updated':
            from train import get_dataset_config
            config = get_dataset_config('assist2009_updated', 'yeung')
        elif dataset_name == 'STATICS':
            from train import get_dataset_config
            config = get_dataset_config('STATICS', 'yeung')
        elif dataset_name == 'assist2015':
            from train import get_dataset_config
            config = get_dataset_config('assist2015', 'yeung')
        elif dataset_name == 'fsaif1tof3':
            from train import get_dataset_config
            config = get_dataset_config('fsaif1tof3', 'yeung')
        elif dataset_name == 'synthetic':
            from train import get_dataset_config
            config = get_dataset_config('synthetic', 'yeung')
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    else:
        # Load config from file
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
    
    # Extract model dimensions from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    actual_n_questions = state_dict['q_embed.weight'].shape[0] - 1
    
    # Determine model type from config or default to optimized
    model_type = getattr(config, 'model_type', 'optimized')
    
    # Create model
    model = create_model(
        model_type=model_type,
        n_questions=actual_n_questions,
        memory_size=config.memory_size,
        key_dim=getattr(config, 'key_dim', 50),
        value_dim=getattr(config, 'value_dim', 200),
        summary_dim=getattr(config, 'summary_dim', 50),
        q_embed_dim=config.q_embed_dim,
        qa_embed_dim=config.qa_embed_dim,
        ability_scale=config.ability_scale,
        use_discrimination=config.use_discrimination,
        dropout_rate=config.dropout_rate,
        q_matrix_path=getattr(config, 'q_matrix_path', None),
        skill_mapping_path=getattr(config, 'skill_mapping_path', None)
    ).to(device)
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info(f"Loaded model: {dataset_name}, {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, config


def evaluate_model_detailed(model, data_loader, device, desc="Evaluating"):
    """
    Perform detailed evaluation of the model.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        desc: Description for progress bar
        
    Returns:
        dict: Detailed evaluation results
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    all_question_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            q_data = batch['q_data'].to(device)
            qa_data = batch['qa_data'].to(device)
            targets = batch['target'].to(device)
            
            predictions, student_abilities, item_difficulties, discrimination_params, z_values, kc_info = model(q_data, qa_data)
            
            loss = model.compute_loss(predictions, targets)
            total_loss += loss.item()
            
            # Collect predictions and targets
            mask = targets >= 0  # Valid targets
            if mask.any():
                all_predictions.extend(predictions[mask].cpu().numpy())
                all_targets.extend(targets[mask].cpu().numpy())
                all_question_ids.extend(q_data[mask].cpu().numpy())
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    if all_predictions:
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_question_ids = np.array(all_question_ids)
        
        # Calculate metrics
        auc = roc_auc_score(all_targets, all_predictions)
        acc = accuracy_score(all_targets, all_predictions > 0.5)
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(all_targets, all_predictions)
        
        # Calculate Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(all_targets, all_predictions)
        
        # Per-question statistics
        unique_questions = np.unique(all_question_ids[all_question_ids > 0])
        per_question_stats = {}
        
        for q_id in unique_questions:
            q_mask = all_question_ids == q_id
            if np.sum(q_mask) > 0:
                q_predictions = all_predictions[q_mask]
                q_targets = all_targets[q_mask]
                
                if len(np.unique(q_targets)) > 1:  # Both correct and incorrect answers exist
                    q_auc = roc_auc_score(q_targets, q_predictions)
                else:
                    q_auc = None
                
                per_question_stats[int(q_id)] = {
                    'count': int(np.sum(q_mask)),
                    'accuracy': float(accuracy_score(q_targets, q_predictions > 0.5)),
                    'auc': float(q_auc) if q_auc is not None else None,
                    'mean_difficulty': float(np.mean(q_targets)),
                    'mean_prediction': float(np.mean(q_predictions))
                }
        
        results = {
            'loss': float(avg_loss),
            'auc': float(auc),
            'accuracy': float(acc),
            'n_samples': len(all_predictions),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'pr_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            },
            'per_question_stats': per_question_stats,
            'prediction_distribution': {
                'mean': float(np.mean(all_predictions)),
                'std': float(np.std(all_predictions)),
                'min': float(np.min(all_predictions)),
                'max': float(np.max(all_predictions))
            }
        }
    else:
        results = {
            'loss': float(avg_loss),
            'auc': 0.0,
            'accuracy': 0.0,
            'n_samples': 0,
            'error': 'No valid predictions found'
        }
    
    return results


def main():
    """Main function for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate Deep-IRT Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, 
                        choices=['assist2009_updated', 'assist2015', 'STATICS', 'fsaif1tof3', 'synthetic', 
                                'assist2009', 'assist2017', 'statics2011', 'kddcup2010'],
                        help='Dataset name (optional, will be inferred from model path)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'valid', 'test'],
                        help='Which split to evaluate on')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index for validation/train evaluation')
    parser.add_argument('--output_dir', type=str, default='results/test',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model, config = load_trained_model(args.model_path, device)
    
    # Create dataset
    logger.info(f"Loading {config.dataset_name} dataset...")
    try:
        train_dataset, valid_dataset, test_dataset = create_datasets(
            data_dir=config.data_dir,
            dataset_name=config.dataset_name,
            seq_len=config.seq_len,
            n_questions=config.n_questions,
            k_fold=config.k_fold,
            fold_idx=args.fold
        )
        
        # Select appropriate dataset
        if args.split == 'train':
            eval_dataset = train_dataset
        elif args.split == 'valid':
            eval_dataset = valid_dataset
        else:  # test
            eval_dataset = test_dataset
            
        eval_loader = create_dataloader(eval_dataset, batch_size=32, shuffle=False)
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Evaluate model
    logger.info(f"Evaluating on {args.split} set...")
    results = evaluate_model_detailed(model, eval_loader, device, desc=f"Evaluating {args.split}")
    
    # Add metadata
    results['metadata'] = {
        'model_path': args.model_path,
        'dataset': config.dataset_name,
        'split': args.split,
        'fold': args.fold,
        'evaluation_time': datetime.now().isoformat(),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'per_kc_mode': model.per_kc_mode,
        'use_discrimination': config.use_discrimination
    }
    
    # Save results
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    results_path = os.path.join(args.output_dir, f"eval_{model_name}_{args.split}.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("=== Evaluation Results ===")
    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Split: {args.split}")
    if 'error' not in results:
        logger.info(f"Loss: {results['loss']:.4f}")
        logger.info(f"AUC: {results['auc']:.4f}")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Samples: {results['n_samples']:,}")
        logger.info(f"Questions: {len(results['per_question_stats'])}")
    else:
        logger.error(f"Evaluation failed: {results['error']}")
    
    logger.info(f"Detailed results saved to: {results_path}")
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Model: {model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Split: {args.split}")
    if 'error' not in results:
        print(f"Loss: {results['loss']:.4f}")
        print(f"AUC: {results['auc']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Samples: {results['n_samples']:,}")
    else:
        print(f"Error: {results['error']}")


if __name__ == "__main__":
    main()