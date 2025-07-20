#!/usr/bin/env python3
"""
Unified Training Script for Deep-IRT Model

Supports:
- Both data-orig and data-yeung formats
- Automatic per-KC detection based on Q-matrix availability
- Proper checkpoint naming: checkpoints_torch/best_model_assist2015.pth
"""

import os
import torch
import torch.optim as optim
import numpy as np
import json
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from models.model import DeepIRTModel
from data.dataloader import create_datasets, create_dataloader, get_dataset_info
from utils.config import Config


def setup_logging(log_dir, dataset_name):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{dataset_name}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_dataset_config(dataset_name, fold_idx=0):
    """Get configuration for specific dataset."""
    config = Config()
    
    # Common settings
    config.data_dir = "./data"
    config.fold_idx = fold_idx
    config.k_fold = 5
    config.dataset_name = dataset_name
    
    # Get dataset info automatically
    dataset_info = get_dataset_info(dataset_name, config.data_dir)
    config.n_questions = dataset_info['n_questions']
    config.q_matrix_path = dataset_info['qmatrix_path']
    config.skill_mapping_path = dataset_info['skill_mapping_path']
    
    # Dataset-specific configurations
    dataset_configs = {
        "assist2009_updated": {
            "seq_len": 1200,
            "batch_size": 32,
            "n_epochs": 20
        },
        "STATICS": {
            "seq_len": 1200,
            "batch_size": 16,
            "n_epochs": 20
        },
        "assist2015": {
            "seq_len": 650,
            "batch_size": 32,
            "n_epochs": 20
        },
        "fsaif1tof3": {
            "seq_len": 700,
            "batch_size": 32,
            "n_epochs": 20
        },
        "synthetic": {
            "seq_len": 50,
            "batch_size": 32,
            "n_epochs": 20
        },
        "assist2009": {
            "seq_len": 50,
            "batch_size": 32,
            "n_epochs": 20
        },
        "assist2017": {
            "seq_len": 50,
            "batch_size": 32,
            "n_epochs": 20
        },
        "statics2011": {
            "seq_len": 50,
            "batch_size": 32,
            "n_epochs": 20
        },
        "kddcup2010": {
            "seq_len": 50,
            "batch_size": 32,
            "n_epochs": 20
        }
    }
    
    if dataset_name in dataset_configs:
        ds_config = dataset_configs[dataset_name]
        config.seq_len = ds_config["seq_len"]
        config.batch_size = ds_config["batch_size"]
        config.n_epochs = ds_config["n_epochs"]
    else:
        # Default values
        config.seq_len = 50
        config.batch_size = 32
        config.n_epochs = 20
    
    # Model configuration
    config.memory_size = 50
    config.key_dim = 50
    config.value_dim = 200
    config.summary_dim = 50
    config.q_embed_dim = 50
    config.qa_embed_dim = 200
    config.ability_scale = 3.0
    config.use_discrimination = True  # Default to 2PL model
    config.dropout_rate = 0.1
    
    # Training configuration
    config.learning_rate = 0.001
    config.max_grad_norm = 5.0
    config.weight_decay = 1e-5
    config.eval_every = 5
    config.save_every = 10
    config.verbose = True
    config.seed = 42
    
    # Save directories
    config.save_dir = "save_models"
    config.log_dir = "logs"
    
    return config


def evaluate_model(model, data_loader, device, desc="Evaluating"):
    """Evaluate model performance."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False):
            q_data = batch['q_data'].to(device)
            qa_data = batch['qa_data'].to(device)
            targets = batch['target'].to(device)
            
            predictions, student_abilities, item_difficulties, z_values, kc_info = model(q_data, qa_data)
            
            loss = model.compute_loss(predictions, targets)
            total_loss += loss.item()
            
            mask = targets >= 0
            if mask.any():
                all_predictions.extend(predictions[mask].cpu().numpy())
                all_targets.extend(targets[mask].cpu().numpy())
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    if all_predictions:
        auc = roc_auc_score(all_targets, all_predictions)
        acc = accuracy_score(all_targets, np.array(all_predictions) > 0.5)
    else:
        auc = 0.0
        acc = 0.0
    
    return avg_loss, auc, acc


def save_checkpoint(model, optimizer, epoch, loss, save_path, dataset_name, is_best=False):
    """Save model checkpoint with proper naming."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'dataset_name': dataset_name,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        dir_path = os.path.dirname(save_path)
        best_path = os.path.join(dir_path, f"best_model_{dataset_name}.pth")
        torch.save(checkpoint, best_path)


def train_model(config, logger, fold_idx=None):
    """Main training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs("results/train", exist_ok=True)
    os.makedirs("results/valid", exist_ok=True)
    os.makedirs("results/test", exist_ok=True)
    
    logger.info(f"Loading {config.dataset_name} dataset...")
    try:
        train_dataset, valid_dataset, test_dataset = create_datasets(
            data_dir=config.data_dir,
            dataset_name=config.dataset_name,
            seq_len=config.seq_len,
            n_questions=config.n_questions,
            k_fold=config.k_fold,
            fold_idx=config.fold_idx
        )
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return
    
    train_loader = create_dataloader(train_dataset, config.batch_size, shuffle=True)
    valid_loader = create_dataloader(valid_dataset, config.batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, config.batch_size, shuffle=False)
    
    model = DeepIRTModel(
        n_questions=config.n_questions,
        memory_size=config.memory_size,
        key_dim=config.key_dim,
        value_dim=config.value_dim,
        summary_dim=config.summary_dim,
        q_embed_dim=config.q_embed_dim,
        qa_embed_dim=config.qa_embed_dim,
        ability_scale=config.ability_scale,
        use_discrimination=config.use_discrimination,
        dropout_rate=config.dropout_rate,
        q_matrix_path=getattr(config, 'q_matrix_path', None),
        skill_mapping_path=getattr(config, 'skill_mapping_path', None)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Data: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)} | "
               f"Model: {n_params:,} params | Per-KC: {model.per_kc_mode}"
               + (f" ({model.n_kcs} KCs)" if model.per_kc_mode else ""))
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    logger.info("Starting training...")
    best_auc = 0.0
    
    # Track metrics for plotting
    train_losses = []
    valid_losses = []
    valid_aucs = []
    valid_accs = []
    
    # Create epoch progress bar
    epoch_pbar = tqdm(range(config.n_epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Create batch progress bar
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")
        
        for batch_idx, batch in enumerate(batch_pbar):
            q_data = batch['q_data'].to(device)
            qa_data = batch['qa_data'].to(device)
            targets = batch['target'].to(device)
            
            predictions, student_abilities, item_difficulties, z_values, kc_info = model(q_data, qa_data)
            loss = model.compute_loss(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update batch progress bar
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_loss = total_loss / num_batches if num_batches > 0 else 0
        train_losses.append(train_loss)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}")
        
        if (epoch + 1) % config.eval_every == 0:
            valid_loss, valid_auc, valid_acc = evaluate_model(model, valid_loader, device, desc="Validation")
            valid_losses.append(valid_loss)
            valid_aucs.append(valid_auc)
            valid_accs.append(valid_acc)
            
            logger.info(f"Epoch {epoch+1}: Train={train_loss:.4f}, Valid={valid_loss:.4f}, AUC={valid_auc:.4f}")
            
            if (epoch + 1) % config.save_every == 0:
                fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
                checkpoint_path = os.path.join(config.save_dir, f"checkpoint_epoch_{epoch+1}_{config.dataset_name}{fold_suffix}.pth")
                save_checkpoint(model, optimizer, epoch+1, valid_loss, checkpoint_path, config.dataset_name)
            
            if valid_auc > best_auc:
                best_auc = valid_auc
                fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
                best_path = os.path.join(config.save_dir, f"best_model_{config.dataset_name}{fold_suffix}.pth")
                save_checkpoint(model, optimizer, epoch+1, valid_loss, best_path, config.dataset_name, is_best=True)
                epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", best_auc=f"{best_auc:.4f}")
        else:
            # Add placeholder values for non-evaluation epochs
            valid_losses.append(None)
            valid_aucs.append(None)
            valid_accs.append(None)
    
    epoch_pbar.close()
    
    logger.info("Final evaluation on test set...")
    test_loss, test_auc, test_acc = evaluate_model(model, test_loader, device, desc="Testing")
    logger.info(f"Test: Loss={test_loss:.4f}, AUC={test_auc:.4f}, Acc={test_acc:.4f}")
    
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    final_path = os.path.join(config.save_dir, f"final_model_{config.dataset_name}{fold_suffix}.pth")
    save_checkpoint(model, optimizer, config.n_epochs, test_loss, final_path, config.dataset_name)
    
    config_path = os.path.join(config.save_dir, f"config_{config.dataset_name}{fold_suffix}.json")
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    # Save training metrics for plotting
    metrics = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_aucs': valid_aucs,
        'valid_accs': valid_accs,
        'best_valid_auc': best_auc,
        'test_loss': test_loss,
        'test_auc': test_auc,
        'test_acc': test_acc,
        'config': vars(config)
    }
    
    metrics_path = os.path.join("results/train", f"metrics_{config.dataset_name}{fold_suffix}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best validation AUC: {best_auc:.4f}")
    logger.info(f"Final test AUC: {test_auc:.4f}")
    
    return best_auc, test_auc, metrics


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Train Deep-IRT Model')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['assist2009_updated', 'assist2015', 'STATICS', 'assist2009', 'fsaif1tof3', 'synthetic', 
                                'assist2017', 'statics2011', 'kddcup2010'],
                        help='Dataset name')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index for cross-validation (0-4)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides default)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides default)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides default)')
    parser.add_argument('--use_discrimination', action='store_true',
                        help='Use discrimination parameter (2PL model) - now default')
    parser.add_argument('--no_discrimination', action='store_true',
                        help='Disable discrimination parameter (use 1PL model)')
    parser.add_argument('--single_fold', action='store_true',
                        help='Train only the specified fold instead of all 5 folds')
    
    args = parser.parse_args()
    
    if args.fold < 0 or args.fold >= 5:
        print("Error: Fold index must be between 0 and 4")
        return
    
    # Setup base config
    config = get_dataset_config(args.dataset, 0)  # Start with fold 0
    
    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.no_discrimination:
        config.use_discrimination = False
    elif args.use_discrimination:
        config.use_discrimination = True  # Redundant since it's now default
    
    logger = setup_logging(config.log_dir, config.dataset_name)
    
    if args.single_fold:
        # Train only the specified fold
        config.fold_idx = args.fold
        logger.info(f"=== Training Deep-IRT (Single Fold) ===")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Fold: {args.fold}")
        logger.info(f"Epochs: {config.n_epochs}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Learning rate: {config.learning_rate}")
        
        best_auc, test_auc, metrics = train_model(config, logger, fold_idx=args.fold)
        
        print(f"\n=== Training Summary (Fold {args.fold}) ===")
        print(f"Dataset: {args.dataset}")
        print(f"Best Validation AUC: {best_auc:.4f}")
        print(f"Final Test AUC: {test_auc:.4f}")
        
    else:
        # Train all 5 folds
        logger.info(f"=== Training Deep-IRT (5-Fold Cross Validation) ===")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Epochs: {config.n_epochs}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Learning rate: {config.learning_rate}")
        
        all_results = []
        for fold in range(5):
            logger.info(f"\n=== Starting Fold {fold} ===")
            config.fold_idx = fold
            
            best_auc, test_auc, metrics = train_model(config, logger, fold_idx=fold)
            all_results.append({
                'fold': fold,
                'best_valid_auc': best_auc,
                'test_auc': test_auc,
                'test_loss': metrics['test_loss'],
                'test_acc': metrics['test_acc']
            })
        
        # Calculate cross-validation statistics
        valid_aucs = [r['best_valid_auc'] for r in all_results]
        test_aucs = [r['test_auc'] for r in all_results]
        test_accs = [r['test_acc'] for r in all_results]
        
        cv_summary = {
            'dataset': args.dataset,
            'mean_valid_auc': np.mean(valid_aucs),
            'std_valid_auc': np.std(valid_aucs),
            'mean_test_auc': np.mean(test_aucs),
            'std_test_auc': np.std(test_aucs),
            'mean_test_acc': np.mean(test_accs),
            'std_test_acc': np.std(test_accs),
            'fold_results': all_results
        }
        
        # Save cross-validation summary
        cv_path = os.path.join("results/train", f"cv_summary_{config.dataset_name}.json")
        with open(cv_path, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
        print(f"\n=== 5-Fold Cross Validation Summary ===")
        print(f"Dataset: {args.dataset}")
        print(f"Validation AUC: {cv_summary['mean_valid_auc']:.4f} ± {cv_summary['std_valid_auc']:.4f}")
        print(f"Test AUC: {cv_summary['mean_test_auc']:.4f} ± {cv_summary['std_test_auc']:.4f}")
        print(f"Test Accuracy: {cv_summary['mean_test_acc']:.4f} ± {cv_summary['std_test_acc']:.4f}")
        
        logger.info("5-fold cross validation completed!")
        logger.info(f"Mean validation AUC: {cv_summary['mean_valid_auc']:.4f} ± {cv_summary['std_valid_auc']:.4f}")
        logger.info(f"Mean test AUC: {cv_summary['mean_test_auc']:.4f} ± {cv_summary['std_test_auc']:.4f}")


if __name__ == "__main__":
    main()