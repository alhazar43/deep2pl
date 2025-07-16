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

from models.irt import DeepIRTModel
from data.dataloader import create_datasets, create_dataloader
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


def get_dataset_config(dataset_name, data_style, fold_idx=0):
    """Get configuration for specific dataset."""
    config = Config()
    
    # Common settings
    config.data_style = data_style
    config.fold_idx = fold_idx
    config.k_fold = 5
    
    # Dataset-specific configurations
    if data_style == "yeung":
        config.data_dir = "./data-yeung"
        
        if dataset_name == "assist2009_updated":
            config.dataset_name = "assist2009_updated"
            config.n_questions = 111
            config.seq_len = 1200  # Max sequence length: 1,146
            config.batch_size = 32
            config.n_epochs = 20
            config.q_matrix_path = "./data-yeung/assist2009_updated/assist2009_updated_qid_sid"
            config.skill_mapping_path = "./data-yeung/assist2009_updated/assist2009_updated_skill_mapping.txt"
            
        elif dataset_name == "STATICS":
            config.dataset_name = "STATICS"
            config.n_questions = 1223
            config.seq_len = 1200  # Max sequence length: 1,162
            config.batch_size = 16
            config.n_epochs = 20
            config.q_matrix_path = "./data-yeung/STATICS/Qmatrix.csv"
            config.skill_mapping_path = None
            
        elif dataset_name == "assist2015":
            config.dataset_name = "assist2015"
            config.n_questions = 100
            config.seq_len = 650  # Max sequence length: 618
            config.batch_size = 32
            config.n_epochs = 20
            config.q_matrix_path = "./data-yeung/assist2015/assist2015_qid_sid"
            config.skill_mapping_path = None
            
        elif dataset_name == "fsaif1tof3":
            config.dataset_name = "fsaif1tof3"
            config.n_questions = 100  # Will be determined from data
            config.seq_len = 700  # Max sequence length: 668
            config.batch_size = 32
            config.n_epochs = 20
            config.q_matrix_path = "./data-yeung/fsaif1tof3/conceptname_question_id.csv"
            config.skill_mapping_path = None
            
        elif dataset_name == "synthetic":
            config.dataset_name = "synthetic"
            config.n_questions = 100  # Will be determined from data
            config.seq_len = 50  # Max sequence length: 50 (uniform)
            config.batch_size = 32
            config.n_epochs = 20
            config.q_matrix_path = None
            config.skill_mapping_path = None
            
    elif data_style == "torch":
        config.data_dir = "./data-orig"
        
        if dataset_name == "assist2015":
            config.dataset_name = "assist2015"
            config.n_questions = 100
            config.seq_len = 50
            config.batch_size = 32
            config.n_epochs = 25
            config.q_matrix_path = None
            config.skill_mapping_path = None
            
        elif dataset_name == "assist2009":
            config.dataset_name = "assist2009"
            config.n_questions = 111
            config.seq_len = 50
            config.batch_size = 32
            config.n_epochs = 20
            config.q_matrix_path = None
            config.skill_mapping_path = None
    
    # Model configuration
    config.memory_size = 50
    config.key_memory_state_dim = 50
    config.value_memory_state_dim = 200
    config.summary_vector_dim = 50
    config.q_embed_dim = 50
    config.qa_embed_dim = 200
    config.ability_scale = 3.0
    config.use_discrimination = True
    config.discrimination_type = "static"  # "static", "dynamic", or "both"
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
    config.save_dir = f"checkpoints_{data_style}"
    config.log_dir = f"logs_{data_style}"
    
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
            
            predictions, student_abilities, item_difficulties, item_discriminations, z_values, kc_info = model(q_data, qa_data)
            
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


def train_model(config, logger):
    """Main training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    logger.info(f"Loading {config.dataset_name} dataset...")
    try:
        train_dataset, valid_dataset, test_dataset = create_datasets(
            data_style=config.data_style,
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
        key_memory_state_dim=config.key_memory_state_dim,
        value_memory_state_dim=config.value_memory_state_dim,
        summary_vector_dim=config.summary_vector_dim,
        q_embed_dim=config.q_embed_dim,
        qa_embed_dim=config.qa_embed_dim,
        ability_scale=config.ability_scale,
        use_discrimination=config.use_discrimination,
        discrimination_type=config.discrimination_type,
        dropout_rate=config.dropout_rate,
        q_matrix_path=getattr(config, 'q_matrix_path', None),
        skill_mapping_path=getattr(config, 'skill_mapping_path', None)
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Data: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)} | "
               f"Model: {total_params:,} params | Per-KC: {model.per_kc_mode}"
               + (f" ({model.n_kcs} KCs)" if model.per_kc_mode else ""))
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    logger.info("Starting training...")
    best_valid_auc = 0.0
    
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
            
            predictions, student_abilities, item_difficulties, item_discriminations, z_values, kc_info = model(q_data, qa_data)
            loss = model.compute_loss(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update batch progress bar
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Update epoch progress bar
        epoch_pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}")
        
        if (epoch + 1) % config.eval_every == 0:
            valid_loss, valid_auc, valid_acc = evaluate_model(model, valid_loader, device, desc="Validation")
            
            logger.info(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Valid={valid_loss:.4f}, AUC={valid_auc:.4f}")
            
            if (epoch + 1) % config.save_every == 0:
                checkpoint_path = os.path.join(config.save_dir, f"checkpoint_epoch_{epoch+1}_{config.dataset_name}.pth")
                save_checkpoint(model, optimizer, epoch+1, valid_loss, checkpoint_path, config.dataset_name)
            
            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                best_path = os.path.join(config.save_dir, f"best_model_{config.dataset_name}.pth")
                save_checkpoint(model, optimizer, epoch+1, valid_loss, best_path, config.dataset_name, is_best=True)
                epoch_pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}", best_auc=f"{best_valid_auc:.4f}")
    
    epoch_pbar.close()
    
    logger.info("Final evaluation on test set...")
    test_loss, test_auc, test_acc = evaluate_model(model, test_loader, device, desc="Testing")
    logger.info(f"Test: Loss={test_loss:.4f}, AUC={test_auc:.4f}, Acc={test_acc:.4f}")
    
    final_path = os.path.join(config.save_dir, f"final_model_{config.dataset_name}.pth")
    save_checkpoint(model, optimizer, config.n_epochs, test_loss, final_path, config.dataset_name)
    
    config_path = os.path.join(config.save_dir, f"config_{config.dataset_name}.json")
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best validation AUC: {best_valid_auc:.4f}")
    logger.info(f"Final test AUC: {test_auc:.4f}")
    
    return best_valid_auc, test_auc


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Train Deep-IRT Model')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['assist2009_updated', 'assist2015', 'STATICS', 'assist2009', 'fsaif1tof3', 'synthetic'],
                        help='Dataset name')
    parser.add_argument('--data_style', type=str, required=True,
                        choices=['yeung', 'torch'],
                        help='Data format style')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index for cross-validation (0-4)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides default)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides default)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides default)')
    parser.add_argument('--use_discrimination', action='store_true',
                        help='Enable discrimination parameters (2PL model)')
    parser.add_argument('--discrimination_type', type=str, default='static',
                        choices=['static', 'dynamic', 'both'],
                        help='Type of discrimination estimation')
    
    args = parser.parse_args()
    
    if args.fold < 0 or args.fold >= 5:
        print("Error: Fold index must be between 0 and 4")
        return
    
    config = get_dataset_config(args.dataset, args.data_style, args.fold)
    
    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.use_discrimination:
        config.use_discrimination = True
        config.discrimination_type = args.discrimination_type
    
    logger = setup_logging(config.log_dir, config.dataset_name)
    
    logger.info(f"=== Training Deep-IRT ===")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Data style: {args.data_style}")
    logger.info(f"Fold: {args.fold}")
    logger.info(f"Epochs: {config.n_epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    
    best_auc, test_auc = train_model(config, logger)
    
    print(f"\n=== Training Summary ===")
    print(f"Dataset: {args.dataset}")
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"Final Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()