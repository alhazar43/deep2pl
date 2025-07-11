import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging
from tqdm import tqdm
import json

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Disable tensorboard logging.")

from models.model import DeepIRTModel
from data.dataloader import KnowledgeTracingDataset, CSVKnowledgeTracingDataset, create_dataloader, create_datasets
from utils.config import get_config


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def setup_logging(config):
    """Setup logging configuration."""
    os.makedirs(config.log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO if config.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger


def evaluate_model(model, dataloader, device, eval_type="Validation"):
    """Evaluate model on given dataloader with progress bar."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Enhanced evaluation progress bar
    eval_pbar = tqdm(
        dataloader,
        desc=f'📊 {eval_type:^10}',
        leave=False,
        dynamic_ncols=True,
        colour='green'
    )
    
    batch_losses = []
    batch_accs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_pbar):
            q_data = batch['q_data'].to(device)
            qa_data = batch['qa_data'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            predictions, _, _, _ = model(q_data, qa_data)
            
            # Compute loss
            loss = model.compute_loss(predictions, targets)
            total_loss += loss.item()
            batch_losses.append(loss.item())
            
            # Compute accuracy
            mask = targets >= 0
            masked_predictions = predictions[mask]
            masked_targets = targets[mask]
            
            predicted_labels = (masked_predictions > 0.5).float()
            correct = (predicted_labels == masked_targets).sum().item()
            total_correct += correct
            total_samples += masked_targets.size(0)
            
            # Batch accuracy
            batch_acc = correct / masked_targets.size(0) if masked_targets.size(0) > 0 else 0.0
            batch_accs.append(batch_acc)
            
            # Running averages for progress display
            current_avg_loss = np.mean(batch_losses)
            current_avg_acc = np.mean(batch_accs)
            
            # Update progress bar
            eval_pbar.set_postfix({
                'Loss': f'{current_avg_loss:.4f}',
                'Acc': f'{current_avg_acc:.4f}',
                'Samples': f'{total_samples:5d}'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, accuracy


def train_model(config):
    """Main training function."""
    logger = setup_logging(config)
    logger.info(f"Configuration:\n{config}")
    logger.info(f"Key config values: n_questions={config.n_questions}, data_style={config.data_style}, dataset_name={config.dataset_name}")
    
    # Set random seed
    set_seed(config.seed)
    
    # Handle device selection
    if config.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {config.device}")
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    if config.tensorboard and TENSORBOARD_AVAILABLE:
        os.makedirs(config.log_dir, exist_ok=True)
        writer = SummaryWriter(config.log_dir)
    else:
        writer = None
        if config.tensorboard:
            logger.warning("TensorBoard requested but not available. Skipping tensorboard logging.")
    
    # Load datasets using the new unified interface
    logger.info(f"Loading datasets using {config.data_style} style...")
    
    # Determine data directory based on style
    if config.data_style == 'yeung':
        data_dir = config.data_dir if 'yeung' in config.data_dir else './data-yeung'
    else:  # torch
        data_dir = config.data_dir if 'orig' in config.data_dir else './data-orig'
    
    logger.info(f"Using data directory: {data_dir}")
    logger.info(f"Dataset: {config.dataset_name}, Fold: {config.fold_idx}/{config.k_fold}")
    
    train_dataset, valid_dataset, test_dataset = create_datasets(
        data_style=config.data_style,
        data_dir=data_dir,
        dataset_name=config.dataset_name,
        seq_len=config.seq_len,
        n_questions=config.n_questions,
        k_fold=config.k_fold,
        fold_idx=config.fold_idx
    )
    
    train_dataloader = create_dataloader(train_dataset, config.batch_size, shuffle=True)
    valid_dataloader = create_dataloader(valid_dataset, config.batch_size, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, config.batch_size, shuffle=False)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(valid_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    logger.info("Initializing model...")
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
        dropout_rate=config.dropout_rate
    )
    
    model = model.to(config.device)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    logger.info("Starting training...")
    best_test_acc = 0
    
    for epoch in range(config.n_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0
        
        # Training phase with enhanced progress bar
        pbar = tqdm(
            train_dataloader, 
            desc=f'🚀 Epoch {epoch+1:2d}/{config.n_epochs}',
            leave=True,
            dynamic_ncols=True,
            colour='blue'
        )
        
        batch_metrics = []
        for batch_idx, batch in enumerate(pbar):
            q_data = batch['q_data'].to(config.device)
            qa_data = batch['qa_data'].to(config.device)
            targets = batch['target'].to(config.device)
            
            # Forward pass
            predictions, student_abilities, item_difficulties, z_values = model(q_data, qa_data)
            
            # Compute loss
            loss = model.compute_loss(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            
            # Compute accuracy
            mask = targets >= 0
            masked_predictions = predictions[mask]
            masked_targets = targets[mask]
            
            predicted_labels = (masked_predictions > 0.5).float()
            correct = (predicted_labels == masked_targets).sum().item()
            train_correct += correct
            train_samples += masked_targets.size(0)
            
            # Batch metrics
            batch_acc = correct / masked_targets.size(0) if masked_targets.size(0) > 0 else 0.0
            batch_metrics.append({'loss': loss.item(), 'acc': batch_acc})
            
            # Running averages for smoother display
            recent_loss = np.mean([m['loss'] for m in batch_metrics[-10:]])
            recent_acc = np.mean([m['acc'] for m in batch_metrics[-10:]])
            
            # Enhanced progress bar with running stats
            pbar.set_postfix({
                'Loss': f'{recent_loss:.4f}',
                'Acc': f'{recent_acc:.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}',
                'Batch': f'{batch_idx+1:4d}/{len(train_dataloader)}'
            })
        
        # Update learning rate
        scheduler.step()
        
        # Compute training metrics
        avg_train_loss = train_loss / len(train_dataloader)
        train_acc = train_correct / train_samples if train_samples > 0 else 0
        
        logger.info(f'Epoch {epoch+1}/{config.n_epochs}:')
        logger.info(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        # Evaluation phase
        if (epoch + 1) % config.eval_every == 0:
            # Validate on validation set
            valid_loss, valid_acc = evaluate_model(model, valid_dataloader, config.device, "Validation")
            logger.info(f'  Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
            
            # Test on test set (less frequent)
            test_loss, test_acc = evaluate_model(model, test_dataloader, config.device, "Test")
            logger.info(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            
            # Save best model based on validation accuracy
            if valid_acc > best_test_acc:
                best_test_acc = valid_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_acc': valid_acc,
                    'test_acc': test_acc,
                    'config': config.__dict__
                }, os.path.join(config.save_dir, 'best_model.pth'))
                logger.info(f'  New best model saved with validation accuracy: {valid_acc:.4f}')
            
            # TensorBoard logging
            if config.tensorboard and writer is not None:
                writer.add_scalar('Loss/Train', avg_train_loss, epoch)
                writer.add_scalar('Loss/Valid', valid_loss, epoch)
                writer.add_scalar('Loss/Test', test_loss, epoch)
                writer.add_scalar('Accuracy/Train', train_acc, epoch)
                writer.add_scalar('Accuracy/Valid', valid_acc, epoch)
                writer.add_scalar('Accuracy/Test', test_acc, epoch)
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__
            }, os.path.join(config.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': config.n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__
    }, os.path.join(config.save_dir, 'final_model.pth'))
    
    logger.info(f"Training completed. Best validation accuracy: {best_test_acc:.4f}")
    
    # Final evaluation on test set
    final_test_loss, final_test_acc = evaluate_model(model, test_dataloader, config.device, "Final Test")
    logger.info(f"Final test accuracy: {final_test_acc:.4f}")
    
    if config.tensorboard and writer is not None:
        writer.close()


def main():
    """Main function."""
    config = get_config()
    train_model(config)


if __name__ == "__main__":
    main()