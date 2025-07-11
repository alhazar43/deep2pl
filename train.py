import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import logging
from tqdm import tqdm
import json

from models.model import DeepIRTModel
from data.dataloader import KnowledgeTracingDataset, CSVKnowledgeTracingDataset, create_dataloader
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


def evaluate_model(model, dataloader, device):
    """Evaluate model on given dataloader."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            q_data = batch['q_data'].to(device)
            qa_data = batch['qa_data'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            predictions, _, _, _ = model(q_data, qa_data)
            
            # Compute loss
            loss = model.compute_loss(predictions, targets)
            total_loss += loss.item()
            
            # Compute accuracy
            mask = targets >= 0
            masked_predictions = predictions[mask]
            masked_targets = targets[mask]
            
            predicted_labels = (masked_predictions > 0.5).float()
            correct = (predicted_labels == masked_targets).sum().item()
            total_correct += correct
            total_samples += masked_targets.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, accuracy


def train_model(config):
    """Main training function."""
    logger = setup_logging(config)
    logger.info(f"Configuration:\n{config}")
    
    # Set random seed
    set_seed(config.seed)
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    if config.tensorboard:
        os.makedirs(config.log_dir, exist_ok=True)
        writer = SummaryWriter(config.log_dir)
    
    # Load datasets
    logger.info("Loading datasets...")
    
    if config.data_format == 'txt':
        train_dataset = KnowledgeTracingDataset(
            data_path=os.path.join(config.data_dir, config.train_file),
            seq_len=config.seq_len,
            n_questions=config.n_questions
        )
        
        test_dataset = KnowledgeTracingDataset(
            data_path=os.path.join(config.data_dir, config.test_file),
            seq_len=config.seq_len,
            n_questions=config.n_questions
        )
    else:
        train_dataset = CSVKnowledgeTracingDataset(
            csv_path=os.path.join(config.data_dir, config.train_file),
            seq_len=config.seq_len,
            n_questions=config.n_questions
        )
        
        test_dataset = CSVKnowledgeTracingDataset(
            csv_path=os.path.join(config.data_dir, config.test_file),
            seq_len=config.seq_len,
            n_questions=config.n_questions
        )
    
    train_dataloader = create_dataloader(train_dataset, config.batch_size, shuffle=True)
    test_dataloader = create_dataloader(test_dataset, config.batch_size, shuffle=False)
    
    logger.info(f"Training samples: {len(train_dataset)}")
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
        
        # Training phase
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{config.n_epochs}')
        for batch in pbar:
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
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct/masked_targets.size(0):.4f}' if masked_targets.size(0) > 0 else '0.0000'
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
            test_loss, test_acc = evaluate_model(model, test_dataloader, config.device)
            logger.info(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'config': config.__dict__
                }, os.path.join(config.save_dir, 'best_model.pth'))
                logger.info(f'  New best model saved with test accuracy: {test_acc:.4f}')
            
            # TensorBoard logging
            if config.tensorboard:
                writer.add_scalar('Loss/Train', avg_train_loss, epoch)
                writer.add_scalar('Loss/Test', test_loss, epoch)
                writer.add_scalar('Accuracy/Train', train_acc, epoch)
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
    
    logger.info(f"Training completed. Best test accuracy: {best_test_acc:.4f}")
    
    if config.tensorboard:
        writer.close()


def main():
    """Main function."""
    config = get_config()
    train_model(config)


if __name__ == "__main__":
    main()