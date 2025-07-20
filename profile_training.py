#!/usr/bin/env python3
"""
Quick profiling script to identify training bottlenecks.
"""

import torch
import time
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))

from models.model import DeepIRTModel
from dataloader import create_datasets, create_dataloader, get_dataset_info

def profile_training():
    """Profile training to find bottlenecks."""
    dataset_name = 'assist2009_updated'
    
    # Get dataset info
    dataset_info = get_dataset_info(dataset_name)
    n_questions = dataset_info['n_questions']
    
    print(f"Dataset: {dataset_name}")
    print(f"Questions: {n_questions}")
    
    # Create datasets
    train_dataset, _, _ = create_datasets(
        dataset_name=dataset_name, 
        n_questions=n_questions,
        fold_idx=0
    )
    
    # Create data loader
    train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
    print(f"Batches: {len(train_loader)}")
    print(f"Training samples: {len(train_dataset)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = DeepIRTModel(
        n_questions=n_questions,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Profile different batch sizes
    batch_sizes = [16, 32, 64, 128]
    
    for bs in batch_sizes:
        print(f"\n=== Profiling batch size {bs} ===")
        
        test_loader = create_dataloader(train_dataset, batch_size=bs, shuffle=False)
        
        # Time a few batches
        times = []
        model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 3:  # Test 3 batches
                    break
                
                q_data = batch['q_data'].to(device)
                qa_data = batch['qa_data'].to(device)
                targets = batch['target'].to(device)
                
                start_time = time.time()
                
                # Forward pass
                predictions, _, _, _, _ = model(q_data, qa_data)
                loss = model.compute_loss(predictions, targets)
                
                end_time = time.time()
                batch_time = end_time - start_time
                times.append(batch_time)
                
                print(f"  Batch {i+1}: {batch_time:.3f}s, "
                      f"Shape: {q_data.shape}, "
                      f"Loss: {loss.item():.4f}")
        
        if times:
            avg_time = sum(times) / len(times)
            total_batches = len(test_loader)
            estimated_epoch_time = avg_time * total_batches
            
            print(f"  Average batch time: {avg_time:.3f}s")
            print(f"  Estimated epoch time: {estimated_epoch_time:.1f}s ({estimated_epoch_time/60:.1f}m)")
            print(f"  Samples/second: {bs / avg_time:.1f}")
    
    # Profile sequence length impact
    print(f"\n=== Profiling sequence length impact ===")
    seq_lengths = [20, 50, 100]
    batch_size = 32
    
    for seq_len in seq_lengths:
        print(f"\nSequence length {seq_len}:")
        
        # Create dummy data
        q_data = torch.randint(1, n_questions + 1, (batch_size, seq_len)).to(device)
        qa_data = torch.randint(1, 2 * n_questions + 1, (batch_size, seq_len)).to(device)
        targets = torch.randint(0, 2, (batch_size, seq_len)).float().to(device)
        
        model.eval()
        times = []
        
        with torch.no_grad():
            for i in range(3):
                start_time = time.time()
                predictions, _, _, _, _ = model(q_data, qa_data)
                loss = model.compute_loss(predictions, targets)
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Time per timestep: {avg_time/seq_len*1000:.1f}ms")

def profile_memory_operations():
    """Profile individual memory operations."""
    print(f"\n=== Profiling Memory Operations ===")
    
    from models.memory import DKVMN
    
    batch_size = 32
    seq_len = 50
    memory_size = 50
    key_dim = 50
    value_dim = 200
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize memory
    memory = DKVMN(memory_size=memory_size, key_dim=key_dim, value_dim=value_dim).to(device)
    memory.init_value_memory(batch_size)
    
    # Create dummy data
    q_embed = torch.randn(batch_size, key_dim).to(device)
    qa_embed = torch.randn(batch_size, value_dim).to(device)
    
    # Profile operations
    operations = ['attention', 'read', 'write']
    
    for op in operations:
        times = []
        
        for i in range(10):
            start_time = time.time()
            
            if op == 'attention':
                correlation_weight = memory.attention(q_embed)
            elif op == 'read':
                correlation_weight = memory.attention(q_embed)
                read_content = memory.read(correlation_weight)
            elif op == 'write':
                correlation_weight = memory.attention(q_embed)
                memory.write(correlation_weight, qa_embed)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        print(f"  {op}: {avg_time*1000:.2f}ms")

if __name__ == "__main__":
    print("Training Performance Profiler")
    print("=" * 50)
    
    profile_training()
    profile_memory_operations()