#!/usr/bin/env python3
"""
Example usage of the newly implemented item discrimination networks in Deep-2PL.

This example demonstrates how to use the three discrimination types:
1. Static discrimination: a_j = softplus(W_a [k_t; v_t] + b_a)
2. Dynamic discrimination: a_j = softplus(W_a f_t + b_a)  
3. Both (averaged): combination of static and dynamic
"""

import torch
import torch.nn as nn
import numpy as np
from models.irt import DeepIRTModel

def create_sample_data(n_questions=100, batch_size=8, seq_len=10):
    """Create sample data for demonstration."""
    # Generate random question sequences
    q_data = torch.randint(1, n_questions + 1, (batch_size, seq_len))
    
    # Generate random question-answer sequences
    qa_data = torch.randint(1, 2 * n_questions + 1, (batch_size, seq_len))
    
    # Generate random binary targets (0 or 1)
    targets = torch.randint(0, 2, (batch_size, seq_len)).float()
    
    return q_data, qa_data, targets

def demonstrate_discrimination_types():
    """Demonstrate the three discrimination types."""
    print("=== Deep-2PL Item Discrimination Demo ===\n")
    
    # Model parameters
    n_questions = 100
    memory_size = 20
    key_memory_state_dim = 64
    value_memory_state_dim = 64
    summary_vector_dim = 128
    
    # Create sample data
    q_data, qa_data, targets = create_sample_data(n_questions)
    batch_size, seq_len = q_data.shape
    
    print(f"Sample data shape: {batch_size} students × {seq_len} questions")
    print(f"Question IDs range: {q_data.min().item()} to {q_data.max().item()}")
    print(f"QA IDs range: {qa_data.min().item()} to {qa_data.max().item()}")
    
    # Test each discrimination type
    discrimination_types = ["static", "dynamic", "both"]
    
    for disc_type in discrimination_types:
        print(f"\n--- {disc_type.upper()} DISCRIMINATION ---")
        
        # Create model
        model = DeepIRTModel(
            n_questions=n_questions,
            memory_size=memory_size,
            key_memory_state_dim=key_memory_state_dim,
            value_memory_state_dim=value_memory_state_dim,
            summary_vector_dim=summary_vector_dim,
            use_discrimination=True,
            discrimination_type=disc_type,
            ability_scale=3.0
        )
        
        # Forward pass
        predictions, student_abilities, item_difficulties, item_discriminations, z_values = model(q_data, qa_data)
        
        # Compute loss
        loss = model.compute_loss(predictions, targets)
        
        # Print statistics
        print(f"Model type: 2PL with {disc_type} discrimination")
        print(f"Loss: {loss.item():.4f}")
        
        # Student ability statistics
        print(f"Student abilities - Mean: {student_abilities.mean():.3f}, Std: {student_abilities.std():.3f}")
        print(f"Student abilities - Range: [{student_abilities.min():.3f}, {student_abilities.max():.3f}]")
        
        # Item difficulty statistics  
        print(f"Item difficulties - Mean: {item_difficulties.mean():.3f}, Std: {item_difficulties.std():.3f}")
        print(f"Item difficulties - Range: [{item_difficulties.min():.3f}, {item_difficulties.max():.3f}]")
        
        # Item discrimination statistics
        print(f"Item discriminations - Mean: {item_discriminations.mean():.3f}, Std: {item_discriminations.std():.3f}")
        print(f"Item discriminations - Range: [{item_discriminations.min():.3f}, {item_discriminations.max():.3f}]")
        
        # Prediction statistics
        print(f"Predictions - Mean: {predictions.mean():.3f}, Std: {predictions.std():.3f}")
        print(f"Predictions - Range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        # Show sample values for first student
        print(f"Sample values for student 0:")
        for t in range(min(3, seq_len)):
            q_id = q_data[0, t].item()
            ability = student_abilities[0, t].item()
            difficulty = item_difficulties[0, t].item()
            discrimination = item_discriminations[0, t].item()
            pred = predictions[0, t].item()
            target = targets[0, t].item()
            
            print(f"  Q{q_id}: θ={ability:.3f}, β={difficulty:.3f}, α={discrimination:.3f} → P={pred:.3f} (target={target:.0f})")

def compare_with_without_discrimination():
    """Compare results with and without discrimination."""
    print(f"\n\n=== COMPARISON: WITH vs WITHOUT DISCRIMINATION ===")
    
    # Model parameters
    n_questions = 50
    memory_size = 15
    key_memory_state_dim = 32
    value_memory_state_dim = 32
    summary_vector_dim = 64
    
    # Create sample data
    q_data, qa_data, targets = create_sample_data(n_questions, batch_size=4, seq_len=5)
    
    # Model without discrimination (1PL)
    print("\n--- 1PL MODEL (No discrimination) ---")
    model_1pl = DeepIRTModel(
        n_questions=n_questions,
        memory_size=memory_size,
        key_memory_state_dim=key_memory_state_dim,
        value_memory_state_dim=value_memory_state_dim,
        summary_vector_dim=summary_vector_dim,
        use_discrimination=False,
        ability_scale=3.0
    )
    
    pred_1pl, ability_1pl, diff_1pl, disc_1pl, z_1pl = model_1pl(q_data, qa_data)
    loss_1pl = model_1pl.compute_loss(pred_1pl, targets)
    
    print(f"1PL Loss: {loss_1pl.item():.4f}")
    print(f"1PL Predictions range: [{pred_1pl.min():.3f}, {pred_1pl.max():.3f}]")
    print(f"1PL Z-values range: [{z_1pl.min():.3f}, {z_1pl.max():.3f}]")
    
    # Model with discrimination (2PL)
    print("\n--- 2PL MODEL (With static discrimination) ---")
    model_2pl = DeepIRTModel(
        n_questions=n_questions,
        memory_size=memory_size,
        key_memory_state_dim=key_memory_state_dim,
        value_memory_state_dim=value_memory_state_dim,
        summary_vector_dim=summary_vector_dim,
        use_discrimination=True,
        discrimination_type="static",
        ability_scale=3.0
    )
    
    pred_2pl, ability_2pl, diff_2pl, disc_2pl, z_2pl = model_2pl(q_data, qa_data)
    loss_2pl = model_2pl.compute_loss(pred_2pl, targets)
    
    print(f"2PL Loss: {loss_2pl.item():.4f}")
    print(f"2PL Predictions range: [{pred_2pl.min():.3f}, {pred_2pl.max():.3f}]")
    print(f"2PL Z-values range: [{z_2pl.min():.3f}, {z_2pl.max():.3f}]")
    print(f"2PL Discriminations range: [{disc_2pl.min():.3f}, {disc_2pl.max():.3f}]")
    
    # Compare the effect of discrimination
    print(f"\n--- COMPARISON SUMMARY ---")
    print(f"Loss improvement: {loss_1pl.item() - loss_2pl.item():.4f}")
    print(f"Z-value variance change: {z_1pl.var().item():.4f} → {z_2pl.var().item():.4f}")
    print(f"Prediction variance change: {pred_1pl.var().item():.4f} → {pred_2pl.var().item():.4f}")

if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducible results
    
    demonstrate_discrimination_types()
    compare_with_without_discrimination()
    
    print("\n\n=== USAGE NOTES ===")
    print("1. Static discrimination: a_j = softplus(W_a [k_t; v_t] + b_a)")
    print("   - Uses question embedding and question-answer embedding")
    print("   - More aligned with traditional IRT (item property)")
    print("   - Good for modeling item characteristics")
    print("")
    print("2. Dynamic discrimination: a_j = softplus(W_a f_t + b_a)")
    print("   - Uses summary vector (student knowledge state)")
    print("   - Allows discrimination to vary with student ability")
    print("   - Good for modeling adaptive difficulty")
    print("")
    print("3. Both (averaged): (static + dynamic) / 2")
    print("   - Combines both approaches")
    print("   - Balances item properties and student state")
    print("   - More flexible but potentially more complex")