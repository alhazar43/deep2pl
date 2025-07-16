#!/usr/bin/env python3
"""
Test script for the newly implemented item discrimination networks.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.irt import DeepIRTModel, ItemDiscriminationStaticNetwork, ItemDiscriminationDynamicNetwork

def test_discrimination_networks():
    """Test the discrimination networks independently."""
    print("Testing discrimination networks...")
    
    # Test parameters
    batch_size = 4
    q_embed_dim = 32
    qa_embed_dim = 32
    summary_vector_dim = 64
    
    # Test static discrimination network
    print("\n1. Testing ItemDiscriminationStaticNetwork...")
    static_net = ItemDiscriminationStaticNetwork(q_embed_dim, qa_embed_dim)
    
    q_embed = torch.randn(batch_size, q_embed_dim)
    qa_embed = torch.randn(batch_size, qa_embed_dim)
    
    static_disc = static_net(q_embed, qa_embed)
    print(f"   Static discrimination shape: {static_disc.shape}")
    print(f"   Static discrimination values: {static_disc.squeeze()}")
    print(f"   All positive (due to softplus): {torch.all(static_disc > 0)}")
    
    # Test dynamic discrimination network
    print("\n2. Testing ItemDiscriminationDynamicNetwork...")
    dynamic_net = ItemDiscriminationDynamicNetwork(summary_vector_dim)
    
    summary_vector = torch.randn(batch_size, summary_vector_dim)
    
    dynamic_disc = dynamic_net(summary_vector)
    print(f"   Dynamic discrimination shape: {dynamic_disc.shape}")
    print(f"   Dynamic discrimination values: {dynamic_disc.squeeze()}")
    print(f"   All positive (due to softplus): {torch.all(dynamic_disc > 0)}")


def test_deep_irt_model():
    """Test the DeepIRTModel with discrimination enabled."""
    print("\n\nTesting DeepIRTModel with discrimination...")
    
    # Model parameters
    n_questions = 100
    memory_size = 10
    key_memory_state_dim = 32
    value_memory_state_dim = 32
    summary_vector_dim = 64
    batch_size = 4
    seq_len = 5
    
    # Test all discrimination types
    discrimination_types = ["static", "dynamic", "both"]
    
    for disc_type in discrimination_types:
        print(f"\n--- Testing with discrimination_type='{disc_type}' ---")
        
        # Create model
        model = DeepIRTModel(
            n_questions=n_questions,
            memory_size=memory_size,
            key_memory_state_dim=key_memory_state_dim,
            value_memory_state_dim=value_memory_state_dim,
            summary_vector_dim=summary_vector_dim,
            use_discrimination=True,
            discrimination_type=disc_type
        )
        
        # Create sample data
        q_data = torch.randint(1, n_questions + 1, (batch_size, seq_len))
        qa_data = torch.randint(1, 2 * n_questions + 1, (batch_size, seq_len))
        
        # Forward pass
        predictions, student_abilities, item_difficulties, item_discriminations, z_values = model(q_data, qa_data)
        
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Student abilities shape: {student_abilities.shape}")
        print(f"   Item difficulties shape: {item_difficulties.shape}")
        print(f"   Item discriminations shape: {item_discriminations.shape}")
        print(f"   Z values shape: {z_values.shape}")
        
        # Check discrimination values
        print(f"   Sample discrimination values: {item_discriminations[0, :3]}")
        print(f"   All discriminations positive: {torch.all(item_discriminations > 0)}")
        print(f"   Min discrimination: {item_discriminations.min():.4f}")
        print(f"   Max discrimination: {item_discriminations.max():.4f}")
        
        # Test loss computation
        targets = torch.randint(0, 2, (batch_size, seq_len)).float()
        loss = model.compute_loss(predictions, targets)
        print(f"   Loss: {loss.item():.4f}")


def test_backward_compatibility():
    """Test that the model still works without discrimination."""
    print("\n\nTesting backward compatibility (no discrimination)...")
    
    # Model parameters
    n_questions = 100
    memory_size = 10
    key_memory_state_dim = 32
    value_memory_state_dim = 32
    summary_vector_dim = 64
    batch_size = 4
    seq_len = 5
    
    # Create model without discrimination
    model = DeepIRTModel(
        n_questions=n_questions,
        memory_size=memory_size,
        key_memory_state_dim=key_memory_state_dim,
        value_memory_state_dim=value_memory_state_dim,
        summary_vector_dim=summary_vector_dim,
        use_discrimination=False
    )
    
    # Create sample data
    q_data = torch.randint(1, n_questions + 1, (batch_size, seq_len))
    qa_data = torch.randint(1, 2 * n_questions + 1, (batch_size, seq_len))
    
    # Forward pass
    predictions, student_abilities, item_difficulties, item_discriminations, z_values = model(q_data, qa_data)
    
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Student abilities shape: {student_abilities.shape}")
    print(f"   Item difficulties shape: {item_difficulties.shape}")
    print(f"   Item discriminations: {item_discriminations} (should be None)")
    print(f"   Z values shape: {z_values.shape}")
    
    # Test loss computation
    targets = torch.randint(0, 2, (batch_size, seq_len)).float()
    loss = model.compute_loss(predictions, targets)
    print(f"   Loss: {loss.item():.4f}")


if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducible results
    
    test_discrimination_networks()
    test_deep_irt_model()
    test_backward_compatibility()
    
    print("\n\nAll tests completed successfully! 🎉")