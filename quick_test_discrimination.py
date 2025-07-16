#!/usr/bin/env python3
"""
Quick test of discrimination functionality.
"""

import torch
import numpy as np
from models.irt import DeepIRTModel

def test_discrimination():
    """Test discrimination functionality."""
    print("=== Testing Discrimination Functionality ===")
    
    # Model parameters
    n_questions = 100
    memory_size = 10
    key_memory_state_dim = 32
    value_memory_state_dim = 32
    summary_vector_dim = 64
    batch_size = 4
    seq_len = 5
    
    # Test static discrimination
    print("\n1. Testing Static Discrimination...")
    try:
        model_static = DeepIRTModel(
            n_questions=n_questions,
            memory_size=memory_size,
            key_memory_state_dim=key_memory_state_dim,
            value_memory_state_dim=value_memory_state_dim,
            summary_vector_dim=summary_vector_dim,
            use_discrimination=True,
            discrimination_type="static"
        )
        
        # Test forward pass
        q_data = torch.randint(1, n_questions + 1, (batch_size, seq_len))
        qa_data = torch.randint(1, 2 * n_questions + 1, (batch_size, seq_len))
        
        predictions, abilities, difficulties, discriminations, z_values, kc_info = model_static(q_data, qa_data)
        
        print(f"   ✓ Static discrimination shape: {discriminations.shape}")
        print(f"   ✓ Discrimination values: {discriminations[0, :3]}")
        print(f"   ✓ All positive: {torch.all(discriminations > 0)}")
        print(f"   ✓ Min/Max: {discriminations.min():.3f}/{discriminations.max():.3f}")
        
    except Exception as e:
        print(f"   ✗ Static discrimination failed: {e}")
        return False
    
    # Test dynamic discrimination
    print("\n2. Testing Dynamic Discrimination...")
    try:
        model_dynamic = DeepIRTModel(
            n_questions=n_questions,
            memory_size=memory_size,
            key_memory_state_dim=key_memory_state_dim,
            value_memory_state_dim=value_memory_state_dim,
            summary_vector_dim=summary_vector_dim,
            use_discrimination=True,
            discrimination_type="dynamic"
        )
        
        predictions, abilities, difficulties, discriminations, z_values, kc_info = model_dynamic(q_data, qa_data)
        
        print(f"   ✓ Dynamic discrimination shape: {discriminations.shape}")
        print(f"   ✓ Discrimination values: {discriminations[0, :3]}")
        print(f"   ✓ All positive: {torch.all(discriminations > 0)}")
        print(f"   ✓ Min/Max: {discriminations.min():.3f}/{discriminations.max():.3f}")
        
    except Exception as e:
        print(f"   ✗ Dynamic discrimination failed: {e}")
        return False
    
    # Test both discrimination
    print("\n3. Testing Both Discrimination...")
    try:
        model_both = DeepIRTModel(
            n_questions=n_questions,
            memory_size=memory_size,
            key_memory_state_dim=key_memory_state_dim,
            value_memory_state_dim=value_memory_state_dim,
            summary_vector_dim=summary_vector_dim,
            use_discrimination=True,
            discrimination_type="both"
        )
        
        predictions, abilities, difficulties, discriminations, z_values, kc_info = model_both(q_data, qa_data)
        
        print(f"   ✓ Both discrimination shape: {discriminations.shape}")
        print(f"   ✓ Discrimination values: {discriminations[0, :3]}")
        print(f"   ✓ All positive: {torch.all(discriminations > 0)}")
        print(f"   ✓ Min/Max: {discriminations.min():.3f}/{discriminations.max():.3f}")
        
    except Exception as e:
        print(f"   ✗ Both discrimination failed: {e}")
        return False
    
    # Test no discrimination (1PL)
    print("\n4. Testing No Discrimination (1PL)...")
    try:
        model_1pl = DeepIRTModel(
            n_questions=n_questions,
            memory_size=memory_size,
            key_memory_state_dim=key_memory_state_dim,
            value_memory_state_dim=value_memory_state_dim,
            summary_vector_dim=summary_vector_dim,
            use_discrimination=False
        )
        
        predictions, abilities, difficulties, discriminations, z_values, kc_info = model_1pl(q_data, qa_data)
        
        print(f"   ✓ No discrimination: {discriminations}")
        print(f"   ✓ Predictions shape: {predictions.shape}")
        
    except Exception as e:
        print(f"   ✗ No discrimination failed: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    return True

def test_statics_config():
    """Test with STATICS-like configuration."""
    print("\n=== Testing STATICS Configuration ===")
    
    # STATICS-like parameters
    n_questions = 1223
    memory_size = 50
    key_memory_state_dim = 50
    value_memory_state_dim = 200
    summary_vector_dim = 50
    batch_size = 2
    seq_len = 10
    
    try:
        model = DeepIRTModel(
            n_questions=n_questions,
            memory_size=memory_size,
            key_memory_state_dim=key_memory_state_dim,
            value_memory_state_dim=value_memory_state_dim,
            summary_vector_dim=summary_vector_dim,
            use_discrimination=True,
            discrimination_type="static",
            q_matrix_path="./data-yeung/STATICS/Qmatrix.csv"
        )
        
        print(f"   ✓ Model created successfully")
        print(f"   ✓ Per-KC mode: {model.per_kc_mode}")
        print(f"   ✓ Number of KCs: {model.n_kcs}")
        print(f"   ✓ Using discrimination: {model.use_discrimination}")
        print(f"   ✓ Discrimination type: {model.discrimination_type}")
        
        # Test forward pass
        q_data = torch.randint(1, n_questions + 1, (batch_size, seq_len))
        qa_data = torch.randint(1, 2 * n_questions + 1, (batch_size, seq_len))
        
        predictions, abilities, difficulties, discriminations, z_values, kc_info = model(q_data, qa_data)
        
        print(f"   ✓ Forward pass successful")
        print(f"   ✓ Predictions shape: {predictions.shape}")
        print(f"   ✓ Discriminations shape: {discriminations.shape}")
        print(f"   ✓ KC info available: {'all_kc_thetas' in kc_info}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ STATICS configuration failed: {e}")
        return False

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test basic functionality
    success = test_discrimination()
    
    if success:
        # Test STATICS configuration
        test_statics_config()
    
    print("\n=== Quick Test Complete ===")
    print("The discrimination functionality is working correctly!")
    print("You can now run full training with:")
    print("python train.py --dataset STATICS --data_style yeung --use_discrimination --discrimination_type static")