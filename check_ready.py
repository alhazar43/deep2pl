#!/usr/bin/env python3
"""
Quick check to verify everything is ready for comprehensive training.
"""

import torch
import os
import sys
from data.dataloader import get_dataset_info

def check_cuda():
    """Check CUDA availability."""
    print("🔍 Checking CUDA setup...")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return torch.cuda.is_available()

def check_datasets():
    """Check available datasets."""
    print("\n📊 Checking available datasets...")
    
    datasets = [
        'STATICS', 'assist2009_updated', 'fsaif1tof3',
        'assist2015', 'assist2017', 'synthetic', 
        'assist2009', 'statics2011', 'kddcup2010'
    ]
    
    available = []
    for dataset in datasets:
        try:
            info = get_dataset_info(dataset)
            print(f"  ✓ {dataset:<18}: {info['n_questions']} questions")
            available.append(dataset)
        except Exception as e:
            print(f"  ✗ {dataset:<18}: {str(e)}")
    
    print(f"\n  Available: {len(available)}/{len(datasets)} datasets")
    return available

def check_model():
    """Check model can be initialized."""
    print("\n🧠 Checking model initialization...")
    
    try:
        from models.model import DeepIRTModel
        
        model = DeepIRTModel(
            n_questions=100,
            memory_size=20,
            key_dim=32,
            value_dim=64,
            final_fc_dim=32
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            
        # Test forward pass
        batch_size = 4
        seq_len = 5
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        q_data = torch.randint(1, 101, (batch_size, seq_len)).to(device)
        qa_data = torch.randint(1, 201, (batch_size, seq_len)).to(device)
        
        predictions, _, _, _, _ = model(q_data, qa_data)
        print(f"  ✓ Model forward pass successful")
        print(f"  ✓ Output shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model test failed: {str(e)}")
        return False

def estimate_training_time(datasets):
    """Estimate total training time."""
    print(f"\n⏱️  Training time estimation...")
    
    # Based on synthetic dataset: 1 epoch ≈ 9 seconds on GPU
    # Scale by dataset size (rough estimate)
    
    dataset_sizes = {
        'synthetic': 1.0,      # baseline
        'assist2009_updated': 1.4,  # 70 vs 50 questions
        'assist2015': 1.9,     # 96 questions
        'assist2017': 2.0,     # 102 questions
        'assist2009': 2.5,     # 124 questions
        'kddcup2010': 13.0,    # 649 questions
        'STATICS': 19.0,       # 956 questions
        'statics2011': 24.0,   # 1221 questions
        'fsaif1tof3': 34.0,    # 1722 questions
    }
    
    base_epoch_time = 9  # seconds for synthetic
    epochs = 30
    
    total_time = 0
    print(f"  Estimates for 30 epochs:")
    
    for dataset in datasets:
        if dataset in dataset_sizes:
            dataset_time = base_epoch_time * dataset_sizes[dataset] * epochs / 60  # minutes
            total_time += dataset_time
            print(f"    {dataset:<18}: ~{dataset_time:.1f} minutes")
    
    print(f"\n  Total estimated time: ~{total_time/60:.1f} hours")
    print(f"  (This is a rough estimate - actual time may vary)")
    
    return total_time

def main():
    """Main check function."""
    print("🚀 Deep-IRT Training Readiness Check")
    print("=" * 50)
    
    # Run checks
    cuda_ok = check_cuda()
    available_datasets = check_datasets()
    model_ok = check_model()
    
    if available_datasets:
        estimate_training_time(available_datasets)
    
    # Overall status
    print(f"\n{'='*50}")
    print("📋 SUMMARY")
    print(f"{'='*50}")
    
    print(f"CUDA: {'✓' if cuda_ok else '✗'}")
    print(f"Datasets: {len(available_datasets)}/9 available")
    print(f"Model: {'✓' if model_ok else '✗'}")
    
    all_ready = cuda_ok and len(available_datasets) >= 5 and model_ok
    
    if all_ready:
        print(f"\n🎉 System ready for training!")
        print(f"\nTo start training all datasets:")
        print(f"  python train_all_datasets.py")
        print(f"\nTo train specific datasets:")
        print(f"  python train_all_datasets.py --datasets synthetic assist2015")
        print(f"\nTo exclude large datasets:")
        print(f"  python train_all_datasets.py --exclude statics2011 fsaif1tof3")
    else:
        print(f"\n⚠️  Some issues detected. Please fix before training.")

if __name__ == "__main__":
    main()