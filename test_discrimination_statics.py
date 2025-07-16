#!/usr/bin/env python3
"""
Test script for discrimination functionality with STATICS dataset.
Tests both static and dynamic discrimination types.
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def setup_directories():
    """Create directory structure for different discrimination types."""
    base_dirs = [
        "figs/STATICS_static",
        "figs/STATICS_dynamic", 
        "figs/STATICS_both",
        "checkpoints_static",
        "checkpoints_dynamic",
        "checkpoints_both",
        "logs_static",
        "logs_dynamic", 
        "logs_both"
    ]
    
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def train_model(discrimination_type, epochs=10):
    """Train a model with specified discrimination type."""
    save_dir = f"checkpoints_{discrimination_type}"
    log_dir = f"logs_{discrimination_type}"
    
    cmd = [
        sys.executable, "train.py",
        "--dataset", "STATICS",
        "--data_style", "yeung",
        "--fold", "0",
        "--epochs", str(epochs),
        "--use_discrimination",
        "--discrimination_type", discrimination_type
    ]
    
    # Set environment variables to use custom directories
    env = os.environ.copy()
    env["SAVE_DIR"] = save_dir
    env["LOG_DIR"] = log_dir
    
    print(f"\n=== Training {discrimination_type} discrimination model ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"Save directory: {save_dir}")
    print(f"Log directory: {log_dir}")
    
    result = subprocess.run(cmd, env=env)
    return result.returncode == 0

def create_visualizations(discrimination_type):
    """Create visualizations for a specific discrimination type."""
    checkpoint_path = f"checkpoints_yeung/best_model_STATICS.pth"
    config_path = f"checkpoints_yeung/config_STATICS.json"
    output_dir = f"figs/STATICS_{discrimination_type}"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    cmd = [
        sys.executable, "visualize.py",
        "--checkpoint", checkpoint_path,
        "--config", config_path,
        "--output_dir", output_dir,
        "--student_idx", "0",
        "--save_data", f"saved_data_STATICS_{discrimination_type}.pkl"
    ]
    
    print(f"\n=== Creating visualizations for {discrimination_type} ===")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def compare_results():
    """Compare results between different discrimination types."""
    print("\n=== Comparing Results ===")
    
    # Check which visualizations were created
    discrimination_types = ["static", "dynamic", "both"]
    results = {}
    
    for disc_type in discrimination_types:
        output_dir = f"figs/STATICS_{disc_type}"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            results[disc_type] = files
            print(f"\n{disc_type} discrimination results:")
            for file in files:
                print(f"  - {file}")
        else:
            print(f"\n{disc_type} discrimination: No results directory found")
    
    return results

def test_discrimination_functionality():
    """Test discrimination functionality with quick training."""
    print("=== Testing Discrimination Functionality ===")
    
    # Test with very short training for functionality check
    test_epochs = 2
    
    # Test static discrimination
    print("\n--- Testing Static Discrimination ---")
    if train_model("static", test_epochs):
        print("✓ Static discrimination training successful")
        if create_visualizations("static"):
            print("✓ Static discrimination visualization successful")
        else:
            print("✗ Static discrimination visualization failed")
    else:
        print("✗ Static discrimination training failed")
    
    # Test dynamic discrimination
    print("\n--- Testing Dynamic Discrimination ---")
    if train_model("dynamic", test_epochs):
        print("✓ Dynamic discrimination training successful")
        if create_visualizations("dynamic"):
            print("✓ Dynamic discrimination visualization successful")
        else:
            print("✗ Dynamic discrimination visualization failed")
    else:
        print("✗ Dynamic discrimination training failed")

def main():
    """Main function to run all tests."""
    print("=== Deep-2PL Discrimination Test Suite ===")
    print(f"Started at: {datetime.now()}")
    
    # Setup directories
    setup_directories()
    
    # Test basic functionality
    test_discrimination_functionality()
    
    # Compare results
    compare_results()
    
    print(f"\n=== Test Suite Completed ===")
    print(f"Finished at: {datetime.now()}")
    
    print("\n=== Summary ===")
    print("Directories created:")
    print("- figs/STATICS_static/")
    print("- figs/STATICS_dynamic/")
    print("- figs/STATICS_both/")
    print("- checkpoints_static/")
    print("- checkpoints_dynamic/")
    print("- checkpoints_both/")
    
    print("\nFiles to look for:")
    print("- discrimination_distribution.png (in figs folders)")
    print("- beta_distribution.png")
    print("- global_theta_heatmap.png")
    print("- saved_data_STATICS_*.pkl")
    
    print("\nNext steps:")
    print("1. Run with longer training: python test_discrimination_statics.py --full-training")
    print("2. Compare static vs dynamic discrimination plots")
    print("3. Check model performance differences")

if __name__ == "__main__":
    main()