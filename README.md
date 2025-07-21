# Deep-2PL: Deep Item Response Theory Model

A Deep-IRT implementation using Dynamic Key-Value Memory Networks (DKVMN) for knowledge tracing and educational data analysis.

## Quick Start

### Environment Setup
```bash
# Activate conda environment
conda activate vrec-env
```

### Unified Python Pipeline (Replaces Shell Scripts)
```bash
# Single dataset (replaces ./run_dataset.sh DATASET_NAME)
python main.py --datasets DATASET_NAME

# All datasets (replaces ./run_all_datasets.sh)
python main.py --all

# Quick test mode (replaces --quick flag)
python main.py --datasets synthetic --quick
```

### Examples

#### Single Dataset Training
```bash
# Single dataset with 5-fold cross-validation
python main.py --datasets STATICS

# Quick single fold training
python main.py --datasets synthetic --single_fold --fold 0

# Multiple specific datasets
python main.py --datasets synthetic assist2015 STATICS

# Training with visualization skipped
python main.py --datasets STATICS --skip_visualization
```

#### Multiple Datasets & Advanced Features
```bash
# All datasets with full pipeline
python main.py --all

# Filter by dataset size (replaces shell script --exclude/--only)
python main.py --all --exclude large     # Skip large datasets for faster execution
python main.py --all --only small        # Only small datasets for quick testing

# Time estimation (like shell scripts)
python main.py --all --time_estimate     # Show time estimates and exit
python main.py --all --exclude large --time_estimate  # Estimate excluding large datasets

# Custom training parameters for all datasets
python main.py --all --epochs 50 --batch_size 64 --learning_rate 0.01

# Generate comprehensive comparison across datasets
python main.py --comparison --skip_training --skip_evaluation --skip_statistics --skip_visualization

# Show training progress with dynamic IRT extraction
python main.py --datasets synthetic STATICS    # Shows real-time student/question discovery
```

#### Dynamic IRT Statistics Analysis (New!)
```bash
# Dynamic IRT parameter extraction (automatically adapts to any dataset)
python main.py --show_irt --datasets synthetic STATICS --skip_training --skip_evaluation --skip_statistics --skip_visualization

# Shows real-time discovery: "📊 Progress: 100.0% | Students: 800 | Questions: 50"
# Proper dimensions: Alpha/Beta per-question, Theta per-student

# Full IRT statistics with comprehensive data processing
python main.py --show_irt --irt_type full --datasets synthetic --single_fold --fold 0 --skip_training --skip_evaluation --skip_statistics --skip_visualization

# Combined training and IRT analysis (automatic extraction during training)
python main.py --datasets synthetic --show_irt
```

## Supported Datasets

### Pre-split (with Q-matrix)
- **STATICS** (956 questions, 98 KCs)
- **assist2009_updated** (70 questions)
- **fsaif1tof3** (1722 questions)

### Single-file (global mode)
- **assist2015** (96 questions)
- **assist2017** (102 questions)
- **synthetic** (50 questions)
- **assist2009** (124 questions)
- **statics2011** (1221 questions)
- **kddcup2010** (649 questions)

## Advanced Usage

### Main Pipeline Options
```bash
# Complete pipeline control
python main.py --all                                    # All datasets, full pipeline
python main.py --datasets STATICS assist2015           # Specific datasets
python main.py --datasets synthetic --single_fold --fold 0  # Single fold training

# Pipeline phase control
python main.py --datasets STATICS --skip_training       # Skip training phase
python main.py --datasets STATICS --skip_evaluation     # Skip evaluation phase
python main.py --datasets STATICS --skip_statistics     # Skip statistics extraction
python main.py --datasets STATICS --skip_visualization  # Skip visualization generation

# IRT analysis
python main.py --show_irt --datasets STATICS            # Show IRT statistics
python main.py --show_irt --irt_type full --datasets synthetic  # Full IRT analysis
```

### Direct Training (Lower Level)
```bash
# Basic training with 5-fold cross-validation
python train.py --dataset STATICS --epochs 20

# Single fold training
python train.py --dataset assist2015 --single_fold --fold 0

# Custom parameters
python train.py --dataset synthetic --epochs 10 --batch_size 64 --learning_rate 0.001

# Model architecture selection
python train.py --dataset STATICS --model_type optimized  # default
python train.py --dataset STATICS --model_type original
```

### Direct Evaluation
```bash
# Evaluate specific model
python evaluate.py --model_path save_models/best_model_STATICS_fold0.pth

# Evaluate on different splits
python evaluate.py --model_path save_models/best_model_assist2015_fold0.pth --split test
```

### Direct Visualization
```bash
# Generate model visualizations
python visualize.py --checkpoint save_models/best_model_STATICS_fold0.pth \
                   --config save_models/config_STATICS_fold0.json \
                   --output_dir figs/STATICS
```

## Command Line Options

### main.py Options (Complete Shell Script Replacement)

#### Dataset Selection
- `--all`: Run pipeline for all available datasets (replaces run_all_datasets.sh)
- `--datasets NAMES`: Specific datasets to process (replaces run_dataset.sh)

#### Dataset Filtering (New - from shell scripts)
- `--exclude CATEGORY`: Exclude dataset category: 'large', 'medium', 'small'
- `--only CATEGORY`: Include only dataset category: 'large', 'medium', 'small'

#### Training Control  
- `--single_fold`: Train only single fold instead of 5-fold CV
- `--fold N`: Fold index for single fold training (0-4, default: 0)
- `--epochs N`: Number of training epochs (overrides dataset defaults)
- `--batch_size N`: Batch size (overrides dataset defaults)
- `--learning_rate F`: Learning rate (overrides default 0.001)
- `--model_type TYPE`: Model architecture: 'original' or 'optimized' (default: optimized)

#### Advanced Options (New - from shell scripts)
- `--quick`: Quick mode: 1 epoch, single fold (for testing)
- `--time_estimate`: Show time estimates and exit
- `--comparison`: Generate comprehensive results comparison

#### Pipeline Control
- `--skip_training`: Skip training phase
- `--skip_evaluation`: Skip evaluation phase  
- `--skip_statistics`: Skip statistics extraction
- `--skip_visualization`: Skip visualization generation

#### Dynamic IRT Analysis (Enhanced)
- `--show_irt`: Display IRT statistics with automatic dataset adaptation
- `--irt_type TYPE`: 'fast' (dynamic data-based) or 'full' (comprehensive)

#### Features (Enhanced)
- **🚀 Shell Script Replacement**: Complete functionality of run_dataset.sh and run_all_datasets.sh
- **📊 Dynamic IRT Extraction**: Automatically adapts to any dataset size (no hardcoded limits)
- **⏱️ Smart Time Estimation**: Accurate training time predictions with category filtering
- **📈 Real-time Progress**: Live student/question discovery during IRT extraction
- **🔍 Comprehensive Comparison**: Multi-dataset performance analysis
- **📝 Enhanced Logging**: All operations logged with dataset characteristics
- **🔄 Platform Independent**: Works on Windows, Linux, macOS

### train.py Options (Direct Training)
- `--dataset`: Dataset name (required)
- `--fold`: Fold index for cross-validation (0-4)
- `--epochs`: Number of epochs (overrides default)
- `--batch_size`: Batch size (overrides default)
- `--learning_rate`: Learning rate (overrides default)
- `--model_type`: Model type: 'original' or 'optimized' (default: optimized)
- `--single_fold`: Train only specified fold instead of all 5 folds
- `--use_discrimination`: Use discrimination parameter (2PL model) - now default
- `--no_discrimination`: Disable discrimination parameter (use 1PL model)

### Dataset Categories
- **Small (fast)**: synthetic, assist2009_updated, assist2015, assist2017, assist2009
- **Medium**: kddcup2010, STATICS  
- **Large (slow)**: statics2011, fsaif1tof3

## Output Structure

```
logs/
└── pipeline_TIMESTAMP.log             # Comprehensive pipeline logs with progress tracking

results/
├── train/              # Training metrics and learning curves  
├── test/               # Model evaluation results
└── plots/              # Performance plots and ROC/PR curves

figs/
└── {dataset}/          # Dataset-specific visualizations
    ├── global_theta_heatmap.png
    ├── beta_distribution.png
    ├── alpha_distribution.png
    └── per_kc_*.png    # Per-KC analysis (for Q-matrix datasets)

save_models/
├── best_model_{dataset}_fold{N}.pth    # Best models from training
├── final_model_{dataset}_fold{N}.pth   # Final epoch models
└── config_{dataset}_fold{N}.json       # Model configurations

irt_stats/              # Dynamic IRT parameter extraction (ENHANCED)
├── fast_irt_{dataset}_fold{N}.pkl      # Dynamic data-based extraction (adapts to any dataset)
└── full_irt_{dataset}_fold{N}.pkl      # Comprehensive IRT statistics (full analysis)

stats/                  # Legacy IRT extraction
└── irt_params_{dataset}_fold{N}.pkl    # Backward compatibility
```

## Model Architecture

### Core Components
- **DKVMN**: Dynamic Key-Value Memory Networks for sequence modeling
- **IRT Integration**: Traditional Item Response Theory parameter extraction
- **Embedding Layers**: Question and question-answer embeddings
- **Memory Operations**: Attention-based read/write operations

### Model Types
- **Optimized** (default): Enhanced architecture with improved computational efficiency
- **Original**: Base implementation for compatibility and comparison

### Operating Modes
- **Per-KC Mode**: Automatic activation when Q-matrix is available, enables knowledge component tracking
- **Global Mode**: Single memory bank for datasets without Q-matrix

## Data Requirements

### Dataset Structure
```
data/
└── {dataset_name}/
    ├── {dataset_name}_train.txt
    ├── {dataset_name}_test.txt
    └── Qmatrix.csv (optional, enables per-KC mode)
```

### Data Format
- Tab-separated values
- Columns: user_id, item_id, timestamp, correct, ...
- Q-matrix: CSV format with binary KC associations

## Dependencies

```bash
# Core requirements
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
```

## Citation

```bibtex
@article{yeung2019addressing,
  title={Addressing two problems in deep knowledge tracing via prediction-consistent regularization},
  author={Yeung, Chun-Kit and Yeung, Dit-Yan},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2019}
}
```