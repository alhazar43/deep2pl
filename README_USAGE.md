# Deep-2PL: Unified Knowledge Tracing System

## Quick Start

### 🚀 Ready-to-Run Scripts

#### Yeung Style (Pre-split datasets)
```bash
# Train on assist2009_updated with fold 0
python run_yeung_style.py --dataset assist2009_updated --fold 0

# Train with custom settings
python run_yeung_style.py --dataset assist2015 --fold 2 --epochs 10 --batch_size 16
```

#### Torch Style (Runtime k-fold)
```bash
# Train on assist2015 with fold 0
python run_torch_style.py --dataset assist2015 --fold 0

# Train with custom settings  
python run_torch_style.py --dataset kddcup2010 --fold 1 --epochs 15 --batch_size 8
```

## 📊 Results Summary

### 1. Torch Style Test Results
- **Dataset**: assist2015 (100 questions)
- **Data loaded**: 12,697 train / 3,175 valid / 3,968 test samples
- **Model**: 2PL (with discrimination parameter)
- **Data ranges**: Q: 0-77, QA: 0-177 ✅
- **Forward pass**: Working correctly
- **Initial loss**: 1.5756

### 2. Yeung Style Test Results  
- **Dataset**: assist2009_updated (111 questions)
- **Model**: 1PL (no discrimination)
- **Training progress**: Loss 0.7164 → 0.4988, Accuracy 58.13% → 74.33%
- **Status**: ✅ Successfully completed training

## 📂 Data Organization

```
deep-2pl/
├── data-yeung/          # Pre-split datasets (Yeung style)
│   ├── assist2009_updated/
│   ├── assist2015/
│   └── synthetic/
├── data-orig/           # Single file datasets (Torch style)
│   ├── assist2015/
│   ├── assist2017/
│   └── kddcup2010/
└── ...
```

## 🎯 Available Datasets

### Yeung Style
- `assist2009_updated` (111 questions, 1PL)
- `assist2015` (100 questions, 1PL)  
- `synthetic` (50 questions, 1PL)
- `STATICS` (1PL)
- `fsaif1tof3` (1PL)

### Torch Style
- `assist2015` (100 questions, 2PL)
- `assist2017` (102 questions, 2PL)
- `kddcup2010` (661 questions, 1PL)
- `statics2011` (1223 questions, 1PL)
- `synthetic` (50 questions, 1PL)

## 🔧 Enhanced Features

### Progress Bars
- 🚀 **Training**: Real-time loss, accuracy, learning rate, and batch progress
- 📊 **Evaluation**: Separate progress bars for validation and test phases
- 🎨 **Visual**: Color-coded progress bars with emojis for better UX

### Configuration
- **Automatic**: Dataset-specific configurations (batch size, epochs, model type)
- **Flexible**: Command-line overrides for quick experiments
- **Safe**: Embedding bounds checking and value clamping

## 🏃‍♂️ Example Usage

```bash
# Quick 5-epoch test on assist2009
python run_yeung_style.py --dataset assist2009_updated --epochs 5

# Full training on large dataset
python run_torch_style.py --dataset kddcup2010 --epochs 50 --batch_size 8

# Cross-validation on all folds
for fold in {0..4}; do
    python run_yeung_style.py --dataset assist2015 --fold $fold --epochs 20
done
```

## 🔍 Key Differences

| Feature | Yeung Style | Torch Style |
|---------|-------------|-------------|
| **Data Split** | Pre-computed (train0.csv, valid0.csv) | Runtime k-fold |
| **File Format** | CSV | TXT |
| **Model Type** | Usually 1PL | Usually 2PL |
| **Use Case** | Reproducible splits | Flexible cross-validation |

Both styles now work seamlessly with the unified dataloader system!