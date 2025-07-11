# Deep-2PL: Extended Knowledge Tracing with 2-Parameter Logistic Model

This implementation extends knowledge tracing models to support both Yeung-style pre-split data and dkvmn-torch style runtime k-fold cross-validation.

## Data Organization

### Two Data Loading Styles Supported:

1. **Yeung Style** (`data-yeung/`): Uses pre-computed train/validation splits
   - Format: `dataset_train0.csv`, `dataset_valid0.csv`, etc.
   - Example: `assist2009_updated_train0.csv`, `assist2009_updated_valid0.csv`
   - Benefits: Consistent splits across runs, reproducible results

2. **Torch Style** (`data-orig/`): Runtime k-fold cross-validation  
   - Format: `dataset_train.txt`, `dataset_test.txt`
   - Example: `assist2015_train.txt`, `assist2015_test.txt`
   - Benefits: Flexible fold generation, standard format

## Usage Examples

### 1. Using Yeung Style (Pre-split data)
```bash
# Train with assist2009 using fold 0
python train.py --config config_yeung_assist2009.json

# Train with assist2009 using fold 2
python train.py --config config_yeung_assist2009.json --fold_idx 2

# Train with different dataset
python train.py --data_style yeung --dataset_name assist2015 --data_dir ./data-yeung --fold_idx 1
```

### 2. Using Torch Style (Runtime k-fold)
```bash
# Train with assist2015 using fold 0
python train.py --config config_torch_assist2015.json

# Train with assist2015 using fold 3
python train.py --config config_torch_assist2015.json --fold_idx 3

# Train with different k-fold setting
python train.py --data_style torch --dataset_name assist2015 --k_fold 10 --fold_idx 5
```

### 3. Command Line Arguments
```bash
# Full command line example for Yeung style
python train.py \
    --data_style yeung \
    --data_dir ./data-yeung \
    --dataset_name assist2009_updated \
    --fold_idx 0 \
    --n_questions 110 \
    --batch_size 32 \
    --n_epochs 100 \
    --use_discrimination

# Full command line example for Torch style  
python train.py \
    --data_style torch \
    --data_dir ./data-orig \
    --dataset_name assist2015 \
    --k_fold 5 \
    --fold_idx 0 \
    --n_questions 100 \
    --batch_size 32 \
    --n_epochs 100
```

## Configuration Parameters

### Data Loading Parameters:
- `data_style`: "yeung" or "torch" 
- `data_dir`: Path to data directory ("./data-yeung" or "./data-orig")
- `dataset_name`: Dataset name (e.g., "assist2009_updated", "assist2015")
- `k_fold`: Number of folds for cross-validation (default: 5)
- `fold_idx`: Which fold to use (0 to k_fold-1)

### Model Parameters:
- `n_questions`: Number of unique questions in dataset
- `memory_size`: DKVMN memory size
- `use_discrimination`: Enable 2PL discrimination parameter
- `ability_scale`: IRT ability scaling factor

## Data Directory Structure

```
deep-2pl/
├── data-yeung/          # Yeung style pre-split data
│   ├── assist2009_updated/
│   │   ├── assist2009_updated_train0.csv
│   │   ├── assist2009_updated_valid0.csv
│   │   ├── assist2009_updated_train1.csv
│   │   ├── assist2009_updated_valid1.csv
│   │   ├── ...
│   │   └── assist2009_updated_test.csv
│   └── assist2015/
│       └── ...
└── data-orig/           # Torch style data
    ├── assist2015/
    │   ├── assist2015_train.txt
    │   └── assist2015_test.txt
    └── assist2009/
        ├── builder_train.csv
        └── builder_test.csv
```

## Cross-Validation Workflow

### Yeung Style (Recommended for reproducible research):
```bash
# Run all 5 folds for assist2009
for fold in {0..4}; do
    python train.py --config config_yeung_assist2009.json --fold_idx $fold
done
```

### Torch Style:
```bash
# Run all 5 folds for assist2015  
for fold in {0..4}; do
    python train.py --config config_torch_assist2015.json --fold_idx $fold
done
```

## Output

Each training run creates:
- Model checkpoints in `save_dir`
- Training logs in `log_dir` 
- TensorBoard logs (if enabled)
- Best model based on validation accuracy

The system automatically handles:
- Train/validation/test splits
- Cross-validation folds
- Data format differences
- Sequence padding and encoding