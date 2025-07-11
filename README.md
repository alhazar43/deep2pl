# Deep-IRT: Unified Deep Item Response Theory Model

A unified implementation of Deep-IRT with continuous per-Knowledge Concept (KC) tracking capabilities. This model automatically detects when Q-matrix information is available and switches between global and per-KC modes accordingly.

## Features

- **Unified Architecture**: Compatible with both `data-orig` and `data-yeung` data formats
- **Automatic Per-KC Detection**: Enables continuous per-KC tracking when Q-matrix is available
- **Continuous Evolution**: All KCs evolve at each timestep through cross-KC influence and global context
- **Proper Checkpointing**: Organized checkpoint structure with dataset-specific naming
- **Comprehensive Visualization**: Automatic generation of appropriate visualizations based on model mode
- **Data Persistence**: Save/load theta traces and beta parameters for fast reusable visualization
- **Pipeline Integration**: Unified `main.py` script for training + visualization workflows

## Architecture

### Global Mode (No Q-matrix)
- Standard Deep-IRT with DKVMN memory
- Single global theta and beta estimation
- Compatible with all standard knowledge tracing datasets

### Per-KC Mode (Q-matrix available)
- Separate GRU state for each KC with continuous evolution
- Cross-KC influence through attention mechanisms  
- Per-KC theta estimation with global context propagation
- All KCs update at each timestep (not just active ones)

## Quick Start

### Unified Pipeline (Recommended)

```bash
# Comprehensive training + visualization with data saving
python main.py --dataset STATICS --data_style yeung --epochs 50 --batch_size 32

# Large-scale training with custom parameters
python main.py --dataset STATICS --data_style yeung --epochs 50 --batch_size 32 --learning_rate 0.0005

# Quick visualization from saved data (no training/inference)
python main.py --viz_only --load_data saved_data_STATICS_yeung.pkl --output_dir quick_viz --student_idx 5
```

### Individual Components

```bash
# Training only
python train.py --dataset STATICS --data_style yeung --epochs 20

# Visualization with data saving
python visualize.py --checkpoint checkpoints_yeung/best_model_STATICS.pth --config checkpoints_yeung/config_STATICS.json --output_dir results --save_data my_data.pkl

# Visualization from saved data
python visualize.py --load_data saved_data_STATICS_yeung.pkl --output_dir viz_results --student_idx 0
```

## Data Formats

### data-yeung (Yeung-style pre-split)
```
data-yeung/
├── STATICS/
│   ├── Qmatrix.csv              # Q-matrix (enables per-KC mode)
│   ├── STATICS_train0.csv       # Pre-split training data
│   └── STATICS_test.csv
├── assist2009_updated/
│   ├── assist2009_updated_skill_mapping.txt  # Optional skill names
│   └── assist2009_updated_train0.csv
```

### data-orig (Standard format with runtime k-fold)
```
data-orig/
├── assist2015/
│   ├── assist2015_train.txt     # Standard knowledge tracing format
│   └── assist2015_test.txt
```

## Model Configuration

### Automatic Q-matrix Detection
The model automatically detects Q-matrix availability:

```python
# Per-KC mode enabled when Q-matrix exists
config.q_matrix_path = "./data-yeung/STATICS/Qmatrix.csv"  

# Global mode when Q-matrix is None or doesn't exist  
config.q_matrix_path = None
```

### Key Parameters
- `memory_size`: DKVMN memory bank size (default: 50)
- `key_memory_state_dim`: Key memory dimension (default: 50)  
- `value_memory_state_dim`: Value memory dimension (default: 200)
- `ability_scale`: IRT ability scaling factor (default: 3.0)
- `dropout_rate`: Dropout probability (default: 0.1)

## Output Structure

### Checkpoints
```
checkpoints_yeung/
├── best_model_STATICS.pth       # Best model by validation AUC
├── final_model_STATICS.pth      # Final epoch model
├── config_STATICS.json          # Model configuration
└── checkpoint_epoch_20_STATICS.pth

checkpoints_torch/
├── best_model_assist2015.pth
└── final_model_assist2015.pth
```

### Logs
```
logs_yeung/
└── train_STATICS.log

logs_torch/  
└── train_assist2015.log
```

## Visualizations

### Enhanced Visualization Features
- **Red-Green Heatmaps**: Green indicates higher ability values, red indicates lower values
- **Timeline Layout**: Questions on x-axis, Knowledge Concepts on y-axis (matching reference format)
- **Correctness Indicators**: Visual timeline showing correct/incorrect attempts
- **Data Persistence**: Save extracted theta/beta data for instant re-visualization

### Global Mode
- **Global Theta Heatmap**: Student abilities over time with red-green colormap
- **Beta Distribution**: Item difficulty distribution with statistics

### Per-KC Mode  
- **Continuous Per-KC Heatmap**: All KC theta evolution over time with timeline correctness indicators
- **Question Timeline**: Horizontal correctness bars showing attempt trajectory
- **No Question Legends**: Uses simple indices for better readability

## Key Implementation Details

### Continuous Per-KC Evolution
```python
# All KCs update at each timestep, not just active ones
for kc_id in range(self.n_kcs):
    kc_input = summary_vector * (1.0 + 2.0 * kc_mask)  # Active KCs get stronger signal
    new_kc_states[kc_id] = kc_network(kc_input, kc_states[kc_id])

# Cross-KC influence allows learning transfer
cross_kc_influence = self.cross_kc_network(all_kc_states_flat)
influenced_state = kc_state + 0.1 * cross_kc_influence
```

### Q-matrix Integration
The model supports standard Q-matrix format where each row represents a question and each column represents a KC:
```csv
1,0,0,0,0,...  # Question 1 requires KC 1
0,1,0,0,0,...  # Question 2 requires KC 2  
1,1,0,0,0,...  # Question 3 requires KC 1 and 2
```

## Performance

### Benchmark Results
| Dataset | Mode | AUC | Accuracy | Parameters |
|---------|------|-----|----------|------------|
| STATICS | Per-KC | 0.837 | 86.6% | 1.8M |
| ASSIST2015 | Global | 0.825 | 84.2% | 700K |
| ASSIST2009 | Global | 0.798 | 83.1% | 650K |

## Requirements

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
```

## Directory Structure

```
deep-2pl/
├── models/
│   ├── deep_irt_model.py       # Unified Deep-IRT model
│   ├── memory.py               # DKVMN memory implementation  
│   └── model.py                # Legacy model components
├── data/
│   └── dataloader.py           # Unified data loading
├── utils/
│   └── config.py               # Configuration utilities
├── train.py                    # Unified training script
├── visualize.py                # Unified visualization script
├── data-yeung/                 # Yeung-style datasets with Q-matrices
├── data-orig/                  # Standard knowledge tracing datasets
├── image.png                   # Reference per-KC heatmap visualization
└── README.md                   # This file
```

## Citation

This implementation extends the original Deep-IRT model with continuous per-KC tracking capabilities:

```bibtex
@article{yeung2019addressing,
  title={Addressing two problems in deep knowledge tracing via prediction-consistent regularization},
  author={Yeung, Chun-Kit and Yeung, Dit-Yan},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2019}
}
```

## License

MIT License - see LICENSE file for details.