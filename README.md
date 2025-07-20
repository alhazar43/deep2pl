# Deep-2PL: Optimized Deep Item Response Theory Model

A fixed and optimized Deep-IRT implementation using Dynamic Key-Value Memory Networks (DKVMN). This version addresses critical bugs, improves training performance, and provides comprehensive analysis tools.

## Key Improvements

- **Memory Gradient Fix**: Resolved gradient flow breaking in DKVMN memory operations
- **Training Speed**: Fixed I/O overhead causing extreme slowdown (194s → 5s per epoch)
- **Per-KC Support**: Automatic Q-matrix detection for knowledge component analysis  
- **IRT Parameters**: Fast extraction of theta, alpha, beta parameters from trained models
- **Visualization**: Alpha/beta distributions and per-KC analysis for Q-matrix datasets
- **Cross-Platform**: Complete Linux, Windows batch, and PowerShell support

## Quick Start

### Linux/macOS
```bash
conda activate vrec-env

# Single dataset
./run_dataset.sh STATICS
./run_dataset.sh synthetic --quick

# All datasets
./run_all_datasets.sh --exclude large
```

### Windows
```cmd
conda activate vrec-env

run_dataset.bat STATICS
run_all_datasets.bat --exclude large
```

## Supported Datasets

**With Q-matrix**: STATICS (956q, 98 KCs), assist2009_updated (70q), fsaif1tof3 (1722q)  
**Without Q-matrix**: assist2015 (96q), assist2017 (102q), synthetic (50q), assist2009 (124q), statics2011 (1221q), kddcup2010 (649q)

## Performance

**Training Speed**: 5-10 seconds per epoch (optimized from 194s baseline)  
**Model Performance**: AUC 0.69-0.71, Accuracy 0.65-0.75  
**Time Estimates**: Small datasets 3-5 min, Medium 15-25 min, Large 30-45 min

## Output Structure

```
results/train/          # Training metrics and curves
results/test/           # Model evaluation results  
results/plots/          # ROC/PR curves and performance plots
figs/{dataset}/         # Alpha/beta distributions, per-KC analysis, ability heatmaps
stats/                  # IRT parameters (theta, alpha, beta) from best models
save_models/            # Best and final model checkpoints
```

## Manual Commands

```bash
# Training
python train.py --dataset STATICS --epochs 20
python train.py --dataset assist2015 --single_fold --fold 0

# Evaluation  
python evaluate.py --model_path save_models/best_model_STATICS_fold0.pth

# Visualization
python visualize.py --checkpoint save_models/best_model_STATICS_fold0.pth \
                   --config save_models/config_STATICS_fold0.json
```

## Architecture

**Core**: DKVMN with attention-based memory operations  
**Per-KC Mode**: Automatic Q-matrix detection enables KC-specific tracking  
**Global Mode**: Single memory bank for datasets without Q-matrix  
**IRT Integration**: Extract traditional IRT parameters from neural model

## Dependencies

```bash
conda activate vrec-env
# torch, numpy, pandas, scikit-learn, matplotlib
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