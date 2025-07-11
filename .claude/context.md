# Deep-IRT Enhancement Session

## Summary
Enhanced Deep-IRT visualization and pipeline with improved heatmaps, data persistence, and user-friendly workflows.

## Key Changes Made

### 1. Visualization Improvements
- **Colormap**: Changed from `RdYlBu_r` to `RdYlGn` (green = higher values)
- **Layout**: Aligned with image.png format (questions on x-axis, KC on y-axis)  
- **Timeline**: Horizontal correctness bars instead of scatter points
- **No question legends**: Uses simple indices for readability

### 2. Data Persistence System
- **Save/Load**: Added pickle-based theta/beta data saving
- **Fast Viz**: Can generate plots without model inference
- **Reusable**: Extract once, visualize multiple times with different parameters

### 3. Pipeline Integration
- **main.py**: Unified training + visualization workflow
- **Options**: `--save_data`, `--load_data`, `--viz_only` flags
- **Auto-naming**: Default save paths like `saved_data_STATICS_yeung.pkl`

### 4. Modified Files
- `visualize.py`: Enhanced with save/load functions, new colormap, layout changes
- `main.py`: Added data persistence and visualization-only modes
- `README.md`: Updated with new features and usage examples

## Command Line Usage

### Comprehensive Training + Visualization
```bash
cd /deep-2pl && python main.py --dataset STATICS --data_style yeung --epochs 50 --batch_size 32
```

### Visualization Only (from saved data)
```bash
cd /deep-2pl && python main.py --viz_only --load_data saved_data_STATICS_yeung.pkl --output_dir quick_viz --student_idx 5
```

### Direct Visualization Script
```bash
cd /deep-2pl && python visualize.py --load_data saved_data.pkl --output_dir viz --student_idx 0
```

## Technical Details
- Data saved as pickle files containing `global_data` and `per_kc_data` dictionaries
- Per-KC heatmap uses top 20 most variable KCs for visualization
- Red-green colormap with range [-1.0, 1.0]
- Timeline correctness shown as horizontal bars (green=correct, red=incorrect)