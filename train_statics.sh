#!/bin/bash
# Professional Training and Visualization Script for STATICS Dataset
# 
# This script provides a complete pipeline for training the Deep Item Response
# Theory model on the STATICS dataset and generating comprehensive visualizations.
#
# Features:
# - Automated training with optimized parameters
# - Comprehensive visualization generation
# - Organized output structure in figs/STATICS/
# - Data persistence for efficient re-analysis
#
# Author: Deep-IRT Pipeline System

echo "=================================================="
echo "Deep-IRT Training and Visualization for STATICS"
echo "=================================================="

# Define training configuration parameters
DATASET="STATICS"
DATA_STYLE="yeung"
EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=0.001
STUDENT_IDX=0
FOLD=0

echo "Configuration:"
echo "- Dataset: $DATASET"
echo "- Data Style: $DATA_STYLE (per-KC mode)"
echo "- Epochs: $EPOCHS"
echo "- Batch Size: $BATCH_SIZE"
echo "- Learning Rate: $LEARNING_RATE"
echo "- Student Index: $STUDENT_IDX"
echo "- Fold: $FOLD"
echo "- Output Directory: figs/STATICS"
echo ""

# Execute the complete training and visualization pipeline
echo "Starting training and visualization pipeline..."
python main.py \
    --dataset $DATASET \
    --data_style $DATA_STYLE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --student_idx $STUDENT_IDX \
    --fold $FOLD

# Evaluate pipeline execution results and provide status report
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "SUCCESS: Training and visualization completed!"
    echo "=================================================="
    echo "Results saved in: figs/STATICS/"
    echo ""
    echo "Generated files:"
    if [ -d "figs/STATICS" ]; then
        ls -la figs/STATICS/
    fi
    echo ""
    echo "Saved data file: saved_data_STATICS_yeung.pkl"
    echo ""
    echo "Next steps:"
    echo "1. Review generated visualizations in figs/STATICS/"
    echo "2. Generate additional student visualizations using:"
    echo "   python main.py --viz_only --load_data saved_data_STATICS_yeung.pkl --student_idx N"
    echo "3. Analyze model performance using saved theta and beta parameters"
else
    echo ""
    echo "=================================================="
    echo "FAILED: Training or visualization failed!"
    echo "=================================================="
    echo "Please review error messages above for troubleshooting guidance."
fi