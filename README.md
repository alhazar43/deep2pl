# Deep-2PL: PyTorch Deep-IRT Implementation

PyTorch implementation of Deep-IRT combining DKVMN with Item Response Theory for explainable knowledge tracing.

## Features

- Factored design with separate student ability and item difficulty networks
- DKVMN memory backbone with IRT 1PL/2PL prediction
- Support for text and CSV data formats
- Full training pipeline with TensorBoard logging

## Usage

```bash
python train.py --data_dir data --train_file train.txt --test_file test.txt --n_questions 100
```

## Model Architecture

1. **DKVMN Memory**: Dynamic key-value memory for sequential learning
2. **Student Ability Network**: Estimates θ from memory content
3. **Item Difficulty Network**: Estimates β from question embeddings  
4. **IRT Predictor**: P(correct) = σ(ability_scale × θ - β)

## Data Format

Text format:
```
15
1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
```

CSV format: `student_id,question_id,correct`

## Installation

```bash
pip install -r requirements.txt
```