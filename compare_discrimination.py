#!/usr/bin/env python3
"""
Compare static vs dynamic discrimination results.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

def load_discrimination_data(discrimination_type):
    """Load saved discrimination data."""
    data_path = f"saved_data_STATICS_{discrimination_type}.pkl"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def compare_discrimination_distributions(static_data, dynamic_data, save_dir="figs/comparison"):
    """Compare discrimination distributions between static and dynamic."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract discrimination values
    static_disc = [d for d in static_data['global_data']['item_discriminations'] if d is not None]
    dynamic_disc = [d for d in dynamic_data['global_data']['item_discriminations'] if d is not None]
    
    # Create comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Static discrimination
    ax1.hist(static_disc, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title(f'Static Discrimination\n(k_t + v_t → α)\nMean: {np.mean(static_disc):.3f} ± {np.std(static_disc):.3f}')
    ax1.set_xlabel('Discrimination (α)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Dynamic discrimination
    ax2.hist(dynamic_disc, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title(f'Dynamic Discrimination\n(f_t → α)\nMean: {np.mean(dynamic_disc):.3f} ± {np.std(dynamic_disc):.3f}')
    ax2.set_xlabel('Discrimination (α)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Overlay comparison
    ax3.hist(static_disc, bins=30, alpha=0.5, color='blue', label='Static', edgecolor='black')
    ax3.hist(dynamic_disc, bins=30, alpha=0.5, color='green', label='Dynamic', edgecolor='black')
    ax3.set_title('Discrimination Comparison')
    ax3.set_xlabel('Discrimination (α)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'discrimination_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'static_mean': np.mean(static_disc),
        'static_std': np.std(static_disc),
        'dynamic_mean': np.mean(dynamic_disc),
        'dynamic_std': np.std(dynamic_disc)
    }

def compare_ability_difficulty_patterns(static_data, dynamic_data, save_dir="figs/comparison"):
    """Compare ability and difficulty patterns."""
    os.makedirs(save_dir, exist_ok=True)
    
    static_global = static_data['global_data']
    dynamic_global = dynamic_data['global_data']
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Student abilities comparison
    ax1.hist(static_global['student_abilities'], bins=30, alpha=0.5, color='blue', label='Static', edgecolor='black')
    ax1.hist(dynamic_global['student_abilities'], bins=30, alpha=0.5, color='green', label='Dynamic', edgecolor='black')
    ax1.set_title('Student Abilities (θ) Comparison')
    ax1.set_xlabel('Student Ability (θ)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Item difficulties comparison
    ax2.hist(static_global['item_difficulties'], bins=30, alpha=0.5, color='blue', label='Static', edgecolor='black')
    ax2.hist(dynamic_global['item_difficulties'], bins=30, alpha=0.5, color='green', label='Dynamic', edgecolor='black')
    ax2.set_title('Item Difficulties (β) Comparison')
    ax2.set_xlabel('Item Difficulty (β)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Scatter plot: Ability vs Difficulty (Static)
    ax3.scatter(static_global['student_abilities'], static_global['item_difficulties'], 
               alpha=0.5, color='blue', s=1)
    ax3.set_title('Static: Ability vs Difficulty')
    ax3.set_xlabel('Student Ability (θ)')
    ax3.set_ylabel('Item Difficulty (β)')
    ax3.grid(True, alpha=0.3)
    
    # Scatter plot: Ability vs Difficulty (Dynamic)
    ax4.scatter(dynamic_global['student_abilities'], dynamic_global['item_difficulties'], 
               alpha=0.5, color='green', s=1)
    ax4.set_title('Dynamic: Ability vs Difficulty')
    ax4.set_xlabel('Student Ability (θ)')
    ax4.set_ylabel('Item Difficulty (β)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ability_difficulty_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_discrimination_heatmap(static_data, dynamic_data, save_dir="figs/comparison"):
    """Create discrimination heatmap comparison."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract discrimination by question ID
    static_disc_by_q = {}
    dynamic_disc_by_q = {}
    
    static_global = static_data['global_data']
    dynamic_global = dynamic_data['global_data']
    
    for i, q_id in enumerate(static_global['question_ids']):
        if static_global['item_discriminations'][i] is not None:
            if q_id not in static_disc_by_q:
                static_disc_by_q[q_id] = []
            static_disc_by_q[q_id].append(static_global['item_discriminations'][i])
    
    for i, q_id in enumerate(dynamic_global['question_ids']):
        if dynamic_global['item_discriminations'][i] is not None:
            if q_id not in dynamic_disc_by_q:
                dynamic_disc_by_q[q_id] = []
            dynamic_disc_by_q[q_id].append(dynamic_global['item_discriminations'][i])
    
    # Average discrimination per question
    static_avg = {q_id: np.mean(discs) for q_id, discs in static_disc_by_q.items()}
    dynamic_avg = {q_id: np.mean(discs) for q_id, discs in dynamic_disc_by_q.items()}
    
    # Create comparison dataframe
    common_questions = set(static_avg.keys()) & set(dynamic_avg.keys())
    comparison_data = []
    
    for q_id in sorted(common_questions):
        comparison_data.append({
            'question_id': q_id,
            'static_discrimination': static_avg[q_id],
            'dynamic_discrimination': dynamic_avg[q_id],
            'difference': dynamic_avg[q_id] - static_avg[q_id]
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Question-wise discrimination comparison
    sample_questions = df.head(50)  # Show first 50 questions
    
    width = 0.35
    x = np.arange(len(sample_questions))
    
    ax1.bar(x - width/2, sample_questions['static_discrimination'], width, 
            label='Static', color='blue', alpha=0.7)
    ax1.bar(x + width/2, sample_questions['dynamic_discrimination'], width, 
            label='Dynamic', color='green', alpha=0.7)
    
    ax1.set_xlabel('Question ID')
    ax1.set_ylabel('Discrimination (α)')
    ax1.set_title('Discrimination by Question (First 50 Questions)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot: Static vs Dynamic discrimination
    ax2.scatter(df['static_discrimination'], df['dynamic_discrimination'], 
               alpha=0.6, s=20)
    ax2.plot([df['static_discrimination'].min(), df['static_discrimination'].max()], 
             [df['static_discrimination'].min(), df['static_discrimination'].max()], 
             'r--', label='y=x')
    ax2.set_xlabel('Static Discrimination')
    ax2.set_ylabel('Dynamic Discrimination')
    ax2.set_title('Static vs Dynamic Discrimination')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'discrimination_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def generate_summary_report(discrimination_stats, save_dir="figs/comparison"):
    """Generate a summary report."""
    os.makedirs(save_dir, exist_ok=True)
    
    report = f"""
# Discrimination Comparison Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Discrimination Statistics

### Static Discrimination (α = softplus(W_a [k_t; v_t] + b_a))
- Mean: {discrimination_stats['static_mean']:.4f}
- Standard Deviation: {discrimination_stats['static_std']:.4f}
- Uses question embedding (k_t) + question-answer embedding (v_t)
- Traditional IRT approach - item property

### Dynamic Discrimination (α = softplus(W_a f_t + b_a))
- Mean: {discrimination_stats['dynamic_mean']:.4f}
- Standard Deviation: {discrimination_stats['dynamic_std']:.4f}
- Uses summary vector (f_t) - student knowledge state
- Adaptive approach - varies with student state

## Key Differences

### Theoretical
- **Static**: Discrimination depends on item characteristics
- **Dynamic**: Discrimination adapts to student knowledge state

### Practical
- **Static**: More stable, aligns with traditional IRT
- **Dynamic**: More flexible, potentially more personalized

### Statistical
- Mean difference: {discrimination_stats['dynamic_mean'] - discrimination_stats['static_mean']:.4f}
- Variance ratio: {discrimination_stats['dynamic_std'] / discrimination_stats['static_std']:.4f}

## Files Generated
- discrimination_comparison.png
- ability_difficulty_comparison.png
- discrimination_heatmap.png
- comparison_report.md
"""
    
    with open(os.path.join(save_dir, 'comparison_report.md'), 'w') as f:
        f.write(report)
    
    print(report)

def main():
    """Main comparison function."""
    print("=== Discrimination Comparison Analysis ===")
    
    # Load data
    print("Loading data...")
    static_data = load_discrimination_data("static")
    dynamic_data = load_discrimination_data("dynamic")
    
    if static_data is None or dynamic_data is None:
        print("Error: Could not load data files. Make sure to run the test script first.")
        return
    
    print("Data loaded successfully.")
    
    # Create comparison directory
    comparison_dir = "figs/comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Compare discrimination distributions
    print("Comparing discrimination distributions...")
    discrimination_stats = compare_discrimination_distributions(static_data, dynamic_data, comparison_dir)
    
    # Compare ability/difficulty patterns
    print("Comparing ability and difficulty patterns...")
    compare_ability_difficulty_patterns(static_data, dynamic_data, comparison_dir)
    
    # Create discrimination heatmap
    print("Creating discrimination heatmap...")
    comparison_df = create_discrimination_heatmap(static_data, dynamic_data, comparison_dir)
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(discrimination_stats, comparison_dir)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {comparison_dir}/")
    print("Generated files:")
    print("- discrimination_comparison.png")
    print("- ability_difficulty_comparison.png")
    print("- discrimination_heatmap.png")
    print("- comparison_report.md")

if __name__ == "__main__":
    main()