#!/usr/bin/env python3
"""
Visualization Module for Deep Item Response Theory Model

This module provides comprehensive visualization capabilities for Deep-IRT models,
supporting both global and per-knowledge component tracking modes.

Features:
- Global theta/beta distribution plots for standard models
- Per-KC continuous heatmaps for models with Q-matrix support
- Data persistence and loading capabilities for saved theta/beta parameters
- Professional visualization output with organized directory structure

Author: Deep-IRT Visualization Suite
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse
import json
import pickle
from tqdm import tqdm

from models.irt import DeepIRTModel
from data.dataloader import create_datasets, create_dataloader


def load_model(checkpoint_path, config):
    """
    Load trained Deep-IRT model from checkpoint file.
    
    Parameters:
        checkpoint_path (str): Path to the model checkpoint file
        config (dict): Model configuration dictionary
        
    Returns:
        DeepIRTModel: Loaded model in evaluation mode
    """
    # Load checkpoint to extract actual model dimensions
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Extract actual n_questions from q_embed.weight shape
    # q_embed has shape (n_questions + 1, embed_dim) due to padding_idx=0
    actual_n_questions = state_dict['q_embed.weight'].shape[0] - 1
    
    model = DeepIRTModel(
        n_questions=actual_n_questions,  # Use actual value from checkpoint
        memory_size=config['memory_size'],
        key_memory_state_dim=config['key_memory_state_dim'],
        value_memory_state_dim=config['value_memory_state_dim'],
        summary_vector_dim=config['summary_vector_dim'],
        q_embed_dim=config['q_embed_dim'],
        qa_embed_dim=config['qa_embed_dim'],
        ability_scale=config['ability_scale'],
        use_discrimination=config['use_discrimination'],
        dropout_rate=config['dropout_rate'],
        q_matrix_path=config.get('q_matrix_path'),
        skill_mapping_path=config.get('skill_mapping_path')
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_data(model, data_loader, max_students=50, max_timesteps=None):
    """
    Extract theta and beta parameters from trained model for visualization.
    
    Parameters:
        model (DeepIRTModel): Trained Deep-IRT model
        data_loader (DataLoader): Data loader for test dataset
        max_students (int): Maximum number of students to process
        max_timesteps (int, optional): Maximum timesteps per student. If None, uses full sequence length
        
    Returns:
        tuple: (global_data, per_kc_data) containing extracted parameters
    """
    model.eval()
    
    global_data = {
        'student_abilities': [],
        'item_difficulties': [],
        'student_ids': [],
        'timesteps': [],
        'question_ids': []
    }
    
    per_kc_data = {
        'all_kc_thetas_list': [],
        'question_sequences': [],
        'correctness_sequences': [],
        'beta_parameters': [],
        'probability_data': []
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting data")):
            if batch_idx >= max_students:
                break
                
            q_data, qa_data = batch['q_data'], batch['qa_data']
            _, student_abilities, item_difficulties, _, kc_info = model(q_data, qa_data)
            
            batch_size, seq_len = q_data.shape
            if max_timesteps is not None:
                seq_len = min(seq_len, max_timesteps)
            
            # Extract global theta and beta parameters
            for student_idx in range(batch_size):
                for t in range(seq_len):
                    if q_data[student_idx, t].item() > 0:
                        global_data['student_abilities'].append(student_abilities[student_idx, t].item())
                        global_data['item_difficulties'].append(item_difficulties[student_idx, t].item())
                        global_data['student_ids'].append(batch_idx * batch_size + student_idx)
                        global_data['timesteps'].append(t)
                        global_data['question_ids'].append(q_data[student_idx, t].item())
            
            # Extract per-knowledge component data if in per-KC mode
            if 'all_kc_thetas' in kc_info:
                batch_size, seq_len, _ = kc_info['all_kc_thetas'].shape
                if max_timesteps is not None:
                    seq_len = min(seq_len, max_timesteps)
                
                for student_idx in range(batch_size):
                    # Extract knowledge component theta evolution
                    student_kc_thetas = kc_info['all_kc_thetas'][student_idx, :seq_len, :].cpu().numpy()
                    per_kc_data['all_kc_thetas_list'].append(student_kc_thetas)
                    
                    # Extract question sequence and correctness information
                    student_questions = []
                    student_correctness = []
                    student_betas = []
                    student_probs = []
                    
                    for t in range(seq_len):
                        q_id = q_data[student_idx, t].item()
                        student_questions.append(q_id)
                        student_betas.append(item_difficulties[student_idx, t].cpu().item())
                        
                        # Extract correctness based on qa_data encoding
                        if q_id > 0:
                            qa_val = qa_data[student_idx, t].item()
                            # qa_data encodes: correct = q_id * 2 + 1, incorrect = q_id * 2
                            if qa_val == q_id * 2 + 1:  # Correct answer
                                student_correctness.append(1)
                            elif qa_val == q_id * 2:  # Incorrect answer
                                student_correctness.append(0)
                            else:  # Invalid qa value
                                student_correctness.append(-1)
                        else:
                            student_correctness.append(-1)  # No question
                        
                        # Calculate per-KC success probabilities using IRT model
                        beta_val = student_betas[t]
                        kc_probs = torch.sigmoid(torch.tensor(student_kc_thetas[t, :]) - beta_val).numpy()
                        student_probs.append(kc_probs)
                    
                    per_kc_data['question_sequences'].append(student_questions)
                    per_kc_data['correctness_sequences'].append(student_correctness)
                    per_kc_data['beta_parameters'].append(student_betas)
                    per_kc_data['probability_data'].append(np.array(student_probs))
    
    # Convert lists to numpy arrays for efficient processing
    for key in global_data:
        global_data[key] = np.array(global_data[key])
    
    return global_data, per_kc_data


def plot_global_heatmap(data, save_path=None):
    """
    Create global theta heatmap visualization.
    
    Parameters:
        data (dict): Dictionary containing global theta data
        save_path (str, optional): Path to save the visualization
    """
    df = pd.DataFrame({
        'Student': data['student_ids'],
        'Timestep': data['timesteps'],
        'Theta': data['student_abilities']
    })
    
    theta_matrix = df.pivot_table(values='Theta', index='Student', columns='Timestep', aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(theta_matrix, cmap='RdYlGn', center=0, cbar_kws={'label': 'Student Ability (θ)'})
    plt.title('Student Ability (θ) Evolution Over Time\nDeep-IRT Model (Global)', fontsize=14)
    plt.xlabel('Time Step')
    plt.ylabel('Student ID')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_beta_distribution(data, save_path=None):
    """
    Create item difficulty (beta) distribution visualization.
    
    Parameters:
        data (dict): Dictionary containing beta parameters
        save_path (str, optional): Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(data['item_difficulties'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Item Difficulty (β)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Item Difficulties (β)\nDeep-IRT Model')
    plt.grid(True, alpha=0.3)
    
    mean_beta = np.mean(data['item_difficulties'])
    std_beta = np.std(data['item_difficulties'])
    plt.axvline(mean_beta, color='red', linestyle='--', label=f'Mean: {mean_beta:.3f}')
    plt.axvline(mean_beta + std_beta, color='orange', linestyle='--', alpha=0.7, label=f'±1 STD: {std_beta:.3f}')
    plt.axvline(mean_beta - std_beta, color='orange', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_kc_heatmap(kc_thetas, questions, correctness, student_idx=0, mode='theta', kc_names=None, save_path=None):
    """
    Create per-knowledge component heatmap visualization.
    
    Parameters:
        kc_thetas (list): List of KC theta matrices for each student
        questions (list): List of question sequences for each student
        correctness (list): List of correctness sequences for each student
        student_idx (int): Index of student to visualize
        mode (str): Visualization mode ('theta' or 'probability')
        kc_names (dict, optional): Mapping of KC IDs to names
        save_path (str, optional): Path to save the visualization
    """
    if student_idx >= len(kc_thetas):
        student_idx = 0
    
    data = kc_thetas[student_idx]
    n_timesteps, _ = data.shape
    
    # Select knowledge components with highest variance for visualization
    kc_variations = np.std(data, axis=0)
    top_kcs = np.argsort(kc_variations)[-20:]
    
    # Create subplot structure for heatmap and correctness indicators
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 1], width_ratios=[5, 0.2], 
                         hspace=0.05, wspace=0.02)
    ax1 = fig.add_subplot(gs[0, 0])  # Main heatmap
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Question timeline
    cax = fig.add_subplot(gs[0, 1])  # Colorbar
    
    # Generate main heatmap visualization
    data_subset = data[:, top_kcs].T
    vmin, vmax = (-1.0, 1.0) if mode == 'theta' else (0.0, 1.0)
    
    im = ax1.imshow(data_subset, cmap='RdYlGn', aspect='auto', 
                   vmin=vmin, vmax=vmax, interpolation='bilinear',
                   extent=[0, n_timesteps, 0, len(top_kcs)], origin='lower')
    
    # Add colorbar with appropriate label using dedicated colorbar axis
    cbar = plt.colorbar(im, cax=cax)
    label = 'Student Ability Level (θ)' if mode == 'theta' else 'Success Probability P(correct)'
    cbar.set_label(label, fontsize=12)
    
    # Configure plot title and axis labels
    title = f'Student Ability Level' if mode == 'theta' else 'Per-KC Success Probability'
    ax1.set_title(f'{title}\nStudent {student_idx} - Per-KC Evolution', fontsize=14, pad=20)
    ax1.set_ylabel('Knowledge Concepts', fontsize=12)
    
    # Configure Y-axis labels for knowledge components
    y_ticks = np.arange(len(top_kcs)) + 0.5
    if kc_names:
        y_labels = [kc_names.get(kc, f'KC_{kc+1}') for kc in top_kcs]
    else:
        y_labels = [f'KC_{kc+1}' for kc in top_kcs]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels, fontsize=8)
    ax1.set_xticks([])
    
    # Generate bottom subplot with correctness indicators
    x_positions = np.linspace(0.5, n_timesteps - 0.5, n_timesteps)
    
    # Extract student-specific data
    student_questions = questions[student_idx] if student_idx < len(questions) else []
    student_correctness = correctness[student_idx] if student_idx < len(correctness) else []
    
    for t in range(min(n_timesteps, len(student_questions))):
        q_id = student_questions[t]
        # Handle case where q_id might be a list or array
        if hasattr(q_id, '__iter__') and not isinstance(q_id, str):
            q_id = q_id[0] if len(q_id) > 0 else 0
        
        if isinstance(q_id, (int, float)) and q_id > 0:
            correct = student_correctness[t] if t < len(student_correctness) else -1
            if correct == 1:
                # Correct answer - filled green circle without black outline
                ax2.scatter(x_positions[t], 0.5, s=120, c='green', alpha=0.8)
            elif correct == 0:
                # Incorrect answer - empty red circle without outline
                ax2.scatter(x_positions[t], 0.5, s=120, facecolors='none', 
                           edgecolors='red', linewidth=1.5)
            # If correct == -1, don't plot anything (no valid answer)
    
    # Align axes and configure subplot layout
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Questions (Timeline)', fontsize=12)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    
    ax1.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_data(data, filepath):
    """
    Save extracted data to pickle file for future use.
    
    Parameters:
        data (dict): Dictionary containing extracted data
        filepath (str): Path to save the pickle file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved data to {filepath}")


def load_data(filepath):
    """
    Load previously saved data from pickle file.
    
    Parameters:
        filepath (str): Path to the pickle file
        
    Returns:
        dict or None: Loaded data dictionary or None if file doesn't exist
    """
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


def visualize_from_data(data_path, output_dir, student_idx=0, dataset_name=None):
    """
    Create visualizations from previously saved data without model inference.
    
    Parameters:
        data_path (str): Path to saved data pickle file
        output_dir (str): Directory to save visualizations
        student_idx (int): Index of student to visualize
        dataset_name (str, optional): Name of dataset for directory structure
        
    Returns:
        bool: True if successful, False otherwise
    """
    data = load_data(data_path)
    if data is None:
        print(f"No data found at {data_path}")
        return False
    
    # Determine dataset name from data or use provided name
    if dataset_name is None:
        dataset_name = data.get('dataset_name', 'unknown')
    
    # Create organized output directory structure
    if not output_dir.startswith('figs/'):
        output_dir = os.path.join('figs', dataset_name)
    
    os.makedirs(output_dir, exist_ok=True)
    created_files = []
    
    # Generate per-knowledge component visualizations
    if 'per_kc_data' in data:
        per_kc = data['per_kc_data']
        kc_thetas = per_kc['all_kc_thetas_list']
        questions = per_kc['question_sequences']
        correctness = per_kc['correctness_sequences']
        probabilities = per_kc.get('probability_data', [])
        
        if kc_thetas:
            # Generate theta heatmap visualization
            theta_path = os.path.join(output_dir, f"per_kc_theta_student_{student_idx}.png")
            plot_per_kc_heatmap(kc_thetas, questions, correctness, student_idx, 'theta', None, theta_path)
            created_files.append(f"per_kc_theta_student_{student_idx}.png")
            
            # Generate probability heatmap visualization
            if probabilities:
                prob_path = os.path.join(output_dir, f"per_kc_probability_student_{student_idx}.png")
                plot_per_kc_heatmap(probabilities, questions, correctness, student_idx, 'probability', None, prob_path)
                created_files.append(f"per_kc_probability_student_{student_idx}.png")
    
    # Generate global visualizations
    if 'global_data' in data:
        global_data = data['global_data']
        
        # Generate global theta heatmap
        global_path = os.path.join(output_dir, "global_theta_heatmap.png")
        plot_global_heatmap(global_data, global_path)
        created_files.append("global_theta_heatmap.png")
        
        # Generate beta distribution plot
        beta_path = os.path.join(output_dir, "beta_distribution.png")
        plot_beta_distribution(global_data, beta_path)
        created_files.append("beta_distribution.png")
    
    print(f"Created {len(created_files)} visualizations:")
    for file in created_files:
        print(f"  - {file}")
    
    return True


def main():
    """
    Main function for command-line interface to visualization module.
    
    Handles argument parsing and coordinates visualization workflow.
    """
    parser = argparse.ArgumentParser(description='Deep-IRT Model Visualization')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--config', type=str, help='Model config JSON file')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory')
    parser.add_argument('--student_idx', type=int, default=0, help='Student index for per-KC plots')
    parser.add_argument('--load_data', type=str, help='Load saved data file')
    parser.add_argument('--save_data', type=str, help='Save extracted data file')
    
    args = parser.parse_args()
    
    # Handle visualization from saved data
    if args.load_data:
        return visualize_from_data(args.load_data, args.output_dir, args.student_idx)
    
    # Handle model-based visualization
    if not args.checkpoint or not args.config:
        print("Error: --checkpoint and --config required when not using --load_data")
        return
    
    # Load model configuration and checkpoint
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    model = load_model(args.checkpoint, config)
    
    # Initialize test data loader
    try:
        _, _, test_dataset = create_datasets(
            data_style=config['data_style'],
            data_dir=config['data_dir'],
            dataset_name=config['dataset_name'],
            seq_len=config['seq_len'],
            n_questions=config['n_questions'],
            k_fold=config['k_fold'],
            fold_idx=config['fold_idx']
        )
        test_loader = create_dataloader(test_dataset, batch_size=1, shuffle=False)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Extract data from model
    global_data, per_kc_data = extract_data(model, test_loader)
    
    # Create figs/dataset_name structure
    dataset_name = config.get('dataset_name', 'unknown')
    if not args.output_dir.startswith('figs/'):
        args.output_dir = os.path.join('figs', dataset_name)
    
    os.makedirs(args.output_dir, exist_ok=True)
    created_files = []
    
    data_to_save = {'global_data': global_data, 'dataset_name': dataset_name}
    
    # Generate global visualizations
    if len(global_data['student_abilities']) > 0:
        global_path = os.path.join(args.output_dir, "global_theta_heatmap.png")
        plot_global_heatmap(global_data, global_path)
        created_files.append("global_theta_heatmap.png")
        
        beta_path = os.path.join(args.output_dir, "beta_distribution.png")
        plot_beta_distribution(global_data, beta_path)
        created_files.append("beta_distribution.png")
    
    # Generate per-knowledge component visualizations
    if model.per_kc_mode and per_kc_data['all_kc_thetas_list']:
        data_to_save['per_kc_data'] = per_kc_data
        
        theta_path = os.path.join(args.output_dir, f"per_kc_theta_student_{args.student_idx}.png")
        plot_per_kc_heatmap(
            per_kc_data['all_kc_thetas_list'],
            per_kc_data['question_sequences'],
            per_kc_data['correctness_sequences'],
            args.student_idx, 'theta', model.kc_names, theta_path
        )
        created_files.append(f"per_kc_theta_student_{args.student_idx}.png")
        
        if per_kc_data['probability_data']:
            prob_path = os.path.join(args.output_dir, f"per_kc_probability_student_{args.student_idx}.png")
            plot_per_kc_heatmap(
                per_kc_data['probability_data'],
                per_kc_data['question_sequences'],
                per_kc_data['correctness_sequences'],
                args.student_idx, 'probability', model.kc_names, prob_path
            )
            created_files.append(f"per_kc_probability_student_{args.student_idx}.png")
    
    # Save extracted data if requested
    if args.save_data:
        save_data(data_to_save, args.save_data)
    
    # Display summary of generated visualizations
    print(f"Created {len(created_files)} visualizations:")
    for file in created_files:
        print(f"  - {file}")
    
    if len(global_data['student_abilities']) > 0:
        print(f"θ: {np.mean(global_data['student_abilities']):.3f}±{np.std(global_data['student_abilities']):.3f}, "
              f"β: {np.mean(global_data['item_difficulties']):.3f}±{np.std(global_data['item_difficulties']):.3f}")
    
    return True


if __name__ == "__main__":
    main()