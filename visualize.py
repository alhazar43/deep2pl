#!/usr/bin/env python3
"""Visualization module for Deep-IRT models.
Supports global theta/beta plots and per-KC heatmaps.
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
# tqdm removed for performance

from models.model import DeepIRTModel
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
        key_dim=config.get('key_dim', config.get('key_memory_state_dim', 50)),
        value_dim=config.get('value_dim', config.get('value_memory_state_dim', 200)),
        summary_dim=config.get('summary_dim', config.get('summary_vector_dim', 50)),
        q_embed_dim=config['q_embed_dim'],
        qa_embed_dim=config['qa_embed_dim'],
        ability_scale=config['ability_scale'],
        use_discrimination=config['use_discrimination'],
        dropout_rate=config['dropout_rate'],
        q_matrix_path=config.get('q_matrix_path'),
        skill_mapping_path=config.get('skill_mapping_path')
    )
    
    # Handle backward compatibility for renamed parameters
    old_to_new_mapping = {
        'init_kc_states': 'kc_init',
        'cross_kc_network.0.weight': 'kc_cross_net.0.weight',
        'cross_kc_network.0.bias': 'kc_cross_net.0.bias',
        'cross_kc_network.2.weight': 'kc_cross_net.2.weight',
        'cross_kc_network.2.bias': 'kc_cross_net.2.bias'
    }
    
    # Rename keys in state_dict if needed
    for old_key, new_key in old_to_new_mapping.items():
        if old_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_data(model, data_loader, max_students=50, max_timesteps=None):
    """Extract theta/beta parameters from trained model.
    Returns: (global_data, per_kc_data) tuple
    """
    model.eval()
    
    global_data = {
        'student_abilities': [],
        'item_difficulties': [],
        'discrimination_params': [],
        'student_ids': [],
        'timesteps': [],
        'question_ids': []
    }
    
    per_kc_data = {
        'kc_thetas': [],
        'q_seqs': [],
        'correct_seqs': [],
        'beta_parameters': [],
        'probability_data': []
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
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
                        
                        # Extract question-specific discrimination parameters
                        q_id = q_data[student_idx, t].item()
                        if hasattr(model, 'prediction_network') and len(model.prediction_network) >= 4:
                            final_layer = model.prediction_network[-1]
                            if hasattr(final_layer, 'weight') and final_layer.weight is not None:
                                # Get question-specific discrimination from embeddings
                                if hasattr(model, 'q_embed') and q_id > 0 and q_id <= model.q_embed.num_embeddings - 1:
                                    q_embed_weight = model.q_embed.weight[q_id]
                                    # Use embedding norm as discrimination proxy
                                    discrimination_proxy = torch.norm(q_embed_weight).item()
                                    # Scale by final layer weight magnitude
                                    weight_scale = torch.abs(final_layer.weight).mean().item()
                                    discrimination_proxy = discrimination_proxy * weight_scale
                                else:
                                    discrimination_proxy = 1.0
                                global_data['discrimination_params'].append(discrimination_proxy)
                            else:
                                global_data['discrimination_params'].append(1.0)
                        else:
                            global_data['discrimination_params'].append(1.0)
            
            # Extract per-knowledge component data if in per-KC mode
            if 'all_kc_thetas' in kc_info:
                batch_size, seq_len, _ = kc_info['all_kc_thetas'].shape
                if max_timesteps is not None:
                    seq_len = min(seq_len, max_timesteps)
                
                for student_idx in range(batch_size):
                    # Extract knowledge component theta evolution
                    student_kc_thetas = kc_info['all_kc_thetas'][student_idx, :seq_len, :].cpu().numpy()
                    per_kc_data['kc_thetas'].append(student_kc_thetas)
                    
                    # Extract question sequence and correctness information
                    student_questions = []
                    student_correctness = []
                    student_betas = []
                    student_probs = []
                    
                    # Find actual sequence length by stopping at padding (q_id = 0)
                    actual_seq_len = seq_len
                    for t in range(seq_len):
                        if q_data[student_idx, t].item() == 0:  # Hit padding
                            actual_seq_len = t
                            break
                    
                    for t in range(actual_seq_len):
                        q_id = q_data[student_idx, t].item()
                        student_questions.append(q_id)
                        student_betas.append(item_difficulties[student_idx, t].cpu().item())
                        
                        # Extract correctness based on qa_data encoding
                        if q_id > 0:
                            qa_val = qa_data[student_idx, t].item()
                            # qa_data encodes: correct = q_id + n_questions, incorrect = q_id
                            # This encoding is used by all data loaders in the codebase
                            if qa_val == q_id + model.n_questions:  # Correct answer
                                student_correctness.append(1)
                            elif qa_val == q_id:  # Incorrect answer
                                student_correctness.append(0)
                            else:  # Invalid qa value - log for debugging
                                print(f"Warning: Invalid qa_value {qa_val} for q_id {q_id}, expected {q_id} or {q_id + model.n_questions}")
                                student_correctness.append(-1)
                        else:
                            student_correctness.append(-1)  # No question
                        
                        # Calculate per-KC success probabilities using IRT model
                        beta_val = student_betas[t]
                        kc_probs = torch.sigmoid(torch.tensor(student_kc_thetas[t, :]) - beta_val).numpy()
                        student_probs.append(kc_probs)
                    
                    per_kc_data['q_seqs'].append(student_questions)
                    per_kc_data['correct_seqs'].append(student_correctness)
                    per_kc_data['beta_parameters'].append(student_betas)
                    per_kc_data['probability_data'].append(np.array(student_probs))
    
    # Convert lists to numpy arrays for efficient processing
    for key in global_data:
        global_data[key] = np.array(global_data[key])
    
    return global_data, per_kc_data


def plot_global_heatmap(data, save_path=None, max_steps=100):
    """
    Create global theta heatmap visualization.
    
    Parameters:
        data (dict): Dictionary containing global theta data
        save_path (str, optional): Path to save the visualization
        max_steps (int): Maximum timesteps to display in plot (default: 100)
    """
    df = pd.DataFrame({
        'Student': data['student_ids'],
        'Timestep': data['timesteps'],
        'Theta': data['student_abilities']
    })
    
    # Limit display to max_steps while keeping full computation info
    max_timestep = df['Timestep'].max()
    display_timesteps = min(max_timestep, max_steps - 1)  # -1 because timesteps are 0-indexed
    
    # Filter data for display
    df_display = df[df['Timestep'] <= display_timesteps]
    
    theta_matrix = df_display.pivot_table(values='Theta', index='Student', columns='Timestep', aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(theta_matrix, cmap='RdYlGn', center=0, cbar_kws={'label': 'Student Ability (θ)'})
    
    # Update title to show display information
    total_info = f" (showing first {display_timesteps + 1} of {max_timestep + 1} timesteps)" if display_timesteps < max_timestep else ""
    plt.title(f'Student Ability (θ) Evolution Over Time\nDeep-IRT Model (Global){total_info}', fontsize=14)
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


def plot_alpha_distribution(data, save_path=None):
    """
    Plot the distribution of discrimination parameters (alpha).
    
    Parameters:
        data (dict): Data dictionary containing 'discrimination_params'
        save_path (str, optional): Path to save the plot
    """
    if 'discrimination_params' not in data or len(data['discrimination_params']) == 0:
        print("No discrimination parameters found in data")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Filter out default 1.0 values if all are the same (1PL mode)
    alpha_values = data['discrimination_params']
    unique_alphas = set(alpha_values)
    
    if len(unique_alphas) == 1 and 1.0 in unique_alphas:
        plt.text(0.5, 0.5, 'Model trained in 1PL mode\n(no discrimination parameters)', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.xlim(0, 2)
        plt.ylim(0, 1)
    else:
        plt.hist(alpha_values, bins=50, alpha=0.7, edgecolor='black', color='purple')
        mean_alpha = np.mean(alpha_values)
        std_alpha = np.std(alpha_values)
        plt.axvline(mean_alpha, color='red', linestyle='--', label=f'Mean: {mean_alpha:.3f}')
        plt.axvline(mean_alpha + std_alpha, color='orange', linestyle='--', alpha=0.7, label=f'±1 STD: {std_alpha:.3f}')
        plt.axvline(mean_alpha - std_alpha, color='orange', linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.xlabel('Discrimination Parameter (α)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Item Discrimination (α)\nDeep-2PL Model')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_kc_heatmap(kc_thetas, questions, correctness, student_idx=0, mode='theta', kc_names=None, save_path=None, max_steps=100):
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
        max_steps (int): Maximum timesteps to display in plot (default: 100)
    """
    if student_idx >= len(kc_thetas):
        student_idx = 0
    
    data = kc_thetas[student_idx]
    n_timesteps, _ = data.shape
    
    # Limit display to max_steps while keeping full computation info
    display_timesteps = min(n_timesteps, max_steps)
    display_data = data[:display_timesteps, :]
    
    # Select knowledge components with highest variance for visualization (using full data)
    kc_variations = np.std(data, axis=0)
    top_kcs = np.argsort(kc_variations)[-20:]
    
    # Create subplot structure for heatmap and correctness indicators
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 1], width_ratios=[5, 0.2], 
                         hspace=0.05, wspace=0.02)
    ax1 = fig.add_subplot(gs[0, 0])  # Main heatmap
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Question timeline
    cax = fig.add_subplot(gs[0, 1])  # Colorbar
    
    # Generate main heatmap visualization (display subset)
    data_subset = display_data[:, top_kcs].T
    vmin, vmax = (-1.0, 1.0) if mode == 'theta' else (0.0, 1.0)
    
    im = ax1.imshow(data_subset, cmap='RdYlGn', aspect='auto', 
                   vmin=vmin, vmax=vmax, interpolation='bilinear',
                   extent=[0, display_timesteps, 0, len(top_kcs)], origin='lower')
    
    # Add colorbar with appropriate label using dedicated colorbar axis
    cbar = plt.colorbar(im, cax=cax)
    label = 'Student Ability Level (θ)' if mode == 'theta' else 'Success Probability P(correct)'
    cbar.set_label(label, fontsize=12)
    
    # Configure plot title and axis labels
    title = f'Student Ability Level' if mode == 'theta' else 'Per-KC Success Probability'
    total_info = f" (showing first {display_timesteps} of {n_timesteps} timesteps)" if display_timesteps < n_timesteps else ""
    ax1.set_title(f'{title}\nStudent {student_idx} - Per-KC Evolution{total_info}', fontsize=14, pad=20)
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
    
    # Generate bottom subplot with correctness indicators (limited to display_timesteps)
    x_positions = np.linspace(0.5, display_timesteps - 0.5, display_timesteps)
    
    # Extract student-specific data
    student_questions = questions[student_idx] if student_idx < len(questions) else []
    student_correctness = correctness[student_idx] if student_idx < len(correctness) else []
    
    for t in range(min(display_timesteps, len(student_questions))):
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


def visualize_from_data(data_path, output_dir, student_idx=0, dataset_name=None, max_steps=100):
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
        kc_thetas = per_kc['kc_thetas']
        questions = per_kc['q_seqs']
        correctness = per_kc['correct_seqs']
        probabilities = per_kc.get('probability_data', [])
        
        if kc_thetas:
            # Generate theta heatmap visualization
            theta_path = os.path.join(output_dir, f"per_kc_theta_student_{student_idx}.png")
            plot_per_kc_heatmap(kc_thetas, questions, correctness, student_idx, 'theta', None, theta_path, max_steps)
            created_files.append(f"per_kc_theta_student_{student_idx}.png")
            
            # Generate probability heatmap visualization
            if probabilities:
                prob_path = os.path.join(output_dir, f"per_kc_probability_student_{student_idx}.png")
                plot_per_kc_heatmap(probabilities, questions, correctness, student_idx, 'probability', None, prob_path, max_steps)
                created_files.append(f"per_kc_probability_student_{student_idx}.png")
    
    # Generate global visualizations
    if 'global_data' in data:
        global_data = data['global_data']
        
        # Generate global theta heatmap
        global_path = os.path.join(output_dir, "global_theta_heatmap.png")
        plot_global_heatmap(global_data, global_path, max_steps)
        created_files.append("global_theta_heatmap.png")
        
        # Generate beta distribution plot
        beta_path = os.path.join(output_dir, "beta_distribution.png")
        plot_beta_distribution(global_data, beta_path)
        created_files.append("beta_distribution.png")
        
        # Generate alpha distribution plot
        alpha_path = os.path.join(output_dir, "alpha_distribution.png")
        plot_alpha_distribution(global_data, alpha_path)
        created_files.append("alpha_distribution.png")
        
        # Generate per-KC distributions if Q-matrix is available  
        if 'q_to_kc' in data and 'kc_names' in data and data['q_to_kc']:
            # Generate per-KC beta distributions
            per_kc_beta_path = os.path.join(output_dir, "per_kc_beta_distributions.png")
            plot_per_kc_beta_distributions(global_data, data['q_to_kc'], data['kc_names'], per_kc_beta_path)
            created_files.append("per_kc_beta_distributions.png")
            
            # Generate per-KC alpha distributions  
            per_kc_alpha_path = os.path.join(output_dir, "per_kc_alpha_distributions.png")
            plot_per_kc_alpha_distributions(global_data, data['q_to_kc'], data['kc_names'], per_kc_alpha_path)
            created_files.append("per_kc_alpha_distributions.png")
    
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
    parser.add_argument('--max_timesteps', type=int, default=100, help='Maximum timesteps to display in plots (optional, default: 100)')
    parser.add_argument('--load_data', type=str, help='Load saved data file')
    parser.add_argument('--save_data', type=str, help='Save extracted data file')
    
    args = parser.parse_args()
    
    # Handle visualization from saved data
    if args.load_data:
        return visualize_from_data(args.load_data, args.output_dir, args.student_idx, max_steps=args.max_timesteps)
    
    # Handle model-based visualization
    if not args.checkpoint or not args.config:
        print("Error: --checkpoint and --config required when not using --load_data")
        return
    
    # Load model configuration and checkpoint
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    model = load_model(args.checkpoint, config)
    
    # Initialize test data loader with dataset-specific sequence length
    try:
        # Use dataset-specific max sequence lengths for visualization
        dataset_seq_lens = {
            'assist2009_updated': 1200,  # Max: 1,146
            'STATICS': 1200,             # Max: 1,162  
            'assist2015': 650,           # Max: 618
            'fsaif1tof3': 700,           # Max: 668
            'synthetic': 50              # Max: 50 (uniform)
        }
        
        dataset_name = config.get('dataset_name', 'unknown')
        max_seq_len = dataset_seq_lens.get(dataset_name, config.get('seq_len', 1000))
        
        print(f"Using dataset-specific seq_len={max_seq_len} for {dataset_name} visualization")
        
        _, _, test_dataset = create_datasets(
            data_dir=config['data_dir'],
            dataset_name=config['dataset_name'],
            seq_len=max_seq_len,  # Use dataset-specific seq_len
            n_questions=config['n_questions'],
            k_fold=config['k_fold'],
            fold_idx=config['fold_idx']
        )
        test_loader = create_dataloader(test_dataset, batch_size=1, shuffle=False)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Extract data from model with full timesteps (limited students for faster processing)
    global_data, per_kc_data = extract_data(model, test_loader, max_students=10, max_timesteps=None)
    
    # Create figs/dataset_name structure
    dataset_name = config.get('dataset_name', 'unknown')
    if not args.output_dir.startswith('figs/'):
        args.output_dir = os.path.join('figs', dataset_name)
    
    os.makedirs(args.output_dir, exist_ok=True)
    created_files = []
    
    # Add Q-matrix info to data for per-KC visualizations
    data_to_save = {
        'global_data': global_data, 
        'dataset_name': dataset_name,
        'q_to_kc': getattr(model, 'q_to_kc', {}),
        'kc_names': getattr(model, 'kc_names', {}),
        'per_kc_mode': getattr(model, 'per_kc_mode', False)
    }
    
    # Generate global visualizations
    if len(global_data['student_abilities']) > 0:
        global_path = os.path.join(args.output_dir, "global_theta_heatmap.png")
        plot_global_heatmap(global_data, global_path, args.max_timesteps)
        created_files.append("global_theta_heatmap.png")
        
        beta_path = os.path.join(args.output_dir, "beta_distribution.png")
        plot_beta_distribution(global_data, beta_path)
        created_files.append("beta_distribution.png")
        
        # Generate alpha distribution plot
        alpha_path = os.path.join(args.output_dir, "alpha_distribution.png")
        plot_alpha_distribution(global_data, alpha_path)
        created_files.append("alpha_distribution.png")
        
        # Generate per-KC distributions if Q-matrix is available
        per_kc_mode = getattr(model, 'per_kc_mode', False)
        has_q_to_kc = hasattr(model, 'q_to_kc')
        q_to_kc_data = getattr(model, 'q_to_kc', {})
        
        print(f"Per-KC check: per_kc_mode={per_kc_mode}, has_q_to_kc={has_q_to_kc}, q_to_kc_len={len(q_to_kc_data)}")
        
        if per_kc_mode and has_q_to_kc and q_to_kc_data:
            print(f"Generating per-KC distributions for {len(q_to_kc_data)} questions")
            # Generate per-KC beta distributions
            per_kc_beta_path = os.path.join(args.output_dir, "per_kc_beta_distributions.png")
            plot_per_kc_beta_distributions(global_data, model.q_to_kc, model.kc_names, per_kc_beta_path)
            created_files.append("per_kc_beta_distributions.png")
            
            # Generate per-KC alpha distributions  
            per_kc_alpha_path = os.path.join(args.output_dir, "per_kc_alpha_distributions.png")
            plot_per_kc_alpha_distributions(global_data, model.q_to_kc, model.kc_names, per_kc_alpha_path)
            created_files.append("per_kc_alpha_distributions.png")
        else:
            print(f"Skipping per-KC distributions - conditions not met")
    
    # Generate per-knowledge component visualizations
    if model.per_kc_mode and per_kc_data['kc_thetas']:
        data_to_save['per_kc_data'] = per_kc_data
        
        theta_path = os.path.join(args.output_dir, f"per_kc_theta_student_{args.student_idx}.png")
        plot_per_kc_heatmap(
            per_kc_data['kc_thetas'],
            per_kc_data['q_seqs'],
            per_kc_data['correct_seqs'],
            args.student_idx, 'theta', model.kc_names, theta_path, args.max_timesteps
        )
        created_files.append(f"per_kc_theta_student_{args.student_idx}.png")
        
        if per_kc_data['probability_data']:
            prob_path = os.path.join(args.output_dir, f"per_kc_probability_student_{args.student_idx}.png")
            plot_per_kc_heatmap(
                per_kc_data['probability_data'],
                per_kc_data['q_seqs'],
                per_kc_data['correct_seqs'],
                args.student_idx, 'probability', model.kc_names, prob_path, args.max_timesteps
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


def plot_per_kc_beta_distributions(data, q_to_kc, kc_names, save_path=None):
    """
    Plot per-KC beta (difficulty) distributions.
    
    Args:
        data (dict): Data dictionary containing 'item_difficulties'
        q_to_kc (dict): Question to KC mapping
        kc_names (dict): KC ID to name mapping
        save_path (str, optional): Path to save the plot
    """
    if 'item_difficulties' not in data or len(data['item_difficulties']) == 0:
        print("No beta parameters found for per-KC analysis")
        return
    
    # Group difficulties by KC
    kc_difficulties = {}
    for q_idx, difficulty in enumerate(data['item_difficulties']):
        q_id = q_idx + 1  # Assuming 1-indexed questions
        if q_id in q_to_kc:
            for kc_id in q_to_kc[q_id]:
                if kc_id not in kc_difficulties:
                    kc_difficulties[kc_id] = []
                kc_difficulties[kc_id].append(difficulty)
    
    if not kc_difficulties:
        print("No KC mappings found for beta analysis")
        return
    
    # Create subplots
    n_kcs = len(kc_difficulties)
    cols = min(3, n_kcs)
    rows = (n_kcs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_kcs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Per-KC Beta (Difficulty) Distributions', fontsize=16, fontweight='bold')
    
    for idx, (kc_id, difficulties) in enumerate(sorted(kc_difficulties.items())):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        kc_name = kc_names.get(kc_id, f"KC_{kc_id}")
        ax.hist(difficulties, bins=min(20, len(difficulties)), alpha=0.7, edgecolor='black')
        ax.set_title(f'{kc_name} (n={len(difficulties)})')
        ax.set_xlabel('Beta (Difficulty)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_diff = np.mean(difficulties)
        ax.axvline(mean_diff, color='red', linestyle='--', label=f'Mean: {mean_diff:.3f}')
        ax.legend()
    
    # Hide empty subplots
    for idx in range(n_kcs, rows * cols):
        if rows > 1:
            row, col = idx // cols, idx % cols
            axes[row, col].set_visible(False)
        elif cols > 1:
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-KC beta distributions saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_kc_alpha_distributions(data, q_to_kc, kc_names, save_path=None):
    """
    Plot per-KC alpha (discrimination) distributions.
    
    Args:
        data (dict): Data dictionary containing 'discrimination_params'
        q_to_kc (dict): Question to KC mapping  
        kc_names (dict): KC ID to name mapping
        save_path (str, optional): Path to save the plot
    """
    if 'discrimination_params' not in data or len(data['discrimination_params']) == 0:
        print("No alpha parameters found for per-KC analysis")
        return
    
    # Group discrimination params by KC
    kc_alphas = {}
    for q_idx, alpha in enumerate(data['discrimination_params']):
        q_id = q_idx + 1  # Assuming 1-indexed questions
        if q_id in q_to_kc:
            for kc_id in q_to_kc[q_id]:
                if kc_id not in kc_alphas:
                    kc_alphas[kc_id] = []
                kc_alphas[kc_id].append(alpha)
    
    if not kc_alphas:
        print("No KC mappings found for alpha analysis")
        return
    
    # Create subplots
    n_kcs = len(kc_alphas)
    cols = min(3, n_kcs)
    rows = (n_kcs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_kcs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Per-KC Alpha (Discrimination) Distributions', fontsize=16, fontweight='bold')
    
    for idx, (kc_id, alphas) in enumerate(sorted(kc_alphas.items())):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        kc_name = kc_names.get(kc_id, f"KC_{kc_id}")
        ax.hist(alphas, bins=min(20, len(alphas)), alpha=0.7, edgecolor='black', color='purple')
        ax.set_title(f'{kc_name} (n={len(alphas)})')
        ax.set_xlabel('Alpha (Discrimination)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_alpha = np.mean(alphas)
        ax.axvline(mean_alpha, color='red', linestyle='--', label=f'Mean: {mean_alpha:.3f}')
        ax.legend()
    
    # Hide empty subplots
    for idx in range(n_kcs, rows * cols):
        if rows > 1:
            row, col = idx // cols, idx % cols
            axes[row, col].set_visible(False)
        elif cols > 1:
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-KC alpha distributions saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    main()