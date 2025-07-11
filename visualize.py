#!/usr/bin/env python3
"""
Unified Visualization Script for Deep-IRT Model

Automatically detects per-KC mode and generates appropriate visualizations:
- Global theta/beta plots for standard models
- Per-KC continuous heatmaps for models with Q-matrix
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
from collections import defaultdict
from tqdm import tqdm

from models.deep_irt_model import UnifiedDeepIRTModel
from data.dataloader import create_datasets, create_dataloader


def load_trained_model(checkpoint_path, config):
    """Load trained model from checkpoint."""
    model = UnifiedDeepIRTModel(
        n_questions=config['n_questions'],
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
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def extract_global_data(model, data_loader, max_students=50, max_timesteps=30):
    """Extract global theta and beta data and save for reuse."""
    model.eval()
    
    all_student_abilities = []
    all_item_difficulties = []
    student_ids = []
    timesteps = []
    question_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting global data", leave=False)):
            if batch_idx >= max_students:
                break
                
            q_data = batch['q_data']
            qa_data = batch['qa_data']
            targets = batch['target']
            
            predictions, student_abilities, item_difficulties, z_values, kc_info = model(q_data, qa_data)
            
            batch_size, seq_len = q_data.shape
            seq_len = min(seq_len, max_timesteps)
            
            for student_idx in range(batch_size):
                for t in range(seq_len):
                    if q_data[student_idx, t].item() > 0:
                        all_student_abilities.append(student_abilities[student_idx, t].item())
                        all_item_difficulties.append(item_difficulties[student_idx, t].item())
                        student_ids.append(batch_idx * batch_size + student_idx)
                        timesteps.append(t)
                        question_ids.append(q_data[student_idx, t].item())
    
    return {
        'student_abilities': np.array(all_student_abilities),
        'item_difficulties': np.array(all_item_difficulties),
        'student_ids': np.array(student_ids),
        'timesteps': np.array(timesteps),
        'question_ids': np.array(question_ids)
    }


def extract_per_kc_data(model, data_loader, max_students=20, max_timesteps=50):
    """Extract per-KC continuous data and save for reuse."""
    model.eval()
    
    all_kc_thetas_list = []
    question_sequences = []
    correctness_sequences = []
    beta_parameters = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting per-KC data", leave=False)):
            if batch_idx >= max_students:
                break
                
            q_data = batch['q_data']
            qa_data = batch['qa_data']
            targets = batch['target']
            
            predictions, student_abilities, item_difficulties, z_values, kc_info = model(q_data, qa_data)
            
            # Extract per-KC data
            if 'all_kc_thetas' in kc_info:
                batch_size, seq_len, n_kcs = kc_info['all_kc_thetas'].shape
                seq_len = min(seq_len, max_timesteps)
                
                for student_idx in range(batch_size):
                    # Get this student's KC theta evolution
                    student_kc_thetas = kc_info['all_kc_thetas'][student_idx, :seq_len, :].cpu().numpy()
                    all_kc_thetas_list.append(student_kc_thetas)
                    
                    # Get question sequence and correctness
                    student_questions = []
                    student_correctness = []
                    student_betas = []
                    for t in range(seq_len):
                        q_id = q_data[student_idx, t].item()
                        student_questions.append(q_id)
                        student_betas.append(item_difficulties[student_idx, t].cpu().item())
                        if q_id > 0 and targets[student_idx, t].item() >= 0:
                            student_correctness.append(int(targets[student_idx, t].item()))
                        else:
                            student_correctness.append(0)
                    
                    question_sequences.append(student_questions)
                    correctness_sequences.append(student_correctness)
                    beta_parameters.append(student_betas)
    
    return all_kc_thetas_list, question_sequences, correctness_sequences, beta_parameters


def create_global_theta_heatmap(data, save_path=None):
    """Create heatmap of global student abilities over time."""
    df = pd.DataFrame({
        'Student': data['student_ids'],
        'Timestep': data['timesteps'],
        'Theta': data['student_abilities']
    })
    
    theta_matrix = df.pivot_table(values='Theta', index='Student', columns='Timestep', aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(theta_matrix, cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Student Ability (θ)'})
    plt.title('Student Ability (θ) Evolution Over Time\nDeep-IRT Model (Global)', fontsize=14)
    plt.xlabel('Time Step')
    plt.ylabel('Student ID')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def create_per_kc_continuous_heatmap(all_kc_thetas_list, question_sequences, correctness_sequences, 
                                   model, student_idx=0, save_path=None):
    """Create continuous per-KC heatmap with properly aligned x-axis."""
    
    if student_idx >= len(all_kc_thetas_list):
        print(f"Student {student_idx} not available. Using student 0.")
        student_idx = 0
    
    kc_thetas = all_kc_thetas_list[student_idx]  # (timesteps, n_kcs)
    questions = question_sequences[student_idx]
    correctness = correctness_sequences[student_idx]
    
    n_timesteps, n_kcs = kc_thetas.shape
    
    # Select top KCs with most variation
    kc_variations = np.std(kc_thetas, axis=0)
    top_kcs = np.argsort(kc_variations)[-20:]  # Top 20 most dynamic KCs
    
    # Create the plot with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), 
                                  gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.05},
                                  sharex=True)
    
    # Main heatmap - KC on y-axis, timesteps on x-axis
    theta_subset = kc_thetas[:, top_kcs].T  # (n_selected_kcs, n_timesteps)
    
    im = ax1.imshow(theta_subset, cmap='RdYlGn', aspect='auto', 
                   vmin=-1.0, vmax=1.0, interpolation='bilinear',
                   extent=[0, n_timesteps, 0, len(top_kcs)], origin='lower')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Student Ability Level (θ)', fontsize=12)
    
    # Labels for main plot
    ax1.set_title(f'Student Ability Level Estimated by the Deep-IRT model\n'
                 f'Student {student_idx} - Continuous Per-KC Evolution', 
                 fontsize=14, pad=20)
    ax1.set_ylabel('Knowledge Concepts', fontsize=12)
    
    # Y-axis labels (KC names) - adjust for proper positioning
    y_ticks = np.arange(len(top_kcs)) + 0.5  # Center labels on heatmap cells
    if hasattr(model, 'kc_names') and model.kc_names:
        y_labels = [model.kc_names.get(kc, f'KC_{kc+1}') for kc in top_kcs]
    else:
        y_labels = [f'KC_{kc+1}' for kc in top_kcs]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels, fontsize=8)
    
    # Remove x-axis labels from main plot since we're sharing
    ax1.set_xticks([])
    
    # Bottom subplot: Question correctness indicators with circles
    # Get the actual x-axis position range from the main heatmap (excluding colorbar)
    main_bbox = ax1.get_position()
    
    # Create circles for correct/incorrect with proper alignment
    x_positions = np.linspace(0.5, n_timesteps - 0.5, n_timesteps)  # Center circles in timesteps
    
    for t in range(n_timesteps):
        q_id = questions[t] if t < len(questions) else 0
        correct = correctness[t] if t < len(correctness) else 0
        
        if q_id > 0:
            if correct == 1:
                # Filled green circle for correct
                ax2.scatter(x_positions[t], 0.5, s=120, c='green', alpha=0.8, 
                           edgecolors='black', linewidth=1)
            else:
                # Empty red circle for incorrect  
                ax2.scatter(x_positions[t], 0.5, s=120, facecolors='none', 
                           edgecolors='red', linewidth=2)
        # No marker for no question
    
    # Align x-axis exactly with main plot
    ax2.set_xlim(ax1.get_xlim())  # Match main plot x-limits exactly
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Questions (Timeline)', fontsize=12)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    
    # Force same x-axis positioning as main plot
    ax2.set_position([main_bbox.x0, ax2.get_position().y0, 
                     main_bbox.width, ax2.get_position().height])
    
    # Legend for attempt trajectory with circle symbols
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markeredgecolor='black', markersize=10, label='● correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
               markeredgecolor='red', markersize=10, markeredgewidth=2, label='○ incorrect')
    ]
    ax2.legend(handles=legend_elements, loc='center right', 
              bbox_to_anchor=(1.05, 0.5), fontsize=10)
    
    # Add grid to main plot
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Final alignment
    fig.align_xlabels()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def create_per_kc_probability_heatmap(probability_data, question_sequences, correctness_sequences, 
                                     model, student_idx=0, save_path=None):
    """Create per-KC probability heatmap: p_t = sigmoid(theta_tj - beta_j)."""
    
    if student_idx >= len(probability_data):
        print(f"Student {student_idx} not available. Using student 0.")
        student_idx = 0
    
    kc_probs = probability_data[student_idx]  # (timesteps, n_kcs)
    questions = question_sequences[student_idx]
    correctness = correctness_sequences[student_idx]
    
    n_timesteps, n_kcs = kc_probs.shape
    
    # Select top KCs with most variation
    kc_variations = np.std(kc_probs, axis=0)
    top_kcs = np.argsort(kc_variations)[-20:]  # Top 20 most dynamic KCs
    
    # Create the plot with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), 
                                  gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.05},
                                  sharex=True)
    
    # Main heatmap - KC on y-axis, timesteps on x-axis
    prob_subset = kc_probs[:, top_kcs].T  # (n_selected_kcs, n_timesteps)
    
    im = ax1.imshow(prob_subset, cmap='RdYlGn', aspect='auto', 
                   vmin=0.0, vmax=1.0, interpolation='bilinear',
                   extent=[0, n_timesteps, 0, len(top_kcs)], origin='lower')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Success Probability P(correct)', fontsize=12)
    
    # Labels for main plot
    ax1.set_title(f'Per-KC Success Probability: P(correct) = sigmoid(θ_KC - β)\\n'
                 f'Student {student_idx} - Continuous Probability Evolution', 
                 fontsize=14, pad=20)
    ax1.set_ylabel('Knowledge Concepts', fontsize=12)
    
    # Y-axis labels (KC names) - adjust for proper positioning
    y_ticks = np.arange(len(top_kcs)) + 0.5  # Center labels on heatmap cells
    if hasattr(model, 'kc_names') and model.kc_names:
        y_labels = [model.kc_names.get(kc, f'KC_{kc+1}') for kc in top_kcs]
    else:
        y_labels = [f'KC_{kc+1}' for kc in top_kcs]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels, fontsize=8)
    
    # Remove x-axis labels from main plot since we're sharing
    ax1.set_xticks([])
    
    # Bottom subplot: Question correctness indicators
    question_colors = []
    for t in range(n_timesteps):
        q_id = questions[t] if t < len(questions) else 0
        correct = correctness[t] if t < len(correctness) else 0
        
        if q_id > 0:
            if correct == 1:
                color = 'green'  # Correct
            else:
                color = 'red'    # Incorrect
        else:
            color = 'lightgray'  # No question
        
        question_colors.append(color)
    
    # Create question correctness trajectory with circles  
    # Get the actual x-axis position range from the main heatmap
    main_bbox = ax1.get_position()
    
    x_positions = np.linspace(0.5, n_timesteps - 0.5, n_timesteps)  # Center circles in timesteps
    
    for t in range(n_timesteps):
        q_id = questions[t] if t < len(questions) else 0
        correct = correctness[t] if t < len(correctness) else 0
        
        if q_id > 0:
            if correct == 1:
                # Filled green circle for correct
                ax2.scatter(x_positions[t], 0.5, s=120, c='green', alpha=0.8, 
                           edgecolors='black', linewidth=1)
            else:
                # Empty red circle for incorrect  
                ax2.scatter(x_positions[t], 0.5, s=120, facecolors='none', 
                           edgecolors='red', linewidth=2)
    
    # Align x-axis exactly with main plot
    ax2.set_xlim(ax1.get_xlim())  # Match main plot x-limits exactly
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Questions (Timeline)', fontsize=12)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    
    # Force same x-axis positioning as main plot
    ax2.set_position([main_bbox.x0, ax2.get_position().y0, 
                     main_bbox.width, ax2.get_position().height])
    
    # Legend for attempt trajectory with circle symbols
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markeredgecolor='black', markersize=10, label='● correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
               markeredgecolor='red', markersize=10, markeredgewidth=2, label='○ incorrect')
    ]
    ax2.legend(handles=legend_elements, loc='center right', 
              bbox_to_anchor=(1.05, 0.5), fontsize=10)
    
    # Add grid to main plot
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Final alignment
    fig.align_xlabels()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def create_per_kc_probability_heatmap_simple(probability_data, question_sequences, correctness_sequences, 
                                            student_idx=0, save_path=None):
    """Simplified per-KC probability heatmap without model dependency."""
    if student_idx >= len(probability_data):
        print(f"Student {student_idx} not available. Using student 0.")
        student_idx = 0
    
    kc_probs = probability_data[student_idx]
    questions = question_sequences[student_idx]
    correctness = correctness_sequences[student_idx]
    
    n_timesteps, n_kcs = kc_probs.shape
    
    # Select top KCs with most variation
    kc_variations = np.std(kc_probs, axis=0)
    top_kcs = np.argsort(kc_variations)[-20:]
    
    # Create the plot with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), 
                                  gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.05},
                                  sharex=True)
    
    # Main heatmap with proper extent
    prob_subset = kc_probs[:, top_kcs].T
    
    im = ax1.imshow(prob_subset, cmap='RdYlGn', aspect='auto', 
                   vmin=0.0, vmax=1.0, interpolation='bilinear',
                   extent=[0, n_timesteps, 0, len(top_kcs)], origin='lower')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Success Probability P(correct)', fontsize=12)
    
    # Labels
    ax1.set_title(f'Per-KC Success Probability (Saved Data)\\nStudent {student_idx} - Probability Evolution', 
                 fontsize=14, pad=20)
    ax1.set_ylabel('Knowledge Concepts', fontsize=12)
    
    # Y-axis labels - adjust for proper positioning
    y_ticks = np.arange(len(top_kcs)) + 0.5  # Center labels on heatmap cells
    y_labels = [f'KC_{kc+1}' for kc in top_kcs]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels, fontsize=8)
    ax1.set_xticks([])  # No x-ticks on main plot since we're sharing
    
    # Bottom subplot: Question correctness
    question_colors = []
    for t in range(n_timesteps):
        q_id = questions[t] if t < len(questions) else 0
        correct = correctness[t] if t < len(correctness) else 0
        
        if q_id > 0:
            color = 'green' if correct == 1 else 'red'
        else:
            color = 'lightgray'
        question_colors.append(color)
    
    # Create question correctness trajectory with circles
    # Get the actual x-axis position range from the main heatmap  
    main_bbox = ax1.get_position()
    
    x_positions = np.linspace(0.5, n_timesteps - 0.5, n_timesteps)  # Center circles in timesteps
    
    for t in range(n_timesteps):
        q_id = questions[t] if t < len(questions) else 0
        correct = correctness[t] if t < len(correctness) else 0
        
        if q_id > 0:
            if correct == 1:
                # Filled green circle for correct
                ax2.scatter(x_positions[t], 0.5, s=120, c='green', alpha=0.8, 
                           edgecolors='black', linewidth=1)
            else:
                # Empty red circle for incorrect  
                ax2.scatter(x_positions[t], 0.5, s=120, facecolors='none', 
                           edgecolors='red', linewidth=2)
    
    # Align x-axis exactly with main plot
    ax2.set_xlim(ax1.get_xlim())  # Match main plot x-limits exactly
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Questions (Timeline)', fontsize=12)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    
    # Force same x-axis positioning as main plot
    ax2.set_position([main_bbox.x0, ax2.get_position().y0, 
                     main_bbox.width, ax2.get_position().height])
    
    # Legend with circle symbols
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markeredgecolor='black', markersize=10, label='● correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
               markeredgecolor='red', markersize=10, markeredgewidth=2, label='○ incorrect')
    ]
    ax2.legend(handles=legend_elements, loc='center right', 
              bbox_to_anchor=(1.05, 0.5), fontsize=10)
    
    ax1.grid(True, alpha=0.3, axis='x')
    fig.align_xlabels()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def create_beta_distribution(data, save_path=None):
    """Create item difficulty distribution plot."""
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
    else:
        plt.show()
    plt.close()


def save_extracted_data(data, filepath):
    """Save extracted theta/beta data for reuse."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved extracted data to {filepath}")


def load_extracted_data(filepath):
    """Load previously saved theta/beta data."""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded extracted data from {filepath}")
        return data
    return None


def visualize_from_saved_data(data_path, output_dir, student_idx=0):
    """Create visualizations from saved data without model inference."""
    data = load_extracted_data(data_path)
    if data is None:
        print(f"No saved data found at {data_path}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    files_created = []
    
    # Check data format and create appropriate visualizations
    if 'per_kc_data' in data:
        per_kc_data = data['per_kc_data']
        all_kc_thetas_list = per_kc_data['all_kc_thetas_list']
        question_sequences = per_kc_data['question_sequences']
        correctness_sequences = per_kc_data['correctness_sequences']
        beta_parameters = per_kc_data.get('beta_parameters', [])
        probability_data = per_kc_data.get('probability_data', [])
        
        if all_kc_thetas_list:
            per_kc_path = os.path.join(output_dir, f"per_kc_heatmap_student_{student_idx}.png")
            # Create simplified version without model reference
            create_per_kc_heatmap_simple(all_kc_thetas_list, question_sequences, 
                                       correctness_sequences, student_idx, per_kc_path)
            files_created.append(f"per_kc_heatmap_student_{student_idx}.png")
            
            # Calculate probability data if not available but we have theta and beta
            if not probability_data and beta_parameters and all_kc_thetas_list:
                print("Calculating probabilities from existing theta and beta data...")
                probability_data = []
                for student_idx_calc in range(len(all_kc_thetas_list)):
                    kc_thetas = all_kc_thetas_list[student_idx_calc]  # (timesteps, n_kcs)
                    student_betas = beta_parameters[student_idx_calc] if student_idx_calc < len(beta_parameters) else []
                    student_probs = []
                    
                    for t in range(kc_thetas.shape[0]):
                        if t < len(student_betas):
                            beta_val = student_betas[t]
                            # Calculate per-KC probabilities: p_t = sigmoid(theta_tj - beta_j)
                            import torch
                            kc_probs = torch.sigmoid(torch.tensor(kc_thetas[t, :]) - beta_val).numpy()
                            student_probs.append(kc_probs)
                        else:
                            # Default probabilities if no beta available
                            student_probs.append(np.full(kc_thetas.shape[1], 0.5))
                    
                    probability_data.append(np.array(student_probs))  # (timesteps, n_kcs)
            
            # Create probability heatmap
            if probability_data:
                prob_path = os.path.join(output_dir, f"per_kc_probability_heatmap_student_{student_idx}.png")
                create_per_kc_probability_heatmap_simple(probability_data, question_sequences, 
                                                       correctness_sequences, student_idx, prob_path)
                files_created.append(f"per_kc_probability_heatmap_student_{student_idx}.png")
    
    if 'global_data' in data:
        global_data = data['global_data']
        global_path = os.path.join(output_dir, "global_theta_heatmap.png")
        create_global_theta_heatmap(global_data, save_path=global_path)
        files_created.append("global_theta_heatmap.png")
        
        beta_path = os.path.join(output_dir, "beta_distribution.png")
        create_beta_distribution(global_data, save_path=beta_path)
        files_created.append("beta_distribution.png")
    
    print(f"Created {len(files_created)} visualizations from saved data:")
    for file in files_created:
        print(f"  - {file}")
    
    return True


def create_per_kc_heatmap_simple(all_kc_thetas_list, question_sequences, correctness_sequences, 
                                student_idx=0, save_path=None):
    """Simplified per-KC heatmap without model dependency."""
    if student_idx >= len(all_kc_thetas_list):
        print(f"Student {student_idx} not available. Using student 0.")
        student_idx = 0
    
    kc_thetas = all_kc_thetas_list[student_idx]
    questions = question_sequences[student_idx]
    correctness = correctness_sequences[student_idx]
    
    n_timesteps, n_kcs = kc_thetas.shape
    
    # Select top KCs with most variation
    kc_variations = np.std(kc_thetas, axis=0)
    top_kcs = np.argsort(kc_variations)[-20:]
    
    # Create the plot with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), 
                                  gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.05},
                                  sharex=True)
    
    # Main heatmap with proper extent
    theta_subset = kc_thetas[:, top_kcs].T
    
    im = ax1.imshow(theta_subset, cmap='RdYlGn', aspect='auto', 
                   vmin=-1.0, vmax=1.0, interpolation='bilinear',
                   extent=[0, n_timesteps, 0, len(top_kcs)], origin='lower')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Student Ability Level (θ)', fontsize=12)
    
    # Labels
    ax1.set_title(f'Student Ability Level (Saved Data)\nStudent {student_idx} - Per-KC Evolution', 
                 fontsize=14, pad=20)
    ax1.set_ylabel('Knowledge Concepts', fontsize=12)
    
    # Y-axis labels - adjust for proper positioning
    y_ticks = np.arange(len(top_kcs)) + 0.5  # Center labels on heatmap cells
    y_labels = [f'KC_{kc+1}' for kc in top_kcs]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels, fontsize=8)
    ax1.set_xticks([])  # No x-ticks on main plot since we're sharing
    
    # Bottom subplot: Question correctness
    question_colors = []
    for t in range(n_timesteps):
        q_id = questions[t] if t < len(questions) else 0
        correct = correctness[t] if t < len(correctness) else 0
        
        if q_id > 0:
            color = 'green' if correct == 1 else 'red'
        else:
            color = 'lightgray'
        question_colors.append(color)
    
    # Create question correctness trajectory with circles
    # Get the actual x-axis position range from the main heatmap  
    main_bbox = ax1.get_position()
    
    x_positions = np.linspace(0.5, n_timesteps - 0.5, n_timesteps)  # Center circles in timesteps
    
    for t in range(n_timesteps):
        q_id = questions[t] if t < len(questions) else 0
        correct = correctness[t] if t < len(correctness) else 0
        
        if q_id > 0:
            if correct == 1:
                # Filled green circle for correct
                ax2.scatter(x_positions[t], 0.5, s=120, c='green', alpha=0.8, 
                           edgecolors='black', linewidth=1)
            else:
                # Empty red circle for incorrect  
                ax2.scatter(x_positions[t], 0.5, s=120, facecolors='none', 
                           edgecolors='red', linewidth=2)
    
    # Align x-axis exactly with main plot
    ax2.set_xlim(ax1.get_xlim())  # Match main plot x-limits exactly
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Questions (Timeline)', fontsize=12)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    
    # Force same x-axis positioning as main plot
    ax2.set_position([main_bbox.x0, ax2.get_position().y0, 
                     main_bbox.width, ax2.get_position().height])
    
    # Legend with circle symbols
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markeredgecolor='black', markersize=10, label='● correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
               markeredgecolor='red', markersize=10, markeredgewidth=2, label='○ incorrect')
    ]
    ax2.legend(handles=legend_elements, loc='center right', 
              bbox_to_anchor=(1.05, 0.5), fontsize=10)
    
    ax1.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize Deep-IRT Model Results')
    parser.add_argument('--checkpoint', type=str, required=False,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=False,
                        help='Path to model config JSON file')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--student_idx', type=int, default=0,
                        help='Student index for per-KC visualization')
    parser.add_argument('--load_data', type=str, default=None,
                        help='Path to saved data file for visualization without model')
    parser.add_argument('--save_data', type=str, default=None,
                        help='Path to save extracted data for reuse')
    
    args = parser.parse_args()
    
    # Check if we're loading from saved data
    if args.load_data:
        return visualize_from_saved_data(args.load_data, args.output_dir, args.student_idx)
    
    # Otherwise proceed with model-based visualization
    if not args.checkpoint or not args.config:
        print("Error: --checkpoint and --config are required when not using --load_data")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        return
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Load model
    model = load_trained_model(args.checkpoint, config)
    
    # Load test data
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    files_created = []
    
    extracted_data = {}
    
    if model.per_kc_mode:
        # Per-KC visualizations
        all_kc_thetas_list, question_sequences, correctness_sequences, beta_parameters = extract_per_kc_data(
            model, test_loader, max_students=20, max_timesteps=50
        )
        
        # Save per-KC data
        extracted_data['per_kc_data'] = {
            'all_kc_thetas_list': all_kc_thetas_list,
            'question_sequences': question_sequences,
            'correctness_sequences': correctness_sequences,
            'beta_parameters': beta_parameters
        }
        
        if all_kc_thetas_list:
            per_kc_path = os.path.join(args.output_dir, f"per_kc_continuous_heatmap_student_{args.student_idx}.png")
            create_per_kc_continuous_heatmap(
                all_kc_thetas_list, question_sequences, correctness_sequences,
                model, args.student_idx, save_path=per_kc_path
            )
            files_created.append(f"per_kc_continuous_heatmap_student_{args.student_idx}.png")
    
    # Global visualizations (always create these)
    global_data = extract_global_data(model, test_loader, max_students=50, max_timesteps=30)
    extracted_data['global_data'] = global_data
    
    if len(global_data['student_abilities']) > 0:
        global_path = os.path.join(args.output_dir, "global_theta_heatmap.png")
        create_global_theta_heatmap(global_data, save_path=global_path)
        files_created.append("global_theta_heatmap.png")
        
        beta_path = os.path.join(args.output_dir, "beta_distribution.png")
        create_beta_distribution(global_data, save_path=beta_path)
        files_created.append("beta_distribution.png")
        
        # Save extracted data if requested
        if args.save_data:
            save_extracted_data(extracted_data, args.save_data)
        
        # Summary statistics
        print(f"Created {len(files_created)} visualizations:")
        for file in files_created:
            print(f"  - {file}")
        print(f"Interactions: {len(global_data['student_abilities'])}, "
              f"Students: {len(np.unique(global_data['student_ids']))}, "
              f"Questions: {len(np.unique(global_data['question_ids']))}")
        print(f"θ: {np.mean(global_data['student_abilities']):.3f}±{np.std(global_data['student_abilities']):.3f}, "
              f"β: {np.mean(global_data['item_difficulties']):.3f}±{np.std(global_data['item_difficulties']):.3f}")
    else:
        print("No data extracted for visualization.")
    
    return True


if __name__ == "__main__":
    main()