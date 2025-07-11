"""
Unified Deep-IRT Model with Continuous Per-KC Tracking

This model supports:
1. Both data-orig and data-yeung formats
2. Optional per-KC continuous tracking when Q-matrix is available
3. Fallback to global tracking when no Q-matrix is provided
4. Continuous evolution where all KCs update at each timestep
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv
from .memory import DKVMN


class UnifiedDeepIRTModel(nn.Module):
    """
    Unified Deep-IRT model with optional continuous per-KC tracking.
    """
    
    def __init__(self, n_questions, memory_size=50, key_memory_state_dim=50, 
                 value_memory_state_dim=200, summary_vector_dim=50, 
                 q_embed_dim=None, qa_embed_dim=None, ability_scale=3.0, 
                 use_discrimination=False, dropout_rate=0.0, 
                 q_matrix_path=None, skill_mapping_path=None):
        """
        Initialize the unified Deep-IRT model.
        
        Args:
            n_questions: Number of questions in dataset
            q_matrix_path: Path to Q-matrix file (optional, enables per-KC tracking)
            skill_mapping_path: Path to skill mapping file (optional)
            Other args: Standard Deep-IRT parameters
        """
        super(UnifiedDeepIRTModel, self).__init__()
        
        # Set embedding dimensions
        if q_embed_dim is None:
            q_embed_dim = key_memory_state_dim
        if qa_embed_dim is None:
            qa_embed_dim = value_memory_state_dim
        
        self.n_questions = n_questions
        self.memory_size = memory_size
        self.key_memory_state_dim = key_memory_state_dim
        self.value_memory_state_dim = value_memory_state_dim
        self.summary_vector_dim = summary_vector_dim
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.ability_scale = ability_scale
        self.use_discrimination = use_discrimination
        self.dropout_rate = dropout_rate
        
        # Load Q-matrix and KC information if available
        self.per_kc_mode, self.q_to_kc, self.kc_names, self.n_kcs = self._load_kc_info(
            q_matrix_path, skill_mapping_path
        )
        
        # Embedding layers
        self.q_embed = nn.Embedding(n_questions + 1, q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * n_questions + 1, qa_embed_dim, padding_idx=0)
        
        # Memory network
        self.memory = DKVMN(
            memory_size=memory_size,
            key_memory_state_dim=key_memory_state_dim,
            value_memory_state_dim=value_memory_state_dim
        )
        
        # Summary vector network
        summary_input_dim = value_memory_state_dim + q_embed_dim
        self.summary_network = nn.Sequential(
            nn.Linear(summary_input_dim, summary_vector_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        
        if self.per_kc_mode:
            # Per-KC continuous tracking components
            self.kc_state_dim = min(32, summary_vector_dim // 2)
            
            # Per-KC state networks (continuous evolution)
            self.kc_state_networks = nn.ModuleDict({
                f'kc_{i}': nn.GRUCell(summary_vector_dim, self.kc_state_dim)
                for i in range(self.n_kcs)
            })
            
            # Per-KC theta estimation networks
            self.kc_theta_networks = nn.ModuleDict({
                f'kc_{i}': nn.Sequential(
                    nn.Linear(self.kc_state_dim, self.kc_state_dim // 2),
                    nn.Tanh(),
                    nn.Linear(self.kc_state_dim // 2, 1)
                ) for i in range(self.n_kcs)
            })
            
            # Cross-KC influence network
            self.cross_kc_network = nn.Sequential(
                nn.Linear(self.n_kcs * self.kc_state_dim, summary_vector_dim),
                nn.Tanh(),
                nn.Linear(summary_vector_dim, self.n_kcs * self.kc_state_dim),
                nn.Dropout(dropout_rate)
            )
            
            # Item difficulty with KC context
            self.item_difficulty_net = nn.Sequential(
                nn.Linear(q_embed_dim + self.n_kcs, summary_vector_dim),
                nn.Tanh(),
                nn.Linear(summary_vector_dim, 1)
            )
            
            # Initial KC states
            self.init_kc_states = nn.Parameter(
                torch.randn(self.n_kcs, self.kc_state_dim) * 0.1
            )
        else:
            # Global tracking components
            self.student_ability_net = nn.Sequential(
                nn.Linear(summary_vector_dim, summary_vector_dim),
                nn.Tanh(),
                nn.Linear(summary_vector_dim, 1)
            )
            
            self.item_difficulty_net = nn.Sequential(
                nn.Linear(q_embed_dim, summary_vector_dim),
                nn.Tanh(),
                nn.Linear(summary_vector_dim, 1)
            )
        
        # IRT predictor
        if use_discrimination:
            self.discrimination = nn.Parameter(torch.ones(1))
        else:
            self.discrimination = None
        
        # Initial value memory
        self.init_value_memory = nn.Parameter(
            torch.randn(memory_size, value_memory_state_dim) * 0.1
        )
        
        self._init_weights()
    
    def _load_kc_info(self, q_matrix_path, skill_mapping_path):
        """Load Q-matrix and skill mapping information."""
        if q_matrix_path is None or not os.path.exists(q_matrix_path):
            # No Q-matrix available, use global mode
            return False, {}, {}, 0
        
        q_to_kc = {}
        kc_names = {}
        
        try:
            # Load Q-matrix
            with open(q_matrix_path, 'r') as f:
                for q_idx, line in enumerate(f, 1):
                    kc_vector = [int(x) for x in line.strip().split(',')]
                    kcs = [i for i, val in enumerate(kc_vector) if val == 1]
                    q_to_kc[q_idx] = kcs if kcs else [0]  # Default to KC 0 if none
            
            n_kcs = len(kc_vector)
            
            # Load skill mapping if available
            if skill_mapping_path and os.path.exists(skill_mapping_path):
                with open(skill_mapping_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            kc_id = int(parts[0])
                            kc_name = parts[1]
                            if kc_id < n_kcs:
                                kc_names[kc_id] = kc_name
            
            # Fill in missing KC names
            for i in range(n_kcs):
                if i not in kc_names:
                    kc_names[i] = f"KC_{i+1}"
            
            print(f"Loaded Q-matrix: {len(q_to_kc)} questions, {n_kcs} KCs")
            return True, q_to_kc, kc_names, n_kcs
            
        except Exception as e:
            print(f"Warning: Could not load Q-matrix from {q_matrix_path}: {e}")
            return False, {}, {}, 0
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)
        
        # Initialize parameters
        nn.init.kaiming_normal_(self.init_value_memory)
        
        # Initialize summary network
        for module in self.summary_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        # Initialize per-KC components if available
        if self.per_kc_mode:
            nn.init.kaiming_normal_(self.init_kc_states)
    
    def get_kc_encoding(self, question_ids, device):
        """Get one-hot encoding for KCs of given questions."""
        if not self.per_kc_mode:
            return None
        
        batch_size = len(question_ids)
        kc_encoding = torch.zeros(batch_size, self.n_kcs, device=device)
        
        for i, q_id in enumerate(question_ids):
            q_id = q_id.item() if torch.is_tensor(q_id) else q_id
            kcs = self.q_to_kc.get(q_id, [0])
            for kc in kcs:
                if 0 <= kc < self.n_kcs:
                    kc_encoding[i, kc] = 1.0
        
        return kc_encoding
    
    def irt_prediction(self, student_ability, item_difficulty):
        """Compute IRT prediction."""
        if self.use_discrimination and self.discrimination is not None:
            z_value = self.discrimination * (self.ability_scale * student_ability - item_difficulty)
        else:
            z_value = self.ability_scale * student_ability - item_difficulty
        
        prediction = torch.sigmoid(z_value)
        return prediction, z_value
    
    def forward(self, q_data, qa_data, target_mask=None):
        """
        Forward pass through the model.
        
        Returns:
            predictions: Shape (batch_size, seq_len)
            student_abilities: Shape (batch_size, seq_len) or (batch_size, seq_len, n_kcs) for per-KC
            item_difficulties: Shape (batch_size, seq_len)
            z_values: Shape (batch_size, seq_len)
            kc_info: Dict with KC information (if per_kc_mode)
        """
        batch_size, seq_len = q_data.shape
        device = q_data.device
        
        # Safety check
        q_data = torch.clamp(q_data, 0, self.q_embed.num_embeddings - 1)
        qa_data = torch.clamp(qa_data, 0, self.qa_embed.num_embeddings - 1)
        
        # Initialize memory
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
        # Initialize KC states if in per-KC mode
        if self.per_kc_mode:
            kc_states = {}
            for kc_id in range(self.n_kcs):
                kc_states[kc_id] = self.init_kc_states[kc_id].unsqueeze(0).repeat(batch_size, 1)
        
        # Embed sequences
        q_embedded = self.q_embed(q_data)  # (batch_size, seq_len, q_embed_dim)
        qa_embedded = self.qa_embed(qa_data)  # (batch_size, seq_len, qa_embed_dim)
        
        # Process sequence
        predictions = []
        student_abilities = []
        item_difficulties = []
        z_values = []
        kc_info = {'all_kc_thetas': []} if self.per_kc_mode else {}
        
        for t in range(seq_len):
            # Get current embeddings
            q_t = q_embedded[:, t, :]  # (batch_size, q_embed_dim)
            qa_t = qa_embedded[:, t, :]  # (batch_size, qa_embed_dim)
            q_ids_t = q_data[:, t]  # (batch_size,)
            
            # Memory operations
            correlation_weight = self.memory.attention(q_t)
            read_content = self.memory.read(correlation_weight)
            
            # Build summary vector
            summary_input = torch.cat([read_content, q_t], dim=1)
            summary_vector = self.summary_network(summary_input)
            
            if self.per_kc_mode:
                # Per-KC continuous tracking
                kc_encoding = self.get_kc_encoding(q_ids_t, device)
                
                # Update ALL KC states (continuous evolution)
                new_kc_states = {}
                for kc_id in range(self.n_kcs):
                    kc_network = self.kc_state_networks[f'kc_{kc_id}']
                    
                    # KC-specific input (stronger for active KCs)
                    kc_mask = kc_encoding[:, kc_id:kc_id+1]  # (batch_size, 1)
                    kc_input = summary_vector * (1.0 + 2.0 * kc_mask)
                    
                    # Update KC state
                    new_kc_states[kc_id] = kc_network(kc_input, kc_states[kc_id])
                
                # Apply cross-KC influence
                all_kc_states_flat = torch.cat([new_kc_states[i] for i in range(self.n_kcs)], dim=1)
                cross_kc_influence = self.cross_kc_network(all_kc_states_flat)
                cross_kc_influence = cross_kc_influence.view(batch_size, self.n_kcs, self.kc_state_dim)
                
                # Compute theta for ALL KCs
                timestep_kc_thetas = torch.zeros(batch_size, self.n_kcs, device=device)
                
                for kc_id in range(self.n_kcs):
                    # Apply cross-KC influence
                    influenced_state = new_kc_states[kc_id] + 0.1 * cross_kc_influence[:, kc_id, :]
                    
                    # Compute theta for this KC
                    kc_theta = self.kc_theta_networks[f'kc_{kc_id}'](influenced_state)
                    timestep_kc_thetas[:, kc_id] = kc_theta.squeeze(-1)
                    
                    # Update state for next timestep
                    kc_states[kc_id] = influenced_state
                
                # For IRT prediction, use primary KC theta
                primary_kc_thetas = torch.zeros(batch_size, 1, device=device)
                for b in range(batch_size):
                    q_id = q_ids_t[b].item()
                    primary_kc = self.q_to_kc.get(q_id, [0])[0]
                    primary_kc = min(primary_kc, self.n_kcs - 1)
                    primary_kc_thetas[b, 0] = timestep_kc_thetas[b, primary_kc]
                
                # Item difficulty with KC context
                enhanced_q_input = torch.cat([q_t, kc_encoding], dim=1)
                item_difficulty = self.item_difficulty_net(enhanced_q_input)
                
                # Store all KC thetas
                kc_info['all_kc_thetas'].append(timestep_kc_thetas)
                student_ability = primary_kc_thetas
                
            else:
                # Global tracking
                student_ability = self.student_ability_net(summary_vector)
                item_difficulty = self.item_difficulty_net(q_t)
            
            # IRT prediction
            prediction, z_value = self.irt_prediction(student_ability, item_difficulty)
            
            # Store results
            predictions.append(prediction)
            student_abilities.append(student_ability)
            item_difficulties.append(item_difficulty)
            z_values.append(z_value)
            
            # Update memory
            self.memory.write(correlation_weight, qa_t)
        
        # Stack results
        predictions = torch.stack(predictions, dim=1).squeeze(-1)  # (batch_size, seq_len)
        student_abilities = torch.stack(student_abilities, dim=1).squeeze(-1)
        item_difficulties = torch.stack(item_difficulties, dim=1).squeeze(-1)
        z_values = torch.stack(z_values, dim=1).squeeze(-1)
        
        if self.per_kc_mode:
            kc_info['all_kc_thetas'] = torch.stack(kc_info['all_kc_thetas'], dim=1)  # (batch_size, seq_len, n_kcs)
        
        return predictions, student_abilities, item_difficulties, z_values, kc_info
    
    def compute_loss(self, predictions, targets, target_mask=None):
        """Compute binary cross-entropy loss."""
        if target_mask is None:
            target_mask = targets >= 0
        
        masked_predictions = predictions[target_mask]
        masked_targets = targets[target_mask].float()
        
        if len(masked_predictions) == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        loss = F.binary_cross_entropy(masked_predictions, masked_targets)
        return loss