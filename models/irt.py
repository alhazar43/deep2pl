"""
Deep Item Response Theory Model with Continuous Per-Knowledge Component Tracking

This implementation provides a unified architecture supporting multiple data formats
and automatic detection of Q-matrix availability for enhanced knowledge component
tracking capabilities.

Key Features:
- Compatibility with both data-orig and data-yeung data formats
- Optional per-KC continuous tracking when Q-matrix is available
- Fallback to global tracking when no Q-matrix is provided
- Continuous evolution where all KCs update at each timestep

Author: Implementation based on Yeung & Yeung (2019) Deep Knowledge Tracing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv
from .memory import DKVMN


class ItemDiscriminationStaticNetwork(nn.Module):
    """
    Neural network to estimate item discrimination from question and question-answer embeddings.
    This follows the traditional IRT approach where discrimination is an item property.
    """
    
    def __init__(self, q_embed_dim, qa_embed_dim, hidden_dim=None, output_dim=1):
        super(ItemDiscriminationStaticNetwork, self).__init__()
        input_dim = q_embed_dim + qa_embed_dim
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # Ensure positive discrimination
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, question_embedding, qa_embedding):
        """
        Args:
            question_embedding: Shape (batch_size, q_embed_dim)
            qa_embedding: Shape (batch_size, qa_embed_dim)
            
        Returns:
            item_discrimination: Shape (batch_size, output_dim)
        """
        combined_input = torch.cat([question_embedding, qa_embedding], dim=1)
        return self.network(combined_input)


class ItemDiscriminationDynamicNetwork(nn.Module):
    """
    Neural network to estimate item discrimination from summary vector.
    This allows discrimination to vary based on student's knowledge state.
    """
    
    def __init__(self, input_dim, hidden_dim=None, output_dim=1):
        super(ItemDiscriminationDynamicNetwork, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # Ensure positive discrimination
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, summary_vector):
        """
        Args:
            summary_vector: Shape (batch_size, input_dim)
            
        Returns:
            item_discrimination: Shape (batch_size, output_dim)
        """
        return self.network(summary_vector)


class DeepIRTModel(nn.Module):
    """
    Deep Item Response Theory model with optional continuous per-knowledge component tracking
    and item discrimination parameter estimation.
    
    This model extends the standard Deep-IRT architecture to support continuous evolution
    of knowledge component states when Q-matrix information is available. The model
    automatically switches between global and per-KC modes based on data availability.
    
    Additionally supports 2PL IRT with discrimination parameter estimation.
    """
    
    def __init__(self, n_questions, memory_size=50, key_memory_state_dim=50, 
                 value_memory_state_dim=200, summary_vector_dim=50, 
                 q_embed_dim=None, qa_embed_dim=None, ability_scale=3.0, 
                 use_discrimination=False, discrimination_type="static", dropout_rate=0.0, 
                 q_matrix_path=None, skill_mapping_path=None):
        """
        Initialize the Deep Item Response Theory model.
        
        Parameters:
            n_questions (int): Total number of questions in the dataset
            memory_size (int): Size of the DKVMN memory bank
            key_memory_state_dim (int): Dimension of key memory states
            value_memory_state_dim (int): Dimension of value memory states
            summary_vector_dim (int): Dimension of summary vectors
            q_embed_dim (int, optional): Question embedding dimension
            qa_embed_dim (int, optional): Question-answer embedding dimension
            ability_scale (float): Scaling factor for student ability in IRT
            use_discrimination (bool): Whether to use discrimination parameters
            discrimination_type (str): Type of discrimination ("static", "dynamic", or "both")
            dropout_rate (float): Dropout probability for regularization
            q_matrix_path (str, optional): Path to Q-matrix file for per-KC tracking
            skill_mapping_path (str, optional): Path to skill mapping file
        """
        super(DeepIRTModel, self).__init__()
        
        # Configure embedding dimensions with defaults if not specified
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
        self.discrimination_type = discrimination_type
        self.dropout_rate = dropout_rate
        
        # Initialize knowledge component configuration and Q-matrix integration
        self.per_kc_mode, self.q_to_kc, self.kc_names, self.n_kcs = self._load_kc_info(
            q_matrix_path, skill_mapping_path
        )
        
        # Initialize embedding layers for questions and question-answer pairs
        self.q_embed = nn.Embedding(n_questions + 1, q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * n_questions + 1, qa_embed_dim, padding_idx=0)
        
        # Initialize Dynamic Key-Value Memory Network (DKVMN)
        self.memory = DKVMN(
            memory_size=memory_size,
            key_memory_state_dim=key_memory_state_dim,
            value_memory_state_dim=value_memory_state_dim
        )
        
        # Initialize summary vector processing network
        summary_input_dim = value_memory_state_dim + q_embed_dim
        self.summary_network = nn.Sequential(
            nn.Linear(summary_input_dim, summary_vector_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        
        # Initialize discrimination networks
        self.item_discrimination_static_net = None
        self.item_discrimination_dynamic_net = None
        
        if use_discrimination:
            if discrimination_type == "static":
                self.item_discrimination_static_net = ItemDiscriminationStaticNetwork(
                    q_embed_dim, qa_embed_dim
                )
            elif discrimination_type == "dynamic":
                self.item_discrimination_dynamic_net = ItemDiscriminationDynamicNetwork(
                    summary_vector_dim
                )
            elif discrimination_type == "both":
                self.item_discrimination_static_net = ItemDiscriminationStaticNetwork(
                    q_embed_dim, qa_embed_dim
                )
                self.item_discrimination_dynamic_net = ItemDiscriminationDynamicNetwork(
                    summary_vector_dim
                )
        
        if self.per_kc_mode:
            # Initialize per-knowledge component continuous tracking components
            self.kc_state_dim = min(32, summary_vector_dim // 2)
            
            # Initialize per-KC state networks for continuous evolution
            self.kc_state_networks = nn.ModuleDict({
                f'kc_{i}': nn.GRUCell(summary_vector_dim, self.kc_state_dim)
                for i in range(self.n_kcs)
            })
            
            # Initialize per-KC theta estimation networks
            self.kc_theta_networks = nn.ModuleDict({
                f'kc_{i}': nn.Sequential(
                    nn.Linear(self.kc_state_dim, self.kc_state_dim // 2),
                    nn.Tanh(),
                    nn.Linear(self.kc_state_dim // 2, 1)
                ) for i in range(self.n_kcs)
            })
            
            # Initialize cross-knowledge component influence network
            self.cross_kc_network = nn.Sequential(
                nn.Linear(self.n_kcs * self.kc_state_dim, summary_vector_dim),
                nn.Tanh(),
                nn.Linear(summary_vector_dim, self.n_kcs * self.kc_state_dim),
                nn.Dropout(dropout_rate)
            )
            
            # Initialize item difficulty estimation with KC context
            self.item_difficulty_net = nn.Sequential(
                nn.Linear(q_embed_dim + self.n_kcs, summary_vector_dim),
                nn.Tanh(),
                nn.Linear(summary_vector_dim, 1)
            )
            
            # Initialize learnable initial KC states
            self.init_kc_states = nn.Parameter(
                torch.randn(self.n_kcs, self.kc_state_dim) * 0.1
            )
        else:
            # Initialize global tracking components for standard mode
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

        # Note: Discrimination is now handled by the discrimination networks above
        
        # Initialize learnable initial value memory for DKVMN
        self.init_value_memory = nn.Parameter(
            torch.randn(memory_size, value_memory_state_dim) * 0.1
        )
        
        self._init_weights()
    
    def _load_kc_info(self, q_matrix_path, skill_mapping_path):
        """
        Load Q-matrix and skill mapping information for per-KC mode configuration.
        
        Parameters:
            q_matrix_path (str): Path to the Q-matrix file (CSV or qid_sid format)
            skill_mapping_path (str): Path to the skill mapping file
            
        Returns:
            tuple: (per_kc_mode, q_to_kc, kc_names, n_kcs)
        """
        if q_matrix_path is None or not os.path.exists(q_matrix_path):
            # Q-matrix not available, fall back to global tracking mode
            return False, {}, {}, 0
        
        q_to_kc = {}
        kc_names = {}
        
        try:
            # Detect file format and parse accordingly
            if q_matrix_path.endswith('.csv') and 'Qmatrix' in q_matrix_path:
                # Multi-hot Q-matrix format (STATICS style)
                with open(q_matrix_path, 'r') as f:
                    for q_idx, line in enumerate(f, 1):
                        kc_vector = [int(x) for x in line.strip().split(',')]
                        kcs = [i for i, val in enumerate(kc_vector) if val == 1]
                        q_to_kc[q_idx] = kcs if kcs else [0]  # Default to KC 0 if none
                
                n_kcs = len(kc_vector)
                
            elif 'qid_sid' in q_matrix_path or 'conceptname_question_id' in q_matrix_path:
                # 1-to-1 mapping format (assist2015, assist2009_updated, fsaif1tof3)
                max_qid = 0
                max_kc = 0
                
                with open(q_matrix_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                            
                        if 'conceptname_question_id' in q_matrix_path:
                            # CSV format: concept_name,question_id
                            parts = line.split(',')
                            if len(parts) >= 2:
                                kc_name = parts[0].strip()
                                q_id = int(parts[1].strip())
                                
                                # Map concept name to KC index
                                if kc_name not in kc_names:
                                    kc_id = len(kc_names)
                                    kc_names[kc_id] = kc_name
                                else:
                                    kc_id = next(k for k, v in kc_names.items() if v == kc_name)
                                
                                q_to_kc[q_id] = [kc_id]
                                max_qid = max(max_qid, q_id)
                                max_kc = max(max_kc, kc_id)
                        else:
                            # Tab-separated format: question_id\tskill_id
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                q_id = int(parts[0])
                                kc_id = int(parts[1]) - 1  # Convert to 0-indexed
                                q_to_kc[q_id] = [kc_id]
                                max_qid = max(max_qid, q_id)
                                max_kc = max(max_kc, kc_id)
                
                n_kcs = max_kc + 1
                
                # Load skill mapping if available
                if skill_mapping_path and os.path.exists(skill_mapping_path):
                    with open(skill_mapping_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                kc_id = int(parts[0]) - 1  # Convert to 0-indexed
                                kc_name = parts[1]
                                if 0 <= kc_id < n_kcs:
                                    kc_names[kc_id] = kc_name
                
                # Assign default names for KCs without explicit names
                for i in range(n_kcs):
                    if i not in kc_names:
                        kc_names[i] = f"KC_{i+1}"
            
            else:
                print(f"Warning: Unknown Q-matrix format: {q_matrix_path}")
                return False, {}, {}, 0
            
            print(f"Successfully loaded Q-matrix: {len(q_to_kc)} questions, {n_kcs} knowledge components")
            return True, q_to_kc, kc_names, n_kcs
            
        except Exception as e:
            print(f"Warning: Failed to load Q-matrix from {q_matrix_path}: {e}")
            return False, {}, {}, 0
    
    def _init_weights(self):
        """Initialize model weights using appropriate initialization schemes."""
        # Initialize embedding layers with Kaiming normal initialization
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)
        
        # Initialize memory parameters
        nn.init.kaiming_normal_(self.init_value_memory)
        
        # Initialize summary network layers
        for module in self.summary_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        # Initialize per-KC components if in per-KC mode
        if self.per_kc_mode:
            nn.init.kaiming_normal_(self.init_kc_states)
    
    def get_kc_encoding(self, question_ids, device):
        """
        Generate one-hot encoding for knowledge components associated with given questions.
        
        Parameters:
            question_ids (tensor): Tensor of question IDs
            device (torch.device): Device for tensor operations
            
        Returns:
            tensor: One-hot encoded KC matrix or None if not in per-KC mode
        """
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
    
    def irt_prediction(self, student_ability, item_difficulty, discrimination=None):
        """
        Compute Item Response Theory prediction using the logistic model.
        
        Parameters:
            student_ability (tensor): Student ability parameters (theta)
            item_difficulty (tensor): Item difficulty parameters (beta)
            discrimination (tensor, optional): Item discrimination parameters (alpha)
            
        Returns:
            tuple: (prediction, z_value) where prediction is probability and z_value is logit
        """
        if self.use_discrimination and discrimination is not None:
            z_value = discrimination * (self.ability_scale * student_ability - item_difficulty)
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
            item_discriminations: Shape (batch_size, seq_len) or None
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
        item_discriminations = []
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
            
            # Compute item discrimination if enabled
            item_discrimination = None
            if self.use_discrimination:
                if self.discrimination_type == "static":
                    item_discrimination = self.item_discrimination_static_net(q_t, qa_t)
                elif self.discrimination_type == "dynamic":
                    item_discrimination = self.item_discrimination_dynamic_net(summary_vector)
                elif self.discrimination_type == "both":
                    # Average both discriminations
                    static_disc = self.item_discrimination_static_net(q_t, qa_t)
                    dynamic_disc = self.item_discrimination_dynamic_net(summary_vector)
                    item_discrimination = (static_disc + dynamic_disc) / 2.0
            
            # IRT prediction
            prediction, z_value = self.irt_prediction(student_ability, item_difficulty, item_discrimination)
            
            # Store results
            predictions.append(prediction)
            student_abilities.append(student_ability)
            item_difficulties.append(item_difficulty)
            item_discriminations.append(item_discrimination)
            z_values.append(z_value)
            
            # Update memory
            self.memory.write(correlation_weight, qa_t)
        
        # Stack results
        predictions = torch.stack(predictions, dim=1).squeeze(-1)  # (batch_size, seq_len)
        student_abilities = torch.stack(student_abilities, dim=1).squeeze(-1)
        item_difficulties = torch.stack(item_difficulties, dim=1).squeeze(-1)
        z_values = torch.stack(z_values, dim=1).squeeze(-1)
        
        # Handle discriminations (might be None)
        if self.use_discrimination:
            item_discriminations = torch.stack(item_discriminations, dim=1).squeeze(-1)
        else:
            item_discriminations = None
        
        if self.per_kc_mode:
            kc_info['all_kc_thetas'] = torch.stack(kc_info['all_kc_thetas'], dim=1)  # (batch_size, seq_len, n_kcs)
        
        return predictions, student_abilities, item_difficulties, item_discriminations, z_values, kc_info
    
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