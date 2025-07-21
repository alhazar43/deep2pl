"""
Fixed Deep-IRT model based on reference implementations.
Simplified to focus on core DKVMN functionality without over-complex per-KC tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .memory import DKVMN


class DeepIRTModel(nn.Module):
    """
    Fixed Deep-IRT model based on reference implementations.
    Simplified to match proven DKVMN architectures.
    """
    
    def __init__(self, n_questions, memory_size=50, key_dim=50, 
                 value_dim=200, final_fc_dim=50, dropout_rate=0.0,
                 # Legacy parameters for backward compatibility
                 summary_dim=None, q_embed_dim=None, qa_embed_dim=None,
                 ability_scale=3.0, use_discrimination=True, 
                 q_matrix_path=None, skill_mapping_path=None):
        """
        Initialize Deep-IRT model with per-KC support.
        
        Args:
            n_questions: Number of unique questions
            memory_size: Size of memory matrix
            key_dim: Dimension of key embeddings
            value_dim: Dimension of value embeddings
            final_fc_dim: Hidden dimension for final prediction network
            dropout_rate: Dropout rate
            ability_scale: IRT ability scaling factor
            use_discrimination: Whether to use discrimination parameters
            q_matrix_path: Path to Q-matrix file (enables per-KC mode)
            skill_mapping_path: Path to skill mapping file
        """
        super(DeepIRTModel, self).__init__()
        
        # Handle legacy parameters for backward compatibility
        if summary_dim is not None:
            final_fc_dim = summary_dim
        
        # Store core parameters
        self.n_questions = n_questions
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.final_fc_dim = final_fc_dim
        self.ability_scale = ability_scale
        self.use_discrimination = use_discrimination
        self.dropout_rate = dropout_rate
        
        # Initialize per-KC mode and Q-matrix
        self.per_kc_mode, self.q_to_kc, self.kc_names, self.n_kcs = self._load_kc_info(
            q_matrix_path, skill_mapping_path
        )
        if q_embed_dim is not None:
            key_dim = q_embed_dim  
        if qa_embed_dim is not None:
            value_dim = qa_embed_dim
            
        self.n_questions = n_questions
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.final_fc_dim = final_fc_dim
        self.dropout_rate = dropout_rate
        
        # Legacy attributes for backward compatibility (initialized by _load_kc_info above)
        
        # Embedding layers - matching reference implementations
        self.q_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * n_questions + 1, value_dim, padding_idx=0)
        
        # Initialize DKVMN
        init_key_memory = torch.randn(memory_size, key_dim)
        nn.init.kaiming_normal_(init_key_memory)
        
        self.memory = DKVMN(
            memory_size=memory_size,
            key_dim=key_dim,
            value_dim=value_dim,
            init_key_memory=init_key_memory
        )
        
        # 2PL IRT prediction layers - always include discrimination
        # Concatenate read_content + question_embedding for summary vector
        summary_input_dim = value_dim + key_dim
        
        # Summary vector network (equivalent to deep-yeung summary vector)
        self.summary_network = nn.Sequential(
            nn.Linear(summary_input_dim, final_fc_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        
        # Student ability network (theta) - from summary vector
        self.student_ability_network = nn.Linear(final_fc_dim, 1)
        
        # Question difficulty network (beta) - from question embedding
        self.question_difficulty_network = nn.Sequential(
            nn.Linear(key_dim, 1),
            nn.Tanh()  # Matching deep-yeung activation
        )
        
        # 2PL Discrimination parameter (alpha) - from summary + question embedding
        # a = softplus(W_a [f_t; k_t] + b_a) where f_t=summary, k_t=question_embedding
        discrimination_input_dim = final_fc_dim + key_dim  # [f_t; k_t]
        self.discrimination_network = nn.Sequential(
            nn.Linear(discrimination_input_dim, 1),
            nn.Softplus()  # Ensures positive discrimination
        )
        
        # Per-KC networks (when per_kc_mode is enabled)
        if self.per_kc_mode and self.n_kcs > 0:
            # Per-KC ability networks
            self.kc_ability_networks = nn.ModuleList([
                nn.Linear(final_fc_dim, 1) for _ in range(self.n_kcs)
            ])
            # Per-KC difficulty networks  
            self.kc_difficulty_networks = nn.ModuleList([
                nn.Sequential(nn.Linear(key_dim, 1), nn.Tanh()) for _ in range(self.n_kcs)
            ])
            # Per-KC discrimination networks (2PL)
            self.kc_discrimination_networks = nn.ModuleList([
                nn.Sequential(nn.Linear(discrimination_input_dim, 1), nn.Softplus()) for _ in range(self.n_kcs)
            ])
        
        # Initialize value memory parameter
        self.init_value_memory = nn.Parameter(torch.randn(memory_size, value_dim))
        nn.init.kaiming_normal_(self.init_value_memory)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using appropriate initialization schemes."""
        # Initialize embedding layers
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)
        
        # Initialize 2PL IRT networks
        for module in self.summary_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        nn.init.kaiming_normal_(self.student_ability_network.weight)
        nn.init.constant_(self.student_ability_network.bias, 0)
        
        for module in self.question_difficulty_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        for module in self.discrimination_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        # Initialize per-KC networks if enabled
        if self.per_kc_mode and self.n_kcs > 0:
            for kc_net in self.kc_ability_networks:
                nn.init.kaiming_normal_(kc_net.weight)
                nn.init.constant_(kc_net.bias, 0)
            for kc_net in self.kc_difficulty_networks:
                for module in kc_net:
                    if isinstance(module, nn.Linear):
                        nn.init.kaiming_normal_(module.weight)
                        nn.init.constant_(module.bias, 0)
            for kc_net in self.kc_discrimination_networks:
                for module in kc_net:
                    if isinstance(module, nn.Linear):
                        nn.init.kaiming_normal_(module.weight)
                        nn.init.constant_(module.bias, 0)
    
    def forward(self, q_data, qa_data, target_mask=None):
        """
        Forward pass through the model.
        Simplified to match reference implementations.
        
        Args:
            q_data: Question IDs, shape (batch_size, seq_len)
            qa_data: Question-Answer IDs, shape (batch_size, seq_len)
            target_mask: Optional mask for valid positions
            
        Returns:
            tuple: (predictions, student_abilities, item_difficulties, discrimination_params, z_values, kc_info)
            - predictions: Shape (batch_size, seq_len)  
            - student_abilities: Shape (batch_size, seq_len) - theta parameters
            - item_difficulties: Shape (batch_size, seq_len) - beta parameters
            - discrimination_params: Shape (batch_size, seq_len) - alpha parameters (2PL)
            - z_values: Shape (batch_size, seq_len) - z-values for IRT
            - kc_info: Dict with KC information and per-KC parameters
        """
        batch_size, seq_len = q_data.shape
        device = q_data.device
        
        # Safety check for embedding bounds
        q_data = torch.clamp(q_data, 0, self.q_embed.num_embeddings - 1)
        qa_data = torch.clamp(qa_data, 0, self.qa_embed.num_embeddings - 1)
        
        # Initialize memory for this batch
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
        # Embed sequences
        q_embedded = self.q_embed(q_data)    # (batch_size, seq_len, key_dim)
        qa_embedded = self.qa_embed(qa_data)  # (batch_size, seq_len, value_dim)
        
        # Process sequence step by step - 2PL IRT with discrimination
        predictions = []
        student_abilities = []
        item_difficulties = []
        discrimination_params = []
        z_values = []
        
        # Per-KC tracking
        kc_info = {}
        if self.per_kc_mode and self.n_kcs > 0:
            all_kc_abilities = []  # (batch_size, seq_len, n_kcs)
            all_kc_difficulties = []  # (batch_size, seq_len, n_kcs) 
            all_kc_discriminations = []  # (batch_size, seq_len, n_kcs)
            all_kc_z_values = []  # (batch_size, seq_len, n_kcs)
        
        for t in range(seq_len):
            # Get current embeddings
            q_t = q_embedded[:, t, :]   # (batch_size, key_dim)
            qa_t = qa_embedded[:, t, :] # (batch_size, value_dim)
            
            # DKVMN operations - matching reference order
            # 1. Attention
            correlation_weight = self.memory.attention(q_t)
            
            # 2. Read
            read_content = self.memory.read(correlation_weight)
            
            # 3. 2PL IRT Parameter Computation
            # Summary vector (mastery_level_prior_difficulty)
            summary_input = torch.cat([read_content, q_t], dim=1)
            summary_vector = self.summary_network(summary_input)
            
            # Student ability (theta) from summary vector
            student_ability = self.student_ability_network(summary_vector)  # (batch_size, 1)
            
            # Question difficulty (beta) from question embedding  
            question_difficulty = self.question_difficulty_network(q_t)  # (batch_size, 1)
            
            # Discrimination parameter (alpha) - 2PL: a = softplus(W_a [f_t; k_t] + b_a)
            discrimination_input = torch.cat([summary_vector, q_t], dim=1)  # [f_t; k_t]
            discrimination = self.discrimination_network(discrimination_input)  # (batch_size, 1)
            
            # 2PL prediction: z = a * (ability_scale * theta - beta)
            z_value = discrimination * (self.ability_scale * student_ability - question_difficulty)
            prediction = torch.sigmoid(z_value)
            
            # Store step results
            predictions.append(prediction)
            student_abilities.append(student_ability)
            item_difficulties.append(question_difficulty)
            discrimination_params.append(discrimination)
            z_values.append(z_value)
            
            # Per-KC computation if enabled
            if self.per_kc_mode and self.n_kcs > 0:
                step_kc_abilities = []
                step_kc_difficulties = []
                step_kc_discriminations = []
                step_kc_z_values = []
                
                for kc_idx in range(self.n_kcs):
                    kc_ability = self.kc_ability_networks[kc_idx](summary_vector)
                    kc_difficulty = self.kc_difficulty_networks[kc_idx](q_t)
                    kc_discrimination = self.kc_discrimination_networks[kc_idx](discrimination_input)
                    kc_z = kc_discrimination * (self.ability_scale * kc_ability - kc_difficulty)
                    
                    step_kc_abilities.append(kc_ability)
                    step_kc_difficulties.append(kc_difficulty)
                    step_kc_discriminations.append(kc_discrimination)
                    step_kc_z_values.append(kc_z)
                
                all_kc_abilities.append(torch.cat(step_kc_abilities, dim=1))  # (batch_size, n_kcs)
                all_kc_difficulties.append(torch.cat(step_kc_difficulties, dim=1))
                all_kc_discriminations.append(torch.cat(step_kc_discriminations, dim=1))
                all_kc_z_values.append(torch.cat(step_kc_z_values, dim=1))
            
            # 4. Write (update memory for next step)
            self.memory.write(correlation_weight, qa_t)
        
        # Stack all step results
        predictions = torch.stack(predictions, dim=1).squeeze(-1)  # (batch_size, seq_len)
        student_abilities = torch.stack(student_abilities, dim=1).squeeze(-1)  # (batch_size, seq_len)
        item_difficulties = torch.stack(item_difficulties, dim=1).squeeze(-1)  # (batch_size, seq_len)
        discrimination_params = torch.stack(discrimination_params, dim=1).squeeze(-1)  # (batch_size, seq_len)
        z_values = torch.stack(z_values, dim=1).squeeze(-1)  # (batch_size, seq_len)
        
        # Per-KC results
        if self.per_kc_mode and self.n_kcs > 0:
            kc_info = {
                'all_kc_thetas': torch.stack(all_kc_abilities, dim=1),  # (batch_size, seq_len, n_kcs)
                'all_kc_betas': torch.stack(all_kc_difficulties, dim=1),  # (batch_size, seq_len, n_kcs)
                'all_kc_alphas': torch.stack(all_kc_discriminations, dim=1),  # (batch_size, seq_len, n_kcs)
                'all_kc_z_values': torch.stack(all_kc_z_values, dim=1),  # (batch_size, seq_len, n_kcs)
                'q_to_kc': self.q_to_kc,
                'kc_names': self.kc_names,
                'n_kcs': self.n_kcs
            }
        
        # Store discrimination in kc_info for visualization access
        kc_info['discrimination_params'] = discrimination_params
        
        return predictions, student_abilities, item_difficulties, discrimination_params, z_values, kc_info
    
    def compute_loss(self, predictions, targets, target_mask=None):
        """
        Compute binary cross-entropy loss.
        Matching reference implementations.
        """
        if target_mask is None:
            # Create mask for valid targets (non-negative)
            target_mask = targets >= 0
        
        # Apply mask
        masked_predictions = predictions[target_mask]
        masked_targets = targets[target_mask].float()
        
        if len(masked_predictions) == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(masked_predictions, masked_targets)
        return loss

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


class DKVMNModel(nn.Module):
    """
    Ultra-simple version matching dkvmn-torch exactly for debugging.
    """
    
    def __init__(self, n_questions, memory_size=50, key_dim=50, value_dim=200, final_fc_dim=50):
        super(DKVMNModel, self).__init__()
        
        self.n_questions = n_questions
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.final_fc_dim = final_fc_dim
        
        # Embeddings
        self.q_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * n_questions + 1, value_dim, padding_idx=0)
        
        # Memory
        self.init_key_memory = nn.Parameter(torch.randn(memory_size, key_dim))
        self.init_value_memory = nn.Parameter(torch.randn(memory_size, value_dim))
        nn.init.kaiming_normal_(self.init_key_memory)
        nn.init.kaiming_normal_(self.init_value_memory)
        
        self.memory = DKVMN(memory_size=memory_size, key_dim=key_dim, 
                           value_dim=value_dim, init_key_memory=self.init_key_memory)
        
        # Prediction
        self.read_embed_linear = nn.Linear(value_dim + key_dim, final_fc_dim, bias=True)
        self.predict_linear = nn.Linear(final_fc_dim, 1, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)
    
    def forward(self, q_data, qa_data, target):
        batch_size, seq_len = q_data.shape
        
        # Embeddings
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)
        
        # Initialize memory
        memory_value = self.init_value_memory.unsqueeze(0).expand(
            batch_size, self.memory_size, self.value_dim
        ).contiguous()
        self.memory.init_value_memory(batch_size, memory_value)
        
        # Process sequence
        value_read_content_l = []
        input_embed_l = []
        
        for t in range(seq_len):
            q = q_embed_data[:, t, :]
            qa = qa_embed_data[:, t, :]
            
            # Attention
            correlation_weight = self.memory.attention(q)
            
            # Read
            read_content = self.memory.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(q)
            
            # Write
            self.memory.write(correlation_weight, qa)
        
        # Prediction
        all_read_value_content = torch.stack(value_read_content_l, dim=1)
        input_embed_content = torch.stack(input_embed_l, dim=1)
        
        predict_input = torch.cat([all_read_value_content, input_embed_content], dim=2)
        read_content_embed = torch.tanh(
            self.read_embed_linear(predict_input.view(batch_size * seq_len, -1))
        )
        
        pred = self.predict_linear(read_content_embed)
        
        # Loss computation
        target_1d = target.view(-1, 1)
        mask = target_1d.ge(0)  # Changed from ge(1) to ge(0) for 0/1 targets
        pred_1d = pred.view(-1, 1)
        
        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask).float()
        
        if len(filtered_pred) == 0:
            loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
            return loss, torch.tensor([]), torch.tensor([])
        
        loss = F.binary_cross_entropy_with_logits(filtered_pred, filtered_target)
        
        return loss, torch.sigmoid(filtered_pred), filtered_target