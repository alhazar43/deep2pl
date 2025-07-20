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
        
        # Prediction network - matching reference implementations
        # Concatenate read_content + question_embedding for prediction
        prediction_input_dim = value_dim + key_dim
        
        self.prediction_network = nn.Sequential(
            nn.Linear(prediction_input_dim, final_fc_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_fc_dim, 1)
        )
        
        # Initialize value memory parameter
        self.init_value_memory = nn.Parameter(torch.randn(memory_size, value_dim))
        nn.init.kaiming_normal_(self.init_value_memory)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using appropriate initialization schemes."""
        # Initialize embedding layers
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)
        
        # Initialize prediction network
        for module in self.prediction_network:
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
            tuple: (predictions, student_abilities, item_difficulties, z_values, kc_info)
            - predictions: Shape (batch_size, seq_len)  
            - student_abilities: Shape (batch_size, seq_len) - dummy values
            - item_difficulties: Shape (batch_size, seq_len) - dummy values
            - z_values: Shape (batch_size, seq_len) - dummy values
            - kc_info: Dict with KC information - empty dict
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
        
        # Process sequence step by step - matching reference implementations
        predictions = []
        
        for t in range(seq_len):
            # Get current embeddings
            q_t = q_embedded[:, t, :]   # (batch_size, key_dim)
            qa_t = qa_embedded[:, t, :] # (batch_size, value_dim)
            
            # DKVMN operations - matching reference order
            # 1. Attention
            correlation_weight = self.memory.attention(q_t)
            
            # 2. Read
            read_content = self.memory.read(correlation_weight)
            
            # 3. Predict (before write, using current read content)
            # Concatenate read content with question embedding for prediction
            prediction_input = torch.cat([read_content, q_t], dim=1)
            prediction_logit = self.prediction_network(prediction_input)
            prediction = torch.sigmoid(prediction_logit)
            
            predictions.append(prediction)
            
            # 4. Write (update memory for next step)
            self.memory.write(correlation_weight, qa_t)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=1).squeeze(-1)  # (batch_size, seq_len)
        
        # Return dummy values for backward compatibility with training script
        student_abilities = torch.zeros_like(predictions)  # Dummy values
        item_difficulties = torch.zeros_like(predictions)  # Dummy values  
        z_values = torch.zeros_like(predictions)  # Dummy values
        kc_info = {}  # Empty dict
        
        return predictions, student_abilities, item_difficulties, z_values, kc_info
    
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