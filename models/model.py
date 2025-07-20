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
                 ability_scale=None, use_discrimination=None, 
                 q_matrix_path=None, skill_mapping_path=None):
        """
        Initialize simplified Deep-IRT model.
        
        Args:
            n_questions: Number of unique questions
            memory_size: Size of memory matrix
            key_dim: Dimension of key embeddings (usually same as q_embed_dim)
            value_dim: Dimension of value embeddings (usually same as qa_embed_dim)  
            final_fc_dim: Hidden dimension for final prediction network
            dropout_rate: Dropout rate
            
            # Legacy parameters (ignored but kept for compatibility):
            summary_dim: Ignored, use final_fc_dim instead
            q_embed_dim: Ignored, use key_dim instead
            qa_embed_dim: Ignored, use value_dim instead
            ability_scale: Ignored (simplified model)
            use_discrimination: Ignored (simplified model)
            q_matrix_path: Ignored (per-KC tracking removed)
            skill_mapping_path: Ignored (per-KC tracking removed)
        """
        super(DeepIRTModel, self).__init__()
        
        # Handle legacy parameters for backward compatibility
        if summary_dim is not None:
            final_fc_dim = summary_dim
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
        
        # Legacy attributes for backward compatibility
        self.per_kc_mode = False  # Simplified model doesn't use per-KC tracking
        self.q_to_kc = {}
        self.kc_names = {}
        self.n_kcs = 0
        
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