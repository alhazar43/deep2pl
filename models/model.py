import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory import DKVMN


class StudentAbilityNetwork(nn.Module):
    """
    Neural network to estimate student ability from summary vector.
    """
    
    def __init__(self, input_dim, hidden_dim=None, output_dim=1):
        super(StudentAbilityNetwork, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
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
            student_ability: Shape (batch_size, output_dim)
        """
        return self.network(summary_vector)


class ItemDifficultyNetwork(nn.Module):
    """
    Neural network to estimate item difficulty from question embedding.
    """
    
    def __init__(self, input_dim, hidden_dim=None, output_dim=1):
        super(ItemDifficultyNetwork, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, question_embedding):
        """
        Args:
            question_embedding: Shape (batch_size, input_dim)
            
        Returns:
            item_difficulty: Shape (batch_size, output_dim)
        """
        return self.network(question_embedding)


class IRTPredictor(nn.Module):
    """
    Item Response Theory predictor for 1PL or 2PL models.
    """
    
    def __init__(self, model_type="1PL", use_discrimination=False, ability_scale=1.0):
        super(IRTPredictor, self).__init__()
        self.model_type = model_type
        self.use_discrimination = use_discrimination
        self.ability_scale = ability_scale
        
        if use_discrimination:
            # For 2PL model, learn discrimination parameter
            self.discrimination = nn.Parameter(torch.ones(1))
        else:
            self.discrimination = None
    
    def forward(self, student_ability, item_difficulty, discrimination=None):
        """
        Compute IRT prediction.
        
        Args:
            student_ability: Shape (batch_size, 1)
            item_difficulty: Shape (batch_size, 1)
            discrimination: Optional discrimination parameter
            
        Returns:
            prediction: Shape (batch_size, 1)
        """
        if self.use_discrimination:
            if discrimination is None:
                discrimination = self.discrimination
            z_value = discrimination * (self.ability_scale * student_ability - item_difficulty)
        else:
            z_value = self.ability_scale * student_ability - item_difficulty
        
        prediction = torch.sigmoid(z_value)
        return prediction, z_value


class DeepIRTModel(nn.Module):
    """
    Deep-IRT model combining DKVMN with IRT for explainable knowledge tracing.
    """
    
    def __init__(self, n_questions, memory_size, key_memory_state_dim, 
                 value_memory_state_dim, summary_vector_dim, 
                 q_embed_dim=None, qa_embed_dim=None, 
                 ability_scale=3.0, use_discrimination=False,
                 dropout_rate=0.0):
        super(DeepIRTModel, self).__init__()
        
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
        
        # Embedding layers
        self.q_embed = nn.Embedding(n_questions + 1, q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * n_questions + 1, qa_embed_dim, padding_idx=0)
        
        # DKVMN memory
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
        
        # Student ability and item difficulty networks
        self.student_ability_net = StudentAbilityNetwork(summary_vector_dim)
        self.item_difficulty_net = ItemDifficultyNetwork(q_embed_dim)
        
        # IRT predictor
        self.irt_predictor = IRTPredictor(
            model_type="1PL", 
            use_discrimination=use_discrimination,
            ability_scale=ability_scale
        )
        
        # Initial value memory
        self.init_value_memory = nn.Parameter(
            torch.randn(memory_size, value_memory_state_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize embeddings
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)
        
        # Initialize value memory
        nn.init.kaiming_normal_(self.init_value_memory)
        
        # Initialize summary network
        for module in self.summary_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, q_data, qa_data, target_mask=None):
        """
        Forward pass through the Deep-IRT model.
        
        Args:
            q_data: Question data, shape (batch_size, seq_len)
            qa_data: Question-answer data, shape (batch_size, seq_len)
            target_mask: Optional mask for targets, shape (batch_size, seq_len)
            
        Returns:
            predictions: Shape (batch_size, seq_len)
            student_abilities: Shape (batch_size, seq_len)
            item_difficulties: Shape (batch_size, seq_len)
            z_values: Shape (batch_size, seq_len)
        """
        batch_size, seq_len = q_data.shape
        
        # Initialize value memory for this batch
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
        # Embed questions and question-answers
        q_embedded = self.q_embed(q_data)  # (batch_size, seq_len, q_embed_dim)
        qa_embedded = self.qa_embed(qa_data)  # (batch_size, seq_len, qa_embed_dim)
        
        # Process sequence
        predictions = []
        student_abilities = []
        item_difficulties = []
        z_values = []
        
        for t in range(seq_len):
            # Get embeddings for current timestep
            q_t = q_embedded[:, t, :]  # (batch_size, q_embed_dim)
            qa_t = qa_embedded[:, t, :]  # (batch_size, qa_embed_dim)
            
            # DKVMN operations
            correlation_weight = self.memory.attention(q_t)
            read_content = self.memory.read(correlation_weight)
            
            # Build summary vector
            summary_input = torch.cat([read_content, q_t], dim=1)
            summary_vector = self.summary_network(summary_input)
            
            # Compute student ability and item difficulty
            student_ability = self.student_ability_net(summary_vector)
            item_difficulty = self.item_difficulty_net(q_t)
            
            # IRT prediction
            prediction, z_value = self.irt_predictor(student_ability, item_difficulty)
            
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
        
        return predictions, student_abilities, item_difficulties, z_values
    
    def compute_loss(self, predictions, targets, target_mask=None):
        """
        Compute binary cross-entropy loss.
        
        Args:
            predictions: Model predictions, shape (batch_size, seq_len)
            targets: Ground truth labels, shape (batch_size, seq_len)
            target_mask: Optional mask for valid targets, shape (batch_size, seq_len)
            
        Returns:
            loss: Scalar loss value
        """
        if target_mask is None:
            # Create mask for non-negative targets (assuming -1 is mask value)
            target_mask = targets >= 0
        
        # Apply mask
        masked_predictions = predictions[target_mask]
        masked_targets = targets[target_mask].float()
        
        # Compute BCE loss
        loss = F.binary_cross_entropy(masked_predictions, masked_targets)
        
        return loss