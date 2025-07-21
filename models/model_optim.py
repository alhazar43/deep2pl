"""
Optimized Deep-IRT model using SOTA techniques.
Phase 1: Multi-Head Prediction with Shared Encoder for efficient per-KC computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .memory import DKVMN


class OptimizedDeepIRTModel(nn.Module):
    """
    Optimized Deep-IRT model with efficient per-KC computation.
    Uses shared encoder with multi-head prediction instead of separate networks per KC.
    
    Key Optimizations:
    1. Shared encoder for all KC computations
    2. Multi-head outputs (single forward pass → all KC predictions)
    3. Layer normalization for stability
    4. Efficient tensor operations
    """
    
    def __init__(self, n_questions, memory_size=50, key_dim=50, 
                 value_dim=200, final_fc_dim=50, dropout_rate=0.0,
                 # Legacy parameters for backward compatibility
                 summary_dim=None, q_embed_dim=None, qa_embed_dim=None,
                 ability_scale=3.0, use_discrimination=True, 
                 q_matrix_path=None, skill_mapping_path=None):
        """
        Initialize Optimized Deep-IRT model with efficient per-KC support.
        
        Args:
            n_questions: Number of unique questions
            memory_size: Size of memory matrix
            key_dim: Dimension of key embeddings
            value_dim: Dimension of value embeddings
            final_fc_dim: Hidden dimension for prediction networks
            dropout_rate: Dropout rate
            ability_scale: IRT ability scaling factor
            use_discrimination: Whether to use discrimination parameters (always True for 2PL)
            q_matrix_path: Path to Q-matrix file (enables per-KC mode)
            skill_mapping_path: Path to skill mapping file
        """
        super(OptimizedDeepIRTModel, self).__init__()
        
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
        # Only enable per-KC mode for datasets with meaningful multi-KC Q-matrices
        self.per_kc_mode, self.q_to_kc, self.kc_names, self.n_kcs = self._load_kc_info(
            q_matrix_path, skill_mapping_path
        )
        
        # Disable per-KC mode if it's essentially single KC or 1-to-1 mapping (inefficient)
        if self.per_kc_mode and self.n_kcs <= 1:
            print(f"[OPTIM] Disabling per-KC mode: only {self.n_kcs} KC(s) detected")
            self.per_kc_mode = False
            self.n_kcs = 0
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
        
        # 2PL IRT prediction layers - optimized architecture
        # Concatenate read_content + question_embedding for summary vector
        summary_input_dim = value_dim + key_dim
        
        # Shared summary vector network (equivalent to deep-yeung summary vector)
        self.summary_network = nn.Sequential(
            nn.Linear(summary_input_dim, final_fc_dim),
            nn.LayerNorm(final_fc_dim),  # Layer normalization for stability
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        
        # Global IRT parameter networks (for non per-KC datasets)
        self.student_ability_network = nn.Linear(final_fc_dim, 1)
        self.question_difficulty_network = nn.Sequential(
            nn.Linear(key_dim, 1),
            nn.Tanh()  # Matching deep-yeung activation
        )
        
        # Global discrimination parameter (alpha) - 2PL
        discrimination_input_dim = final_fc_dim + key_dim  # [f_t; k_t]
        self.discrimination_network = nn.Sequential(
            nn.Linear(discrimination_input_dim, 1),
            nn.Softplus()  # Ensures positive discrimination
        )
        
        # OPTIMIZED Per-KC networks - using shared encoder with multi-head outputs
        if self.per_kc_mode and self.n_kcs > 0:
            print(f"[OPTIM] Initializing optimized per-KC networks for {self.n_kcs} KCs")
            
            # Shared encoder for all KC computations (major optimization)
            self.shared_kc_encoder = nn.Sequential(
                nn.Linear(summary_input_dim, final_fc_dim * 2),  # Larger hidden dim
                nn.LayerNorm(final_fc_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(final_fc_dim * 2, final_fc_dim),
                nn.LayerNorm(final_fc_dim),
                nn.ReLU()
            )
            
            # Multi-head outputs: single forward pass → all KC predictions
            self.kc_theta_head = nn.Linear(final_fc_dim, self.n_kcs)     # All θ parameters
            self.kc_beta_head = nn.Sequential(                          # All β parameters  
                nn.Linear(key_dim, final_fc_dim // 2),
                nn.Tanh(),
                nn.Linear(final_fc_dim // 2, self.n_kcs),
                nn.Tanh()
            )
            self.kc_alpha_head = nn.Sequential(                         # All α parameters
                nn.Linear(discrimination_input_dim, final_fc_dim // 2),
                nn.ReLU(),
                nn.Linear(final_fc_dim // 2, self.n_kcs),
                nn.Softplus()  # Ensure positive discrimination
            )
            
            print(f"[OPTIM] Replaced {self.n_kcs * 3} individual networks with 1 shared encoder + 3 multi-head outputs")
            print(f"[OPTIM] Expected speedup: ~{self.n_kcs // 3}x for per-KC computation")
        
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
        
        # Initialize optimized per-KC networks if enabled
        if self.per_kc_mode and self.n_kcs > 0:
            # Shared encoder
            for module in self.shared_kc_encoder:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.constant_(module.bias, 0)
            
            # Multi-head outputs
            nn.init.kaiming_normal_(self.kc_theta_head.weight)
            nn.init.constant_(self.kc_theta_head.bias, 0)
            
            for module in self.kc_beta_head:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.constant_(module.bias, 0)
            
            for module in self.kc_alpha_head:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, q_data, qa_data, target_mask=None, training_mode=False):
        """
        Optimized forward pass with efficient per-KC computation.
        
        Args:
            q_data: Question IDs, shape (batch_size, seq_len)
            qa_data: Question-Answer IDs, shape (batch_size, seq_len)
            target_mask: Optional mask for valid positions
            
        Returns:
            tuple: (predictions, student_abilities, item_difficulties, discrimination_params, z_values, kc_info)
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
        
        # Process sequence step by step - optimized 2PL IRT with efficient per-KC
        predictions = []
        student_abilities = []
        item_difficulties = []
        discrimination_params = []
        z_values = []
        
        # Per-KC tracking with optimized computation
        kc_info = {}
        if self.per_kc_mode and self.n_kcs > 0:
            all_kc_abilities = []    # (batch_size, seq_len, n_kcs)
            all_kc_difficulties = [] # (batch_size, seq_len, n_kcs) 
            all_kc_discriminations = [] # (batch_size, seq_len, n_kcs)
            all_kc_z_values = []     # (batch_size, seq_len, n_kcs)
        
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
            
            # Global IRT parameters
            student_ability = self.student_ability_network(summary_vector)  # (batch_size, 1)
            question_difficulty = self.question_difficulty_network(q_t)     # (batch_size, 1)
            
            # Discrimination parameter (alpha) - 2PL: a = softplus(W_a [f_t; k_t] + b_a)
            discrimination_input = torch.cat([summary_vector, q_t], dim=1)  # [f_t; k_t]
            discrimination = self.discrimination_network(discrimination_input)  # (batch_size, 1)
            
            # Per-KC vs Global 2PL prediction logic
            if self.per_kc_mode and self.n_kcs > 0:
                # OPTIMIZED Per-KC computation with proper aggregation
                # Single forward pass through shared encoder for all KCs
                kc_shared_features = self.shared_kc_encoder(summary_input)  # (batch_size, final_fc_dim)
                
                # Multi-head outputs: get ALL KC parameters in single operations
                all_kc_thetas = self.kc_theta_head(kc_shared_features)    # (batch_size, n_kcs)
                all_kc_betas = self.kc_beta_head(q_t)                     # (batch_size, n_kcs)  
                all_kc_alphas = self.kc_alpha_head(discrimination_input)  # (batch_size, n_kcs)
                
                # Compute all KC z-values in single vectorized operation
                all_kc_z = all_kc_alphas * (self.ability_scale * all_kc_thetas - all_kc_betas)  # (batch_size, n_kcs)
                all_kc_probs = torch.sigmoid(all_kc_z)  # (batch_size, n_kcs)
                
                # VECTORIZED per-KC aggregation (major speedup!)
                prediction = self._vectorized_kc_aggregation(all_kc_probs, q_data, t)
                
                # Use aggregated values for global tracking (mean across relevant KCs)
                # This allows backward compatibility with visualization
                student_ability = torch.mean(all_kc_thetas, dim=1, keepdim=True)  # (batch_size, 1)
                question_difficulty = torch.mean(all_kc_betas, dim=1, keepdim=True)  # (batch_size, 1)
                discrimination = torch.mean(all_kc_alphas, dim=1, keepdim=True)  # (batch_size, 1)
                z_value = torch.mean(all_kc_z, dim=1, keepdim=True)  # (batch_size, 1)
                
                # Store per-KC results
                all_kc_abilities.append(all_kc_thetas)
                all_kc_difficulties.append(all_kc_betas)
                all_kc_discriminations.append(all_kc_alphas)
                all_kc_z_values.append(all_kc_z)
                
            else:
                # Global 2PL prediction (fallback when no Q-matrix)
                # Discrimination parameter (alpha) - 2PL: a = softplus(W_a [f_t; k_t] + b_a)
                discrimination = self.discrimination_network(discrimination_input)  # (batch_size, 1)
                
                # 2PL prediction: z = a * (ability_scale * theta - beta)
                z_value = discrimination * (self.ability_scale * student_ability - question_difficulty)
                prediction = torch.sigmoid(z_value)
            
            # Store step results  
            if prediction.dim() == 1:
                prediction = prediction.unsqueeze(-1)  # (batch_size, 1)
            predictions.append(prediction)
            student_abilities.append(student_ability)
            item_difficulties.append(question_difficulty)
            discrimination_params.append(discrimination)
            z_values.append(z_value)
            
            # 4. Write (update memory for next step)
            self.memory.write(correlation_weight, qa_t)
        
        # Stack predictions (always needed)
        predictions = torch.stack(predictions, dim=1).squeeze(-1)  # (batch_size, seq_len)
        
        # OPTIMIZATION: Skip expensive IRT statistics during training for major speedup
        # All IRT stats are still available via compute_irt_statistics() after training
        if training_mode:
            return predictions
        
        # Stack all step results (only for evaluation/visualization)
        student_abilities = torch.stack(student_abilities, dim=1).squeeze(-1)  # (batch_size, seq_len)
        item_difficulties = torch.stack(item_difficulties, dim=1).squeeze(-1)  # (batch_size, seq_len)
        discrimination_params = torch.stack(discrimination_params, dim=1).squeeze(-1)  # (batch_size, seq_len)
        z_values = torch.stack(z_values, dim=1).squeeze(-1)  # (batch_size, seq_len)
        
        # Create kc_info for visualization access
        if self.per_kc_mode and self.n_kcs > 0:
            kc_info = {
                'all_kc_thetas': torch.stack(all_kc_abilities, dim=1),      # (batch_size, seq_len, n_kcs)
                'all_kc_betas': torch.stack(all_kc_difficulties, dim=1),    # (batch_size, seq_len, n_kcs)
                'all_kc_alphas': torch.stack(all_kc_discriminations, dim=1), # (batch_size, seq_len, n_kcs)
                'all_kc_z_values': torch.stack(all_kc_z_values, dim=1),     # (batch_size, seq_len, n_kcs)
                'q_to_kc': self.q_to_kc,
                'kc_names': self.kc_names,
                'n_kcs': self.n_kcs
            }
        else:
            kc_info = {
                'q_to_kc': {},
                'kc_names': {},
                'n_kcs': 0
            }
        
        # Store discrimination in kc_info for visualization access
        kc_info['discrimination_params'] = discrimination_params
        
        return predictions, student_abilities, item_difficulties, discrimination_params, z_values, kc_info
    
    def compute_irt_statistics(self, q_data, qa_data, target_mask=None):
        """
        Compute full IRT statistics efficiently AFTER training.
        This provides all the IRT parameters you need for analysis without slowing down training.
        
        Args:
            q_data: Question IDs, shape (batch_size, seq_len)
            qa_data: Question-Answer IDs, shape (batch_size, seq_len)
            target_mask: Optional mask for valid positions
            
        Returns:
            dict: Complete IRT statistics including theta, alpha, beta, z-values, KC info
        """
        # Run full forward pass (not training mode) to get all statistics
        with torch.no_grad():
            predictions, student_abilities, item_difficulties, discrimination_params, z_values, kc_info = self.forward(
                q_data, qa_data, target_mask, training_mode=False
            )
        
        # Convert to numpy for easy analysis
        irt_stats = {
            'predictions': predictions.cpu().numpy(),
            'student_abilities': student_abilities.cpu().numpy(),  # theta
            'item_difficulties': item_difficulties.cpu().numpy(),  # beta
            'discrimination_params': discrimination_params.cpu().numpy(),  # alpha
            'z_values': z_values.cpu().numpy(),
            'per_kc_mode': self.per_kc_mode,
            'n_kcs': self.n_kcs,
            'q_to_kc': self.q_to_kc,
            'kc_names': self.kc_names
        }
        
        # Add per-KC statistics if available
        if self.per_kc_mode and 'all_kc_thetas' in kc_info:
            irt_stats.update({
                'all_kc_thetas': kc_info['all_kc_thetas'].cpu().numpy(),
                'all_kc_betas': kc_info['all_kc_betas'].cpu().numpy(),
                'all_kc_alphas': kc_info['all_kc_alphas'].cpu().numpy(),
                'all_kc_z_values': kc_info['all_kc_z_values'].cpu().numpy()
            })
        
        return irt_stats
    
    def extract_trained_parameters(self, test_data_loader=None):
        """
        DYNAMIC IRT Parameter Extraction - Adapts to any dataset automatically.
        
        IRT Parameter Definitions:
        - Alpha (discrimination): Per-question parameter indicating how well the question distinguishes abilities
        - Beta (difficulty): Per-question parameter indicating question difficulty level  
        - Theta (ability): Per-student parameter indicating student knowledge/ability level
        
        This method dynamically detects the actual number of questions and students from the data.
        
        Args:
            test_data_loader: DataLoader with student-question interactions (required for proper extraction)
            
        Returns:
            dict: Proper IRT parameters with dynamically determined dimensions
        """
        if test_data_loader is None:
            # Return placeholder estimates with warning - cannot determine actual dimensions without data
            return {
                'alpha_estimates': np.ones(self.n_questions),  # Model's configured question count
                'beta_estimates': np.zeros(self.n_questions),
                'theta_estimates': np.array([0.0]),  # Unknown student count
                'n_questions': self.n_questions,
                'n_students': 0,  # Unknown without data
                'per_kc_mode': self.per_kc_mode,
                'n_kcs': self.n_kcs,
                'extraction_method': 'placeholder_no_data',
                'warning': 'Extraction requires test_data_loader to determine actual dataset dimensions'
            }
        
        # DYNAMIC IRT EXTRACTION: Automatically discover dataset characteristics
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # Dynamic data structures - no hardcoded limits
            question_alpha_accumulator = {}  # question_id -> [alpha_values]
            question_beta_accumulator = {}   # question_id -> [beta_values] 
            student_theta_values = []        # [theta_values] for all students
            
            # Tracking for validation
            total_batches = len(test_data_loader)
            total_students_processed = 0
            unique_questions_seen = set()
            
            print(f"🔍 Dynamic IRT extraction: Processing {total_batches} batches...")
            
            for batch_idx, batch in enumerate(test_data_loader):
                q_data = batch['q_data'].to(device)  # (batch_size, seq_len)
                qa_data = batch['qa_data'].to(device)
                
                # Get model predictions with IRT components
                predictions, student_abilities, item_difficulties, discrimination_params, z_values, kc_info = self.forward(
                    q_data, qa_data, training_mode=False
                )
                
                # Process each student in the batch
                batch_size = q_data.shape[0]
                for student_idx in range(batch_size):
                    # Get valid positions for this student (non-padding)
                    student_mask = q_data[student_idx] > 0
                    
                    if student_mask.any():
                        # Extract per-student theta (ability)
                        student_theta = student_abilities[student_idx][student_mask].mean().cpu().item()
                        student_theta_values.append(student_theta)
                        total_students_processed += 1
                        
                        # Extract per-question alpha and beta for this student's interactions
                        student_questions = q_data[student_idx][student_mask].cpu().numpy()
                        student_alphas = discrimination_params[student_idx][student_mask].cpu().numpy()
                        student_betas = item_difficulties[student_idx][student_mask].cpu().numpy()
                        
                        # Accumulate per-question parameters
                        for q_id, alpha, beta in zip(student_questions, student_alphas, student_betas):
                            q_id = int(q_id)
                            unique_questions_seen.add(q_id)
                            
                            if q_id not in question_alpha_accumulator:
                                question_alpha_accumulator[q_id] = []
                                question_beta_accumulator[q_id] = []
                            
                            question_alpha_accumulator[q_id].append(alpha)
                            question_beta_accumulator[q_id].append(beta)
                
                # Progress reporting
                if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
                    progress = (batch_idx + 1) / total_batches * 100
                    print(f"  📊 Progress: {progress:.1f}% | Students: {total_students_processed} | Questions: {len(unique_questions_seen)}")
            
            # DYNAMIC AGGREGATION - No hardcoded dimensions
            print(f"\n📈 Dataset characteristics discovered:")
            print(f"   Students: {total_students_processed}")
            print(f"   Unique questions: {len(unique_questions_seen)}")
            print(f"   Question ID range: {min(unique_questions_seen)} - {max(unique_questions_seen)}")
            
            # Create final parameter arrays with discovered dimensions
            max_question_id = max(unique_questions_seen)
            
            # Initialize arrays for all question IDs (1-based indexing)
            alpha_estimates = np.ones(max_question_id)  # Default discrimination
            beta_estimates = np.zeros(max_question_id)  # Default difficulty
            
            # Fill in observed values
            questions_with_data = 0
            for q_id in unique_questions_seen:
                if q_id >= 1:  # Valid question ID (1-based)
                    q_idx = q_id - 1  # Convert to 0-based array index
                    if q_idx < len(alpha_estimates):
                        alpha_estimates[q_idx] = np.mean(question_alpha_accumulator[q_id])
                        beta_estimates[q_idx] = np.mean(question_beta_accumulator[q_id])
                        questions_with_data += 1
            
            # Student abilities (theta)
            theta_estimates = np.array(student_theta_values)
            
            # Validation checks
            extraction_quality = {
                'questions_with_data': questions_with_data,
                'questions_total': len(unique_questions_seen),
                'students_processed': total_students_processed,
                'coverage_ratio': questions_with_data / len(unique_questions_seen) if unique_questions_seen else 0
            }
            
            print(f"✅ Extraction completed: {questions_with_data}/{len(unique_questions_seen)} questions, {total_students_processed} students")
            
            return {
                'alpha_estimates': alpha_estimates,      # Per-question discrimination
                'beta_estimates': beta_estimates,        # Per-question difficulty  
                'theta_estimates': theta_estimates,      # Per-student ability
                'n_questions': len(alpha_estimates),     # Dynamically determined
                'n_students': len(theta_estimates),      # Dynamically determined
                'per_kc_mode': self.per_kc_mode,
                'n_kcs': self.n_kcs,
                'extraction_method': 'dynamic_data_based',
                'questions_observed': len(unique_questions_seen),
                'students_observed': total_students_processed,
                'question_id_range': [min(unique_questions_seen), max(unique_questions_seen)],
                'extraction_quality': extraction_quality,
                'dataset_characteristics': {
                    'max_question_id': max_question_id,
                    'total_interactions': sum(len(vals) for vals in question_alpha_accumulator.values()),
                    'avg_interactions_per_question': np.mean([len(vals) for vals in question_alpha_accumulator.values()]) if question_alpha_accumulator else 0
                }
            }

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
        
        # IMPORTANT: Only enable per-KC mode for STATICS which has a real multi-KC Q-matrix
        # All other datasets should use global mode for efficiency
        if 'Qmatrix' not in q_matrix_path or 'STATICS' not in q_matrix_path:
            print(f"[OPTIM] Treating as single KC dataset: {q_matrix_path}")
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
                            # Format: concept_name,question_id
                            parts = line.split(',')
                            if len(parts) >= 2:
                                try:
                                    kc_name = parts[0]
                                    q_id = int(parts[1])
                                    
                                    # Assign KC index based on unique concept names
                                    if kc_name not in kc_names:
                                        kc_names[kc_name] = len(kc_names)
                                    kc_idx = kc_names[kc_name]
                                    
                                    q_to_kc[q_id] = [kc_idx]
                                    max_qid = max(max_qid, q_id)
                                    max_kc = max(max_kc, kc_idx)
                                except ValueError:
                                    continue
                        else:
                            # Format: qid,sid
                            parts = line.split(',')
                            if len(parts) >= 2:
                                try:
                                    q_id = int(parts[0])
                                    kc_idx = int(parts[1])
                                    q_to_kc[q_id] = [kc_idx]
                                    max_qid = max(max_qid, q_id)
                                    max_kc = max(max_kc, kc_idx)
                                except ValueError:
                                    continue
                
                n_kcs = max_kc + 1
            else:
                return False, {}, {}, 0
            
            print(f"Successfully loaded Q-matrix: {len(q_to_kc)} questions, {n_kcs} knowledge components")
            return True, q_to_kc, kc_names, n_kcs
            
        except Exception as e:
            print(f"Warning: Failed to load Q-matrix from {q_matrix_path}: {e}")
            return False, {}, {}, 0
    
    def _setup_kc_mapping_tensors(self):
        """Pre-compute KC mappings as tensors for vectorized operations (one-time setup)."""
        device = next(self.parameters()).device
        
        # Create mapping tensor: (n_questions+1, n_kcs) with 1.0 where question belongs to KC
        max_q_id = max(self.q_to_kc.keys()) if self.q_to_kc else 0
        mapping_tensor = torch.zeros(max_q_id + 1, self.n_kcs, device=device)
        
        for q_id, kc_list in self.q_to_kc.items():
            if kc_list:
                mapping_tensor[q_id, kc_list] = 1.0
            else:
                mapping_tensor[q_id, 0] = 1.0  # Default to KC 0
        
        self._kc_mapping_tensor = mapping_tensor
        print(f"[OPTIM] Created vectorized KC mapping tensor: {mapping_tensor.shape}")
    
    def _vectorized_kc_aggregation(self, all_kc_probs, q_data, t):
        """
        Vectorized KC aggregation - eliminates Python loops for major speedup.
        
        Args:
            all_kc_probs: (batch_size, n_kcs) - probabilities for all KCs
            q_data: (batch_size, seq_len) - question IDs
            t: current timestep
            
        Returns:
            predictions: (batch_size,) - aggregated predictions
        """
        batch_size = q_data.shape[0]
        
        # Setup KC mapping tensors if not already done (one-time cost)
        if not hasattr(self, '_kc_mapping_tensor'):
            self._setup_kc_mapping_tensors()
        
        # Get question IDs for entire batch at timestep t
        q_ids = q_data[:, t]  # Shape: (batch_size,)
        
        # Clamp question IDs to valid range (handle any out-of-bounds)
        max_q_id = self._kc_mapping_tensor.shape[0] - 1
        q_ids = torch.clamp(q_ids, 0, max_q_id)
        
        # Vectorized KC lookup: get KC masks for all questions in batch
        kc_masks = self._kc_mapping_tensor[q_ids]  # Shape: (batch_size, n_kcs)
        
        # Vectorized aggregation using masked operations (GPU-native)
        masked_probs = all_kc_probs * kc_masks  # Element-wise multiplication
        kc_sums = torch.sum(masked_probs, dim=1)  # Sum over KCs: (batch_size,)
        kc_counts = torch.sum(kc_masks, dim=1)    # Count valid KCs: (batch_size,)
        
        # Handle division by zero (fallback to KC 0 if no valid KCs)
        kc_counts = torch.clamp(kc_counts, min=1.0)
        predictions = kc_sums / kc_counts  # Mean aggregation
        
        return predictions