# GPCM Extension for Deep-2PL: Implementation Plan

## Overview
Extend the current Deep-2PL model to support Generalized Partial Credit Model (GPCM) for polytomous knowledge tracing with K-category responses.

## 1. Data Format Support

### 1.1 Response Categories
- **Current**: Binary responses {0, 1}
- **Target**: K-category responses {0, 1, 2, ..., K-1}
- **Formats**:
  - `synthetic_PC`: Decimal scores in [0,1] discretized to K categories
  - `synthetic_OC`: Explicit ordered categories {0, 1, 2, ..., K-1}

### 1.2 Dataset Detection & Loading
- **File**: `data/dataloader.py`
- **Task**: Extend `UnifiedKnowledgeTracingDataset` to detect response categories
- **Implementation**:
  ```python
  def detect_response_categories(self, responses):
      """Detect number of response categories and format type"""
      unique_responses = set(responses)
      if all(0 <= r <= 1 for r in unique_responses):
          # Check if decimal (PC format) or binary
          if any(isinstance(r, float) and r != int(r) for r in unique_responses):
              return 'partial_credit', self._discretize_responses(responses)
          else:
              return 'binary' if len(unique_responses) <= 2 else 'ordered_categorical', unique_responses
      else:
          return 'ordered_categorical', unique_responses
  ```

### 1.3 Data Structure Changes
- **Current**: `target` tensor with binary values
- **Target**: `target` tensor with K-category values and `n_categories` metadata
- **Add**: Dataset configuration parameters:
  ```python
  self.n_categories = K  # Number of response categories
  self.response_type = 'binary' | 'partial_credit' | 'ordered_categorical'
  ```

## 2. Embedding Layer Extensions

### 2.1 Question-Answer Embedding Strategies (4 options from paper)
- **File**: `models/model_optim.py`
- **Location**: Create new `GpcmEmbedding` class

#### Strategy 1: Unordered Categories (R^(KQ))
```python
def unordered_embedding(self, q_data, r_data, n_categories):
    """For MCQ-style responses"""
    batch_size, seq_len = q_data.shape
    embedding = torch.zeros(batch_size, seq_len, n_categories * self.n_questions)
    for k in range(n_categories):
        mask = (r_data == k).unsqueeze(-1).expand(-1, -1, self.n_questions)
        embedding[:, :, k*self.n_questions:(k+1)*self.n_questions] = q_data * mask.float()
    return embedding
```

#### Strategy 2: Ordered Categories (R^(2Q)) - RECOMMENDED
```python
def ordered_embedding(self, q_data, r_data, n_categories):
    """For partial credit responses - most intuitive"""
    # Low component: (K-1-r_t)/(K-1) * q_t
    low_component = ((n_categories - 1 - r_data) / (n_categories - 1)).unsqueeze(-1) * q_data
    # High component: r_t/(K-1) * q_t  
    high_component = (r_data / (n_categories - 1)).unsqueeze(-1) * q_data
    return torch.cat([low_component, high_component], dim=-1)
```

#### Strategy 3: Linear Decay (R^(KQ))
```python
def linear_decay_embedding(self, q_data, r_data, n_categories):
    """With triangular weights around actual response"""
    batch_size, seq_len = q_data.shape
    embedding = torch.zeros(batch_size, seq_len, n_categories * self.n_questions)
    for k in range(n_categories):
        weight = torch.clamp(1 - torch.abs(k - r_data) / (n_categories - 1), min=0)
        embedding[:, :, k*self.n_questions:(k+1)*self.n_questions] = q_data * weight.unsqueeze(-1)
    return embedding
```

#### Strategy 4: Adjacent Weighting (R^(KQ))
```python
def adjacent_weighted_embedding(self, q_data, r_data, n_categories, alpha=0.7):
    """Weight actual response and adjacent categories"""
    batch_size, seq_len = q_data.shape
    embedding = torch.zeros(batch_size, seq_len, n_categories * self.n_questions)
    for k in range(n_categories):
        mask = (r_data == k)  # Exact match
        adjacent_mask = torch.abs(r_data - k) == 1  # Adjacent categories
        
        # Apply weights
        embedding[:, :, k*self.n_questions:(k+1)*self.n_questions] = (
            q_data * (mask.float() * alpha + adjacent_mask.float() * (1-alpha)).unsqueeze(-1)
        )
    return embedding
```

### 2.2 Embedding Layer Integration
```python
class GpcmEmbedding(nn.Module):
    def __init__(self, n_questions, n_categories, embedding_strategy='ordered', d_v=50):
        super().__init__()
        self.n_questions = n_questions
        self.n_categories = n_categories
        self.strategy = embedding_strategy
        
        # Determine embedding dimension based on strategy
        if embedding_strategy == 'ordered':
            self.input_dim = 2 * n_questions
        else:
            self.input_dim = n_categories * n_questions
            
        self.embedding = nn.Linear(self.input_dim, d_v)
        
    def forward(self, q_data, r_data):
        if self.strategy == 'ordered':
            x = self.ordered_embedding(q_data, r_data, self.n_categories)
        elif self.strategy == 'unordered':
            x = self.unordered_embedding(q_data, r_data, self.n_categories)
        elif self.strategy == 'linear_decay':
            x = self.linear_decay_embedding(q_data, r_data, self.n_categories)
        elif self.strategy == 'adjacent_weighted':
            x = self.adjacent_weighted_embedding(q_data, r_data, self.n_categories)
        return self.embedding(x)
```

## 3. GPCM Prediction Layer

### 3.1 IRT Parameter Generation
- **File**: `models/model_optim.py`
- **Task**: Extend prediction layer for K-category GPCM

```python
class GpcmPredictor(nn.Module):
    def __init__(self, d_k, d_v, n_categories, n_questions):
        super().__init__()
        self.n_categories = n_categories
        self.n_questions = n_questions
        
        # Fusion layer
        self.fc_fusion = nn.Linear(d_k + d_v, 64)
        
        # IRT parameters
        self.fc_theta = nn.Linear(64, 1)  # Ability (per student-question interaction)
        self.fc_alpha = nn.Linear(64, 1)  # Discrimination (per question)
        self.fc_beta = nn.Linear(d_k, n_categories-1)  # Difficulty thresholds (per question, per category)
        
    def forward(self, read_content, question_embed):
        # Fusion
        f_t = torch.tanh(self.fc_fusion(torch.cat([read_content, question_embed], dim=-1)))
        
        # IRT parameters
        theta = torch.tanh(self.fc_theta(f_t))  # [-1, 1] ability
        alpha = F.softplus(self.fc_alpha(f_t)) + 1e-6  # >0 discrimination  
        beta = self.fc_beta(question_embed)  # K-1 difficulty thresholds
        
        # GPCM probability calculation
        probs = self.gpcm_probability(theta, alpha, beta)
        return probs
    
    def gpcm_probability(self, theta, alpha, beta):
        """Calculate GPCM probabilities for each category"""
        batch_size, seq_len, _ = theta.shape
        probs = torch.zeros(batch_size, seq_len, self.n_categories)
        
        # Calculate cumulative logits for each category
        cumulative_logits = []
        for k in range(self.n_categories):
            if k == 0:
                cumulative_logits.append(torch.zeros_like(theta.squeeze(-1)))
            else:
                # Sum from 0 to k-1: Σ(α(θ - β_h))
                cum_sum = torch.zeros_like(theta.squeeze(-1))
                for h in range(k):
                    cum_sum += alpha.squeeze(-1) * (theta.squeeze(-1) - beta[:, :, h])
                cumulative_logits.append(cum_sum)
        
        # Convert to probabilities using softmax
        cumulative_logits = torch.stack(cumulative_logits, dim=-1)  # [batch, seq, K]
        probs = F.softmax(cumulative_logits, dim=-1)
        
        return probs
```

### 3.2 Model Integration
- **File**: `models/model_optim.py`
- **Class**: `OptimizedDeepIRTModel`
- **Task**: Replace binary prediction with GPCM prediction

```python
def __init__(self, n_questions, n_categories=2, embedding_strategy='ordered', **kwargs):
    # ... existing initialization ...
    
    self.n_categories = n_categories
    self.is_gpcm = n_categories > 2
    
    if self.is_gpcm:
        self.qa_embedding = GpcmEmbedding(n_questions, n_categories, embedding_strategy)
        self.predictor = GpcmPredictor(self.d_k, self.d_v, n_categories, n_questions)
    else:
        # Keep existing binary setup
        self.qa_embedding = nn.Linear(2 * n_questions, self.d_v)
        self.predictor = IRT2PLPredictor(...)
```

## 4. Loss Function Implementation

### 4.1 Ordinal Loss Function
- **File**: `models/model_optim.py`
- **Task**: Implement ordinal loss from paper

```python
def ordinal_loss(predictions, targets, n_categories):
    """
    Ordinal loss for GPCM as defined in paper
    L = -Σ Σ Σ [I(y≤k)log(P(Y≤k)) + I(y>k)log(1-P(Y≤k))]
    """
    batch_size, seq_len, K = predictions.shape
    loss = 0.0
    
    for k in range(n_categories - 1):  # k from 0 to K-2
        # Cumulative probabilities P(Y <= k)
        cum_prob = torch.sum(predictions[:, :, :k+1], dim=-1)  # Sum from 0 to k
        cum_prob = torch.clamp(cum_prob, 1e-8, 1 - 1e-8)  # Numerical stability
        
        # Indicators
        indicator_leq = (targets <= k).float()  # I(y ≤ k)
        indicator_gt = (targets > k).float()   # I(y > k)
        
        # Loss components
        loss += -torch.sum(
            indicator_leq * torch.log(cum_prob) + 
            indicator_gt * torch.log(1 - cum_prob)
        )
    
    return loss / (batch_size * seq_len)
```

### 4.2 Alternative Loss Functions (for comparison)
```python
def categorical_crossentropy_loss(predictions, targets):
    """Standard categorical cross-entropy"""
    return F.cross_entropy(predictions.view(-1, predictions.size(-1)), targets.view(-1))

def mse_ordinal_loss(predictions, targets, n_categories):
    """MSE-based ordinal loss treating categories as continuous"""
    # Convert categorical predictions to expected values
    expected_values = torch.sum(predictions * torch.arange(n_categories).float().to(predictions.device), dim=-1)
    return F.mse_loss(expected_values, targets.float())
```

## 5. Evaluation Metrics

### 5.1 Polytomous-Specific Metrics
- **File**: `train.py`, `evaluate.py`
- **Task**: Implement appropriate evaluation metrics

```python
def calculate_gpcm_metrics(predictions, targets, n_categories):
    """Calculate metrics appropriate for polytomous responses"""
    metrics = {}
    
    # 1. Categorical Accuracy (exact match)
    pred_categories = torch.argmax(predictions, dim=-1)
    metrics['categorical_accuracy'] = (pred_categories == targets).float().mean().item()
    
    # 2. Ordinal Accuracy (within 1 category)
    ordinal_diff = torch.abs(pred_categories - targets)
    metrics['ordinal_accuracy_1'] = (ordinal_diff <= 1).float().mean().item()
    
    # 3. Mean Absolute Error (treating as regression)
    metrics['mae'] = ordinal_diff.float().mean().item()
    
    # 4. Quadratic Weighted Kappa
    metrics['qwk'] = quadratic_weighted_kappa(pred_categories, targets, n_categories)
    
    # 5. Per-category F1 scores
    for k in range(n_categories):
        tp = ((pred_categories == k) & (targets == k)).sum().float()
        fp = ((pred_categories == k) & (targets != k)).sum().float()
        fn = ((pred_categories != k) & (targets == k)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        metrics[f'f1_category_{k}'] = f1.item()
    
    return metrics
```

## 6. Training Pipeline Updates

### 6.1 Training Loop Modifications
- **File**: `train.py`
- **Task**: Update training loop for GPCM

```python
def train_epoch_gpcm(model, dataloader, optimizer, device, n_categories):
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for batch in dataloader:
        q_data = batch['q_data'].to(device)
        qa_data = batch['qa_data'].to(device) 
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        if model.is_gpcm:
            # Extract response categories from qa_data for GPCM
            r_data = extract_responses_gpcm(qa_data, model.n_questions, n_categories)
            predictions = model(q_data, qa_data, r_data)
            loss = ordinal_loss(predictions, target.long(), n_categories)
        else:
            predictions = model(q_data, qa_data)
            loss = F.binary_cross_entropy_with_logits(predictions, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if model.is_gpcm:
            all_predictions.append(predictions.detach())
            all_targets.append(target.detach())
    
    if model.is_gpcm:
        metrics = calculate_gpcm_metrics(
            torch.cat(all_predictions), 
            torch.cat(all_targets), 
            n_categories
        )
        return total_loss / len(dataloader), metrics
    else:
        return total_loss / len(dataloader), {}
```

## 7. Configuration and Model Selection

### 7.1 Model Configuration
- **File**: `utils/config.py`
- **Task**: Add GPCM configuration options

```python
GPCM_CONFIG = {
    'n_categories': 3,  # K categories {0, 1, 2, ..., K-1}
    'embedding_strategy': 'ordered',  # 'ordered' | 'unordered' | 'linear_decay' | 'adjacent_weighted'
    'loss_function': 'ordinal',  # 'ordinal' | 'categorical_ce' | 'mse_ordinal'
    'response_type': 'auto',  # 'auto' | 'partial_credit' | 'ordered_categorical'
    'discretization_method': 'equal_width',  # For partial credit: 'equal_width' | 'equal_frequency'
}
```

### 7.2 Model Factory Updates
- **File**: `models/model_selector.py`
- **Task**: Support GPCM model creation

```python
def create_gpcm_model(config, n_questions, n_categories=3):
    """Create GPCM-enabled model"""
    return OptimizedDeepIRTModel(
        n_questions=n_questions,
        n_categories=n_categories,
        embedding_strategy=config.get('embedding_strategy', 'ordered'),
        d_k=config.get('d_k', 50),
        d_v=config.get('d_v', 50),
        memory_size=config.get('memory_size', 20),
        final_fc_dim=config.get('final_fc_dim', 10)
    )
```

## 8. Testing and Validation

### 8.1 Unit Tests
- **File**: `tests/test_gpcm.py` (new)
- **Coverage**:
  - Embedding strategies
  - GPCM probability calculations
  - Loss function implementations
  - Metric calculations

### 8.2 Integration Tests
- **File**: `tests/test_gpcm_integration.py` (new)
- **Coverage**:
  - End-to-end training with synthetic data
  - Model saving/loading with GPCM
  - Evaluation pipeline

### 8.3 Validation Strategy
- **Datasets**: Test with synthetic_PC and synthetic_OC
- **Baselines**: Compare against:
  - Binary Deep-IRT (binarized responses)
  - Standard categorical models
  - Traditional GPCM (if available)

## 9. Implementation Priority

### Phase 1 (Core GPCM)
1. ✅ Create synthetic data generators
2. Implement ordered embedding strategy (most intuitive)
3. Implement GPCM predictor with basic probability calculation
4. Implement ordinal loss function
5. Update training loop for GPCM
6. Test with synthetic_OC data

### Phase 2 (Robustness)
1. Implement remaining embedding strategies
2. Add comprehensive evaluation metrics
3. Add proper configuration management
4. Test with synthetic_PC data
5. Performance optimization

### Phase 3 (Advanced Features)
1. Add visualization for polytomous responses
2. Implement IRT statistics extraction for GPCM
3. Add model comparison utilities
4. Documentation and examples

## 10. Expected Challenges & Solutions

### 10.1 Numerical Stability
- **Challenge**: GPCM probability calculation can be numerically unstable
- **Solution**: Use log-sum-exp trick, clamp probabilities

### 10.2 Memory Usage
- **Challenge**: Embedding dimensions increase significantly (KQ vs 2Q)
- **Solution**: Use ordered embedding (2Q) as default, optimize batch sizes

### 10.3 Convergence Issues
- **Challenge**: More complex loss function may be harder to optimize
- **Solution**: Start with smaller learning rates, use gradient clipping

### 10.4 Evaluation Complexity
- **Challenge**: Multiple valid metrics for ordinal data
- **Solution**: Implement comprehensive metric suite, focus on domain-appropriate metrics

---

## Next Steps
1. Create synthetic data generators (`synthetic_PC`, `synthetic_OC`)
2. Implement Phase 1 components
3. Run initial experiments with synthetic data
4. Iterate based on results

**Target**: Working GPCM extension ready for real polytomous datasets by end of implementation phases.