import argparse
import json
import os


class Config:
    """Configuration class for Deep-IRT model."""
    
    def __init__(self):
        # Model parameters
        self.n_questions = 100
        self.memory_size = 50
        self.key_memory_state_dim = 50
        self.value_memory_state_dim = 200
        self.summary_vector_dim = 50
        self.q_embed_dim = 50
        self.qa_embed_dim = 200
        self.ability_scale = 3.0
        self.use_discrimination = False
        self.dropout_rate = 0.1
        
        # Training parameters
        self.batch_size = 32
        self.seq_len = 50
        self.learning_rate = 0.001
        self.n_epochs = 100
        self.max_grad_norm = 5.0
        self.weight_decay = 1e-5
        
        # Data parameters
        self.data_dir = "data"
        self.train_file = "train.txt"
        self.test_file = "test.txt"
        self.data_format = "txt"  # "txt" or "csv"
        
        # New data loading parameters
        self.data_style = "yeung"  # "yeung" (pre-split) or "torch" (runtime k-fold)
        self.dataset_name = "assist2009_updated"  # Dataset name for yeung style
        self.k_fold = 5  # Number of folds for cross-validation
        self.fold_idx = 0  # Which fold to use (0 to k_fold-1)
        
        # Training configuration  
        self.device = "auto"  # Will be set later in get_config()
        self.seed = 42
        self.save_dir = "checkpoints"
        self.log_dir = "logs"
        self.save_every = 10
        self.eval_every = 5
        
        # Logging
        self.verbose = True
        self.tensorboard = True
        
    def update_from_args(self, args):
        """Update config from command line arguments."""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:  # Only update if value is not None
                setattr(self, key, value)
    
    def save(self, path):
        """Save configuration to JSON file."""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load(self, path):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def __str__(self):
        config_str = "Configuration:\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"  {key}: {value}\n"
        return config_str


def get_config():
    """Get configuration from command line arguments."""
    parser = argparse.ArgumentParser(description='Deep-IRT Knowledge Tracing')
    
    # Model parameters
    parser.add_argument('--n_questions', type=int, default=None,
                        help='Number of questions in dataset')
    parser.add_argument('--memory_size', type=int, default=None,
                        help='Memory size for DKVMN')
    parser.add_argument('--key_memory_state_dim', type=int, default=None,
                        help='Key memory state dimension')
    parser.add_argument('--value_memory_state_dim', type=int, default=None,
                        help='Value memory state dimension')
    parser.add_argument('--summary_vector_dim', type=int, default=None,
                        help='Summary vector dimension')
    parser.add_argument('--q_embed_dim', type=int, default=None,
                        help='Question embedding dimension')
    parser.add_argument('--qa_embed_dim', type=int, default=None,
                        help='Question-answer embedding dimension')
    parser.add_argument('--ability_scale', type=float, default=None,
                        help='Ability scaling factor in IRT')
    parser.add_argument('--use_discrimination', action='store_true',
                        help='Use discrimination parameter (2PL model)')
    parser.add_argument('--dropout_rate', type=float, default=None,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory')
    parser.add_argument('--train_file', type=str, default=None,
                        help='Training data file')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Test data file')
    parser.add_argument('--data_format', type=str, default=None,
                        choices=['txt', 'csv'], help='Data format')
    
    # New data loading parameters
    parser.add_argument('--data_style', type=str, default=None,
                        choices=['yeung', 'torch'], 
                        help='Data loading style: yeung (pre-split) or torch (runtime k-fold)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Dataset name (for yeung style)')
    parser.add_argument('--k_fold', type=int, default=None,
                        help='Number of folds for cross-validation')
    parser.add_argument('--fold_idx', type=int, default=None,
                        help='Which fold to use (0 to k_fold-1)')
    
    # Training configuration
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory to save logs')
    parser.add_argument('--save_every', type=int, default=None,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=None,
                        help='Evaluate every N epochs')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Use TensorBoard logging')
    
    # Load/save config
    parser.add_argument('--config', type=str, default=None,
                        help='Load configuration from file')
    parser.add_argument('--save_config', type=str, default=None,
                        help='Save configuration to file')
    
    args = parser.parse_args()
    
    # Create config object
    config = Config()
    
    # Load from file if specified
    if args.config:
        config.load(args.config)
    
    # Update with command line arguments
    config.update_from_args(args)
    
    # Handle device selection
    if config.device == 'auto':
        import torch
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Save config if specified
    if args.save_config:
        config.save(args.save_config)
    
    return config


# Example configuration files
ASSIST2009_CONFIG = {
    "n_questions": 110,
    "memory_size": 50,
    "key_memory_state_dim": 50,
    "value_memory_state_dim": 200,
    "summary_vector_dim": 50,
    "batch_size": 32,
    "seq_len": 50,
    "learning_rate": 0.001,
    "n_epochs": 100,
    "train_file": "assist2009_train.txt",
    "test_file": "assist2009_test.txt"
}

ASSIST2015_CONFIG = {
    "n_questions": 100,
    "memory_size": 50,
    "key_memory_state_dim": 50,
    "value_memory_state_dim": 200,
    "summary_vector_dim": 50,
    "batch_size": 32,
    "seq_len": 50,
    "learning_rate": 0.001,
    "n_epochs": 100,
    "train_file": "assist2015_train.txt",
    "test_file": "assist2015_test.txt"
}

SYNTHETIC_CONFIG = {
    "n_questions": 50,
    "memory_size": 20,
    "key_memory_state_dim": 50,
    "value_memory_state_dim": 200,
    "summary_vector_dim": 50,
    "batch_size": 32,
    "seq_len": 50,
    "learning_rate": 0.001,
    "n_epochs": 50,
    "train_file": "synthetic_train.txt",
    "test_file": "synthetic_test.txt"
}