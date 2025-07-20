import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import itertools
import os


def pad_sequence(seq, target_len, pad_val=0):
    """Pad or truncate sequence to target length."""
    return seq[:target_len] if len(seq) >= target_len else seq + [pad_val] * (target_len - len(seq))


class KnowledgeTracingDataset(Dataset):
    """
    Dataset for knowledge tracing with question-answer sequences.
    """
    
    def __init__(self, data_path, seq_len, n_questions, pad_val=-1):
        self.seq_len = seq_len
        self.n_questions = n_questions
        self.pad_val = pad_val
        
        # Load and process data
        self.q_seqs, self.qa_seqs, self.target_seqs = self._load_data(data_path)
        
    def _load_data(self, data_path):
        """
        Load data from file. Expected format:
        - Each line represents one student sequence
        - First number is sequence length
        - Second line is question sequence
        - Third line is answer sequence
        """
        q_seqs = []
        qa_seqs = []
        target_seqs = []
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            # Parse sequence length
            seq_len = int(lines[i].strip())
            i += 1
            
            # Parse question sequence
            q_seq = list(map(int, lines[i].strip().split(',')))
            i += 1
            
            # Parse answer sequence
            a_seq = list(map(int, lines[i].strip().split(',')))
            i += 1
            
            # Create question-answer sequence
            qa_seq = []
            for q, a in zip(q_seq, a_seq):
                if a == 1:
                    qa_seq.append(q + self.n_questions)  # Correct answer
                else:
                    qa_seq.append(q)  # Incorrect answer
            
            # Create target sequence (shifted by 1)
            target_seq = a_seq[1:] + [self.pad_val]
            
            # Pad or truncate sequences
            q_seq = pad_sequence(q_seq, self.seq_len)
            qa_seq = pad_sequence(qa_seq, self.seq_len)
            target_seq = pad_sequence(target_seq, self.seq_len)
            
            q_seqs.append(q_seq)
            qa_seqs.append(qa_seq)
            target_seqs.append(target_seq)
            
        return q_seqs, qa_seqs, target_seqs
    
    
    def __len__(self):
        return len(self.q_seqs)
    
    def __getitem__(self, idx):
        return {
            'q_data': torch.tensor(self.q_seqs[idx], dtype=torch.long),
            'qa_data': torch.tensor(self.qa_seqs[idx], dtype=torch.long),
            'target': torch.tensor(self.target_seqs[idx], dtype=torch.float)
        }


class CSVKnowledgeTracingDataset(Dataset):
    """
    Dataset for knowledge tracing from CSV format.
    Expected columns: student_id, question_id, correct
    """
    
    def __init__(self, csv_path, seq_len, n_questions, pad_val=-1):
        self.seq_len = seq_len
        self.n_questions = n_questions
        self.pad_val = pad_val
        
        # Load and process data
        self.q_seqs, self.qa_seqs, self.target_seqs = self._load_csv_data(csv_path)
    
    def _load_csv_data(self, csv_path):
        """Load data from CSV file."""
        df = pd.read_csv(csv_path)
        
        q_seqs = []
        qa_seqs = []
        target_seqs = []
        
        # Group by student
        for student_id, group in df.groupby('student_id'):
            # Sort by timestamp if available, otherwise by order
            if 'timestamp' in group.columns:
                group = group.sort_values('timestamp')
            
            q_seq = group['question_id'].tolist()
            a_seq = group['correct'].tolist()
            
            # Create question-answer sequence
            qa_seq = []
            for q, a in zip(q_seq, a_seq):
                if a == 1:
                    qa_seq.append(q + self.n_questions)  # Correct answer
                else:
                    qa_seq.append(q)  # Incorrect answer
            
            # Create target sequence (next answer)
            target_seq = a_seq[1:] + [self.pad_val]
            
            # Pad or truncate sequences
            q_seq = pad_sequence(q_seq, self.seq_len)
            qa_seq = pad_sequence(qa_seq, self.seq_len)
            target_seq = pad_sequence(target_seq, self.seq_len)
            
            q_seqs.append(q_seq)
            qa_seqs.append(qa_seq)
            target_seqs.append(target_seq)
        
        return q_seqs, qa_seqs, target_seqs
    
    
    def __len__(self):
        return len(self.q_seqs)
    
    def __getitem__(self, idx):
        return {
            'q_data': torch.tensor(self.q_seqs[idx], dtype=torch.long),
            'qa_data': torch.tensor(self.qa_seqs[idx], dtype=torch.long),
            'target': torch.tensor(self.target_seqs[idx], dtype=torch.float)
        }


class YeungStyleDataset(Dataset):
    """
    Dataset for Yeung-style pre-split data (train0.csv, valid0.csv, etc.)
    """
    
    def __init__(self, data_dir, dataset_name, split_type='train', fold=0, seq_len=50, n_questions=100, pad_val=-1):
        self.seq_len = seq_len
        self.n_questions = n_questions
        self.pad_val = pad_val
        
        # Load data from pre-split files
        if split_type == 'train':
            file_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_{split_type}{fold}.csv")
        elif split_type == 'valid':
            file_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_{split_type}{fold}.csv")
        elif split_type == 'test':
            file_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_{split_type}.csv")
        else:
            raise ValueError(f"Unknown split_type: {split_type}")
            
        self.q_seqs, self.qa_seqs, self.target_seqs = self._load_yeung_data(file_path)
    
    def _load_yeung_data(self, file_path):
        """Load data from Yeung-style format (similar to deep-yeung load_data.py)"""
        q_seqs = []
        qa_seqs = []
        target_seqs = []
        
        with open(file_path, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                # Skip sequence length line
                if line_idx % 3 == 0:
                    continue
                # Handle question line
                elif line_idx % 3 == 1:
                    q_seq = [int(x) for x in line.split(',')]
                # Handle answer line
                elif line_idx % 3 == 2:
                    a_seq = [int(x) for x in line.split(',')]
                    
                    # Create question-answer sequence (Yeung style)
                    qa_seq = []
                    for q, a in zip(q_seq, a_seq):
                        qa_value = q + a * self.n_questions
                        qa_seq.append(qa_value)
                    
                    # Create target sequence (next answer)
                    target_seq = a_seq[1:] + [self.pad_val]
                    
                    # Pad or truncate sequences
                    q_seq = pad_sequence(q_seq, self.seq_len)
                    qa_seq = pad_sequence(qa_seq, self.seq_len)
                    target_seq = pad_sequence(target_seq, self.seq_len)
                    
                    q_seqs.append(q_seq)
                    qa_seqs.append(qa_seq)
                    target_seqs.append(target_seq)
        
        return q_seqs, qa_seqs, target_seqs
    
    
    def __len__(self):
        return len(self.q_seqs)
    
    def __getitem__(self, idx):
        return {
            'q_data': torch.tensor(self.q_seqs[idx], dtype=torch.long),
            'qa_data': torch.tensor(self.qa_seqs[idx], dtype=torch.long),
            'target': torch.tensor(self.target_seqs[idx], dtype=torch.float)
        }


class TorchStyleDataset(Dataset):
    """
    Dataset for dkvmn-torch style data with runtime k-fold splitting
    """
    
    def __init__(self, data_path, seq_len, n_questions, pad_val=-1, k_fold=5, fold_idx=0, split_type='train'):
        self.seq_len = seq_len
        self.n_questions = n_questions
        self.pad_val = pad_val
        
        # Load all data first
        all_data = self._load_torch_data(data_path)
        
        # Apply k-fold splitting if needed
        if split_type in ['train', 'valid']:
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
            all_indices = list(range(len(all_data)))
            
            fold_splits = list(kf.split(all_indices))
            train_indices, valid_indices = fold_splits[fold_idx]
            
            if split_type == 'train':
                selected_data = [all_data[i] for i in train_indices]
            else:  # valid
                selected_data = [all_data[i] for i in valid_indices]
        else:  # test
            selected_data = all_data
        
        # Extract sequences
        self.q_seqs = [item[0] for item in selected_data]
        self.qa_seqs = [item[1] for item in selected_data]
        self.target_seqs = [item[2] for item in selected_data]
    
    def _load_torch_data(self, data_path):
        """Load data from dkvmn-torch style format"""
        all_data = []
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Parse sequence length
            seq_len = int(lines[i].strip())
            i += 1
            
            # Parse question sequence
            q_seq = [int(x) + 1 for x in lines[i].strip().split(',')]  # +1 for 1-based indexing
            i += 1
            
            # Parse answer sequence
            a_seq = [int(x) for x in lines[i].strip().split(',')]
            i += 1
            
            # Create question-answer sequence (torch style)
            qa_seq = []
            for q, a in zip(q_seq, a_seq):
                if a == 1:
                    qa_seq.append(q + self.n_questions)  # Correct answer
                else:
                    qa_seq.append(q)  # Incorrect answer
            
            # Create target sequence (next answer)
            target_seq = a_seq[1:] + [self.pad_val]
            
            # Pad or truncate sequences
            q_seq = pad_sequence(q_seq, self.seq_len)
            qa_seq = pad_sequence(qa_seq, self.seq_len)
            target_seq = pad_sequence(target_seq, self.seq_len)
            
            all_data.append((q_seq, qa_seq, target_seq))
        
        return all_data
    
    
    def __len__(self):
        return len(self.q_seqs)
    
    def __getitem__(self, idx):
        return {
            'q_data': torch.tensor(self.q_seqs[idx], dtype=torch.long),
            'qa_data': torch.tensor(self.qa_seqs[idx], dtype=torch.long),
            'target': torch.tensor(self.target_seqs[idx], dtype=torch.float)
        }


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """Create DataLoader for knowledge tracing dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def create_datasets(data_style='yeung', data_dir='./data-yeung', dataset_name='assist2009_updated', 
                   seq_len=50, n_questions=100, k_fold=5, fold_idx=0):
    """
    Create train, validation, and test datasets based on the specified style.
    
    Args:
        data_style: 'yeung' (pre-split files) or 'torch' (runtime k-fold)
        data_dir: Directory containing the data
        dataset_name: Name of the dataset (for yeung style)
        seq_len: Maximum sequence length
        n_questions: Number of questions in the dataset
        k_fold: Number of folds for cross-validation (torch style)
        fold_idx: Which fold to use (0 to k_fold-1)
    
    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    
    if data_style == 'yeung':
        train_dataset = YeungStyleDataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            split_type='train',
            fold=fold_idx,
            seq_len=seq_len,
            n_questions=n_questions
        )
        
        valid_dataset = YeungStyleDataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            split_type='valid',
            fold=fold_idx,
            seq_len=seq_len,
            n_questions=n_questions
        )
        
        test_dataset = YeungStyleDataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            split_type='test',
            fold=0,  # Test set doesn't have folds
            seq_len=seq_len,
            n_questions=n_questions
        )
    
    elif data_style == 'torch':
        # Determine file paths for torch style
        if 'assist2015' in dataset_name:
            train_file = os.path.join(data_dir, 'assist2015', 'assist2015_train.txt')
            test_file = os.path.join(data_dir, 'assist2015', 'assist2015_test.txt')
        elif 'assist2009' in dataset_name:
            train_file = os.path.join(data_dir, 'assist2009', 'builder_train.csv')
            test_file = os.path.join(data_dir, 'assist2009', 'builder_test.csv')
        else:
            raise ValueError(f"Unknown dataset for torch style: {dataset_name}")
        
        train_dataset = TorchStyleDataset(
            data_path=train_file,
            seq_len=seq_len,
            n_questions=n_questions,
            k_fold=k_fold,
            fold_idx=fold_idx,
            split_type='train'
        )
        
        valid_dataset = TorchStyleDataset(
            data_path=train_file,
            seq_len=seq_len,
            n_questions=n_questions,
            k_fold=k_fold,
            fold_idx=fold_idx,
            split_type='valid'
        )
        
        test_dataset = TorchStyleDataset(
            data_path=test_file,
            seq_len=seq_len,
            n_questions=n_questions,
            split_type='test'
        )
    
    else:
        raise ValueError(f"Unknown data_style: {data_style}. Use 'yeung' or 'torch'")
    
    return train_dataset, valid_dataset, test_dataset


def get_data_info(data_path, format='txt'):
    """
    Get basic information about the dataset.
    
    Args:
        data_path: Path to data file
        format: 'txt' or 'csv'
    
    Returns:
        Dictionary with dataset statistics
    """
    if format == 'txt':
        max_questions = 0
        total_sequences = 0
        total_interactions = 0
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            seq_len = int(lines[i].strip())
            i += 1
            
            q_seq = list(map(int, lines[i].strip().split(',')))
            i += 1
            
            a_seq = list(map(int, lines[i].strip().split(',')))
            i += 1
            
            max_questions = max(max_questions, max(q_seq))
            total_sequences += 1
            total_interactions += len(q_seq)
        
        return {
            'n_questions': max_questions,
            'n_sequences': total_sequences,
            'total_interactions': total_interactions,
            'avg_seq_len': total_interactions / total_sequences
        }
    
    elif format == 'csv':
        df = pd.read_csv(data_path)
        return {
            'n_questions': df['question_id'].max(),
            'n_students': df['student_id'].nunique(),
            'n_sequences': df['student_id'].nunique(),
            'total_interactions': len(df),
            'avg_seq_len': len(df) / df['student_id'].nunique()
        }
    
    else:
        raise ValueError(f"Unsupported format: {format}")


# Example usage
if __name__ == "__main__":
    # Example with text format
    dataset = KnowledgeTracingDataset(
        data_path="data/train.txt",
        seq_len=50,
        n_questions=100
    )
    
    dataloader = create_dataloader(dataset, batch_size=32)
    
    # Print sample batch
    for batch in dataloader:
        print("Q data shape:", batch['q_data'].shape)
        print("QA data shape:", batch['qa_data'].shape)
        print("Target shape:", batch['target'].shape)
        break