import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd


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
            q_seq = self._pad_sequence(q_seq, self.seq_len)
            qa_seq = self._pad_sequence(qa_seq, self.seq_len)
            target_seq = self._pad_sequence(target_seq, self.seq_len)
            
            q_seqs.append(q_seq)
            qa_seqs.append(qa_seq)
            target_seqs.append(target_seq)
            
        return q_seqs, qa_seqs, target_seqs
    
    def _pad_sequence(self, seq, target_len):
        """Pad or truncate sequence to target length."""
        if len(seq) >= target_len:
            return seq[:target_len]
        else:
            return seq + [0] * (target_len - len(seq))
    
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
            q_seq = self._pad_sequence(q_seq, self.seq_len)
            qa_seq = self._pad_sequence(qa_seq, self.seq_len)
            target_seq = self._pad_sequence(target_seq, self.seq_len)
            
            q_seqs.append(q_seq)
            qa_seqs.append(qa_seq)
            target_seqs.append(target_seq)
        
        return q_seqs, qa_seqs, target_seqs
    
    def _pad_sequence(self, seq, target_len):
        """Pad or truncate sequence to target length."""
        if len(seq) >= target_len:
            return seq[:target_len]
        else:
            return seq + [0] * (target_len - len(seq))
    
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