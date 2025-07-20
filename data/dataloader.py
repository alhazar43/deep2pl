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


class UnifiedKnowledgeTracingDataset(Dataset):
    """
    Unified dataset that auto-detects format and handles 5-fold splitting.
    Supports both pre-split (Yeung-style) and single file (orig-style) datasets.
    """
    
    def __init__(self, data_dir, dataset_name, split_type='train', fold=0, seq_len=50, n_questions=100, pad_val=-1):
        self.seq_len = seq_len
        self.n_questions = n_questions
        self.pad_val = pad_val
        self.dataset_path = os.path.join(data_dir, dataset_name)
        
        # Auto-detect format and load data
        self.q_seqs, self.qa_seqs, self.target_seqs = self._auto_detect_and_load(
            dataset_name, split_type, fold
        )
    
    def _auto_detect_and_load(self, dataset_name, split_type, fold):
        """Auto-detect dataset format and load accordingly."""
        
        # Check for pre-split format (Yeung-style)
        if split_type == 'train':
            presplit_file = os.path.join(self.dataset_path, f"{dataset_name}_{split_type}{fold}.csv")
        elif split_type == 'valid':
            presplit_file = os.path.join(self.dataset_path, f"{dataset_name}_{split_type}{fold}.csv")
        elif split_type == 'test':
            presplit_file = os.path.join(self.dataset_path, f"{dataset_name}_{split_type}.csv")
        
        if os.path.exists(presplit_file):
            print(f"Loading pre-split data from {presplit_file}")
            return self._load_presplit_data(presplit_file)
        
        # Check for single file format (orig-style)
        single_train_file = None
        single_test_file = None
        
        # Try different naming patterns
        possible_patterns = [
            (f"{dataset_name}_train.txt", f"{dataset_name}_test.txt"),
            (f"{dataset_name}_train.csv", f"{dataset_name}_test.csv"),
            ("builder_train.csv", "builder_test.csv"),  # assist2009 special case
            (f"{dataset_name.replace('2011', '')}_train.txt", f"{dataset_name.replace('2011', '')}_test.txt"),  # statics2011 -> static
            (f"{dataset_name.replace('statics', 'static')}_train.txt", f"{dataset_name.replace('statics', 'static')}_test.txt")  # statics2011 -> static2011
        ]
        
        for train_pattern, test_pattern in possible_patterns:
            train_path = os.path.join(self.dataset_path, train_pattern)
            test_path = os.path.join(self.dataset_path, test_pattern)
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                single_train_file = train_path
                single_test_file = test_path
                break
        
        if single_train_file and single_test_file:
            print(f"Loading single file data from {single_train_file} and {single_test_file}")
            return self._load_single_file_data(single_train_file, single_test_file, split_type, fold)
        
        raise FileNotFoundError(f"No valid data files found for dataset {dataset_name}")
    
    def _load_presplit_data(self, file_path):
        """Load data from pre-split format (CSV)."""
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
                    
                    # Create question-answer sequence (shifted to use previous interactions)
                    qa_seq = [0]  # Start with padding for first timestep
                    for q, a in zip(q_seq[:-1], a_seq[:-1]):  # Use previous (q,a) pairs
                        qa_value = q + a * self.n_questions
                        qa_seq.append(qa_value)
                    
                    # Create target sequence (current answer - fix for knowledge tracing)
                    target_seq = a_seq
                    
                    # Pad or truncate sequences
                    q_seq = pad_sequence(q_seq, self.seq_len)
                    qa_seq = pad_sequence(qa_seq, self.seq_len)
                    target_seq = pad_sequence(target_seq, self.seq_len)
                    
                    q_seqs.append(q_seq)
                    qa_seqs.append(qa_seq)
                    target_seqs.append(target_seq)
        
        return q_seqs, qa_seqs, target_seqs
    
    def _load_single_file_data(self, train_file, test_file, split_type, fold):
        """Load data from single file format and apply 5-fold splitting."""
        
        # Determine file extension and load accordingly
        if train_file.endswith('.csv'):
            all_data = self._load_csv_format(train_file, test_file, split_type)
        else:
            all_data = self._load_txt_format(train_file, test_file, split_type)
        
        # Apply 5-fold splitting if needed
        if split_type in ['train', 'valid']:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            all_indices = list(range(len(all_data)))
            
            fold_splits = list(kf.split(all_indices))
            train_indices, valid_indices = fold_splits[fold]
            
            if split_type == 'train':
                selected_data = [all_data[i] for i in train_indices]
            else:  # valid
                selected_data = [all_data[i] for i in valid_indices]
        else:  # test
            selected_data = all_data
        
        # Extract sequences
        q_seqs = [item[0] for item in selected_data]
        qa_seqs = [item[1] for item in selected_data]
        target_seqs = [item[2] for item in selected_data]
        
        return q_seqs, qa_seqs, target_seqs
    
    def _load_csv_format(self, train_file, test_file, split_type):
        """Load CSV format data."""
        if split_type == 'test':
            file_to_load = test_file
        else:
            file_to_load = train_file
        
        # Special handling for assist2009 builder format (3-line format in CSV)
        if "builder" in file_to_load:
            return self._load_builder_csv_format(file_to_load)
        
        # Load CSV data
        try:
            df = pd.read_csv(file_to_load)
        except Exception as e:
            print(f"Error loading CSV file {file_to_load}: {e}")
            raise
        
        all_data = []
        
        # Group by student if column exists, otherwise use row index
        if 'student_id' in df.columns:
            student_groups = df.groupby('student_id')
        else:
            # For assist2009 builder format, create student groups based on sequences
            student_groups = self._parse_builder_format(df)
        
        for student_id, group in student_groups:
            if isinstance(group, pd.DataFrame):
                # Standard CSV format
                if 'timestamp' in group.columns:
                    group = group.sort_values('timestamp')
                
                q_seq = group['question_id'].tolist() if 'question_id' in group.columns else group.iloc[:, 0].tolist()
                a_seq = group['correct'].tolist() if 'correct' in group.columns else group.iloc[:, 1].tolist()
            else:
                # Builder format
                q_seq, a_seq = group
            
            # Create question-answer sequence (shifted to use previous interactions)
            qa_seq = [0]  # Start with padding for first timestep
            for q, a in zip(q_seq[:-1], a_seq[:-1]):  # Use previous (q,a) pairs
                if a == 1:
                    qa_seq.append(q + self.n_questions)  # Correct answer
                else:
                    qa_seq.append(q)  # Incorrect answer
            
            # Create target sequence (current answer - fix for knowledge tracing)
            target_seq = a_seq
            
            # Pad or truncate sequences
            q_seq = pad_sequence(q_seq, self.seq_len)
            qa_seq = pad_sequence(qa_seq, self.seq_len)
            target_seq = pad_sequence(target_seq, self.seq_len)
            
            all_data.append((q_seq, qa_seq, target_seq))
        
        return all_data
    
    def _parse_builder_format(self, df):
        """Parse assist2009 builder format."""
        # This is a simplified parser for the builder format
        # Assumes the CSV has sequences separated by empty rows or specific patterns
        groups = []
        current_q_seq = []
        current_a_seq = []
        
        for _, row in df.iterrows():
            # Try to parse as question_id, correct
            try:
                q_id = int(row.iloc[0])
                correct = int(row.iloc[1])
                current_q_seq.append(q_id + 1)  # +1 for 1-based indexing
                current_a_seq.append(correct)
            except:
                # End of sequence or invalid row
                if current_q_seq:
                    groups.append((len(groups), (current_q_seq, current_a_seq)))
                    current_q_seq = []
                    current_a_seq = []
        
        # Add final sequence
        if current_q_seq:
            groups.append((len(groups), (current_q_seq, current_a_seq)))
        
        return groups
    
    def _load_builder_csv_format(self, file_to_load):
        """Load assist2009 builder CSV format (3-line format stored as CSV)."""
        all_data = []
        
        with open(file_to_load, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Parse sequence length
            try:
                seq_len = int(lines[i].strip())
                i += 1
            except:
                i += 1
                continue
            
            # Parse question sequence
            try:
                q_line = lines[i].strip()
                if q_line.endswith(','):
                    q_line = q_line[:-1]  # Remove trailing comma
                q_seq = [int(x) + 1 for x in q_line.split(',') if x.strip()]  # +1 for 1-based indexing
                i += 1
            except:
                i += 1
                continue
            
            # Parse answer sequence
            try:
                a_line = lines[i].strip()
                if a_line.endswith(','):
                    a_line = a_line[:-1]  # Remove trailing comma
                a_seq = [int(x) for x in a_line.split(',') if x.strip()]
                i += 1
            except:
                i += 1
                continue
            
            # Create question-answer sequence (shifted to use previous interactions)
            qa_seq = [0]  # Start with padding for first timestep
            for q, a in zip(q_seq[:-1], a_seq[:-1]):  # Use previous (q,a) pairs
                if a == 1:
                    qa_seq.append(q + self.n_questions)  # Correct answer
                else:
                    qa_seq.append(q)  # Incorrect answer
            
            # Create target sequence (current answer - fix for knowledge tracing)
            target_seq = a_seq
            
            # Pad or truncate sequences
            q_seq = pad_sequence(q_seq, self.seq_len)
            qa_seq = pad_sequence(qa_seq, self.seq_len)
            target_seq = pad_sequence(target_seq, self.seq_len)
            
            all_data.append((q_seq, qa_seq, target_seq))
        
        return all_data
    
    def _load_txt_format(self, train_file, test_file, split_type):
        """Load TXT format data."""
        if split_type == 'test':
            file_to_load = test_file
        else:
            file_to_load = train_file
        
        all_data = []
        
        with open(file_to_load, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Parse sequence length
            try:
                seq_len = int(lines[i].strip())
                i += 1
            except:
                i += 1
                continue
            
            # Parse question sequence
            try:
                q_seq = [int(x) + 1 for x in lines[i].strip().split(',')]  # +1 for 1-based indexing
                i += 1
            except:
                i += 1
                continue
            
            # Parse answer sequence
            try:
                a_seq = [int(x) for x in lines[i].strip().split(',')]
                i += 1
            except:
                i += 1
                continue
            
            # Create question-answer sequence (shifted to use previous interactions)
            qa_seq = [0]  # Start with padding for first timestep
            for q, a in zip(q_seq[:-1], a_seq[:-1]):  # Use previous (q,a) pairs
                if a == 1:
                    qa_seq.append(q + self.n_questions)  # Correct answer
                else:
                    qa_seq.append(q)  # Incorrect answer
            
            # Create target sequence (current answer - fix for knowledge tracing)
            target_seq = a_seq
            
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


def create_datasets(data_dir='./data', dataset_name='assist2015', seq_len=50, n_questions=100, k_fold=5, fold_idx=0):
    """
    Create train, validation, and test datasets with auto-detection.
    
    Args:
        data_dir: Directory containing the datasets
        dataset_name: Name of the dataset
        seq_len: Maximum sequence length
        n_questions: Number of questions in the dataset
        k_fold: Number of folds for cross-validation
        fold_idx: Which fold to use (0 to k_fold-1)
        
    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    
    train_dataset = UnifiedKnowledgeTracingDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split_type='train',
        fold=fold_idx,
        seq_len=seq_len,
        n_questions=n_questions
    )
    
    valid_dataset = UnifiedKnowledgeTracingDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split_type='valid',
        fold=fold_idx,
        seq_len=seq_len,
        n_questions=n_questions
    )
    
    test_dataset = UnifiedKnowledgeTracingDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split_type='test',
        fold=0,  # Test set doesn't have folds
        seq_len=seq_len,
        n_questions=n_questions
    )
    
    return train_dataset, valid_dataset, test_dataset


def get_dataset_info(dataset_name, data_dir='./data'):
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory containing the datasets
        
    Returns:
        dict: Dataset information including n_questions, max_seq_len, etc.
    """
    dataset_path = os.path.join(data_dir, dataset_name)
    
    # Try to determine n_questions and other info
    info = {
        'dataset_name': dataset_name,
        'n_questions': 100,  # Default
        'max_seq_len': 50,   # Default
        'has_qmatrix': False,
        'qmatrix_path': None,
        'skill_mapping_path': None
    }
    
    # Check for Q-matrix
    qmatrix_files = [
        'Qmatrix.csv',
        f'{dataset_name}_qid_sid',
        'conceptname_question_id.csv'
    ]
    
    for qfile in qmatrix_files:
        qpath = os.path.join(dataset_path, qfile)
        if os.path.exists(qpath):
            info['has_qmatrix'] = True
            info['qmatrix_path'] = qpath
            break
    
    # Check for skill mapping
    skill_files = [
        f'{dataset_name}_skill_mapping.txt',
        'clustered_skill_name.txt'
    ]
    
    for sfile in skill_files:
        spath = os.path.join(dataset_path, sfile)
        if os.path.exists(spath):
            info['skill_mapping_path'] = spath
            break
    
    # Try to determine n_questions from data
    try:
        # Create a small test dataset to determine n_questions
        test_dataset = UnifiedKnowledgeTracingDataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            split_type='test',
            fold=0,
            seq_len=50,
            n_questions=1000  # Use large number initially
        )
        
        # Find max question ID
        max_q = 0
        for i in range(min(10, len(test_dataset))):  # Check first 10 samples
            sample = test_dataset[i]
            q_data = sample['q_data']
            valid_qs = q_data[q_data > 0]
            if len(valid_qs) > 0:
                max_q = max(max_q, valid_qs.max().item())
        
        if max_q > 0:
            info['n_questions'] = max_q
    except Exception as e:
        print(f"Warning: Could not determine n_questions for {dataset_name}: {e}")
    
    return info


# Example usage
if __name__ == "__main__":
    # Test the unified dataloader
    datasets = ['assist2015', 'STATICS', 'assist2009_updated', 'assist2017', 'statics2011']
    
    for dataset in datasets:
        try:
            print(f"\nTesting dataset: {dataset}")
            info = get_dataset_info(dataset)
            print(f"Info: {info}")
            
            train_dataset, valid_dataset, test_dataset = create_datasets(
                dataset_name=dataset,
                n_questions=info['n_questions']
            )
            
            print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
            
        except Exception as e:
            print(f"Error with {dataset}: {e}")