import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import itertools
import os

# Simple cache for loaded datasets to avoid redundant file I/O
_dataset_cache = {}


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
        """Auto-detect dataset format and load accordingly with caching."""
        
        # Create cache key
        cache_key = f"{dataset_name}_{split_type}_{fold}_{self.seq_len}_{self.n_questions}"
        
        # Check cache first
        if cache_key in _dataset_cache:
            return _dataset_cache[cache_key]
        
        # Check for pre-split format (Yeung-style)
        if split_type == 'train':
            presplit_file = os.path.join(self.dataset_path, f"{dataset_name}_{split_type}{fold}.csv")
        elif split_type == 'valid':
            presplit_file = os.path.join(self.dataset_path, f"{dataset_name}_{split_type}{fold}.csv")
        elif split_type == 'test':
            presplit_file = os.path.join(self.dataset_path, f"{dataset_name}_{split_type}.csv")
        
        if os.path.exists(presplit_file):
            print(f"Loading pre-split data from {presplit_file}")
            result = self._load_presplit_data(presplit_file)
            _dataset_cache[cache_key] = result
            return result
        
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
            result = self._load_single_file_data(single_train_file, single_test_file, split_type, fold)
            _dataset_cache[cache_key] = result
            return result
        
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
    Get information about a dataset with comprehensive question detection.
    
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
        'skill_mapping_path': None,
        'detection_method': 'default'
    }
    
    # First, get data-based detection (most reliable)
    n_q_from_data = 0
    try:
        n_q_from_data = _get_n_questions_from_data_fixed(dataset_name, data_dir)
        if n_q_from_data > 0:
            info['n_questions'] = n_q_from_data
            info['detection_method'] = 'comprehensive_data_scan'
    except Exception as e:
        print(f"Warning: Comprehensive data scan failed for {dataset_name}: {e}")
    
    # Check for Q-matrix and validate against data
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
            
            # Try to determine n_questions from Q-matrix
            try:
                n_q_from_qmatrix = _get_n_questions_from_qmatrix(qpath)
                if n_q_from_qmatrix > 0:
                    # Validate Q-matrix against data
                    if n_q_from_data > 0:
                        # Check if Q-matrix matches data (with tolerance for small differences)
                        if abs(n_q_from_qmatrix - n_q_from_data) <= 1:
                            # Use Q-matrix if very close to data
                            info['n_questions'] = n_q_from_qmatrix
                            info['detection_method'] = f'qmatrix_{qfile}_validated'
                            break
                        elif n_q_from_qmatrix > n_q_from_data * 2:
                            # Q-matrix seems to have phantom questions, ignore
                            print(f"Warning: Q-matrix {qfile} has {n_q_from_qmatrix} questions but data has {n_q_from_data}. Ignoring Q-matrix.")
                            continue
                        else:
                            # Use data-based detection as it's more reliable
                            print(f"Warning: Q-matrix {qfile} has {n_q_from_qmatrix} questions but data suggests {n_q_from_data}. Using data-based detection.")
                            continue
                    else:
                        # No data detection available, use Q-matrix
                        info['n_questions'] = n_q_from_qmatrix
                        info['detection_method'] = f'qmatrix_{qfile}'
                        break
            except Exception as e:
                print(f"Warning: Could not parse Q-matrix {qfile}: {e}")
    
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
    
    return info


def _get_n_questions_from_qmatrix(qmatrix_path):
    """Extract n_questions from Q-matrix file."""
    try:
        if qmatrix_path.endswith('.csv'):
            # Handle CSV Q-matrix
            df = pd.read_csv(qmatrix_path)
            if 'question_id' in df.columns:
                return df['question_id'].max()
            elif len(df.columns) >= 2:
                # Assume first column is question_id
                return df.iloc[:, 0].max()
            else:
                return 0
        else:
            # Handle text-based Q-matrix (e.g., qid_sid format)
            max_q_id = 0
            with open(qmatrix_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            q_id = int(parts[0])
                            max_q_id = max(max_q_id, q_id)
                        except ValueError:
                            continue
            return max_q_id
    except Exception:
        return 0



def _extract_question_ids_robust(filepath):
    """Robustly extract question IDs from any file format."""
    question_ids = set()
    
    try:
        # Try sequence format first (most common)
        if filepath.endswith('.txt') or filepath.endswith('.csv'):
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Check if it's sequence format: num_questions, question_ids, answers
            i = 0
            sequence_format_detected = False
            
            while i < len(lines) - 2:
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                
                try:
                    # First line should be number of questions
                    n_questions = int(line)
                    
                    # Second line should be comma-separated question IDs
                    q_ids_line = lines[i + 1].strip()
                    if ',' in q_ids_line:
                        q_ids = [int(x.strip()) for x in q_ids_line.split(',')]
                        if len(q_ids) == n_questions:  # Validation
                            question_ids.update(q_ids)
                            sequence_format_detected = True
                            i += 3  # Skip to next sequence
                            continue
                except (ValueError, IndexError):
                    pass
                
                i += 1
            
            if sequence_format_detected:
                return list(question_ids)
        
        # Fallback: try CSV parsing
        if filepath.endswith('.csv'):
            try:
                import pandas as pd
                # Read with error handling
                df = pd.read_csv(filepath, error_bad_lines=False, warn_bad_lines=False)
                
                # Look for question-related columns
                q_columns = []
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(term in col_lower for term in ['question', 'item', 'skill', 'problem', 'q_id']):
                        q_columns.append(col)
                
                for col in q_columns:
                    try:
                        q_vals = pd.to_numeric(df[col], errors='coerce').dropna().astype(int)
                        question_ids.update(q_vals)
                    except:
                        continue
                
                if question_ids:
                    return list(question_ids)
                    
            except Exception:
                pass
        
        # Last resort: try to parse as simple text
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num > 1000:  # Don't scan entire huge files
                    break
                    
                parts = line.strip().split()
                for part in parts[:10]:  # Check first 10 parts
                    try:
                        val = int(part)
                        if 0 <= val <= 50000:  # Reasonable question ID range
                            question_ids.add(val)
                    except ValueError:
                        continue
        
    except Exception as e:
        print(f"Warning: Error parsing {filepath}: {e}")
    
    return list(question_ids)


def _get_n_questions_from_data_fixed(dataset_name, data_dir):
    """Fixed comprehensive n_questions detection with proper validation."""
    dataset_path = os.path.join(data_dir, dataset_name)
    all_question_ids = set()
    
    print(f"[DEBUG] Comprehensive scan for {dataset_name}...")
    
    # Detect format first
    format_info = _detect_dataset_format(dataset_name, dataset_path)
    files_scanned = 0
    
    if format_info['type'] == 'presplit':
        # Scan all pre-split files
        for fold in range(5):
            for split in ['train', 'valid']:
                filename = f"{dataset_name}_{split}{fold}.csv"
                filepath = os.path.join(dataset_path, filename)
                if os.path.exists(filepath):
                    q_ids = _extract_question_ids_robust(filepath)
                    all_question_ids.update(q_ids)
                    files_scanned += 1
                    print(f"[DEBUG]   {filename}: {len(q_ids)} unique questions")
        
        # Scan test file
        test_file = f"{dataset_name}_test.csv"
        test_path = os.path.join(dataset_path, test_file)
        if os.path.exists(test_path):
            q_ids = _extract_question_ids_robust(test_path)
            all_question_ids.update(q_ids)
            files_scanned += 1
            print(f"[DEBUG]   {test_file}: {len(q_ids)} unique questions")
    
    elif format_info['type'] == 'single_file':
        # Scan single files
        for file_key in ['train_file', 'test_file']:
            if file_key in format_info:
                filepath = os.path.join(dataset_path, format_info[file_key])
                if os.path.exists(filepath):
                    q_ids = _extract_question_ids_robust(filepath)
                    all_question_ids.update(q_ids)
                    files_scanned += 1
                    print(f"[DEBUG]   {format_info[file_key]}: {len(q_ids)} unique questions")
    
    else:
        # Unknown format - scan all data files
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.csv', '.txt')) and any(x in file for x in ['train', 'test', 'valid']):
                    filepath = os.path.join(root, file)
                    q_ids = _extract_question_ids_robust(filepath)
                    if q_ids:  # Only count if we found questions
                        all_question_ids.update(q_ids)
                        files_scanned += 1
                        print(f"[DEBUG]   {file}: {len(q_ids)} unique questions")
    
    if all_question_ids:
        min_q = min(all_question_ids)
        max_q = max(all_question_ids)
        n_unique = len(all_question_ids)
        
        print(f"[DEBUG] Summary: {files_scanned} files, {n_unique} unique questions ({min_q}-{max_q})")
        
        # Handle 0-based vs 1-based indexing
        if min_q == 0:
            # 0-based indexing: n_questions = max_id + 1, but use unique count if it's consistent
            if n_unique == max_q + 1:
                detected_n = max_q + 1
            else:
                detected_n = n_unique
            print(f"[DEBUG] Detected 0-based indexing, n_questions = {detected_n}")
        else:
            # 1-based indexing: use unique count if available, otherwise max_id
            if n_unique == max_q:
                detected_n = max_q
            else:
                detected_n = n_unique
            print(f"[DEBUG] Detected 1-based indexing, n_questions = {detected_n}")
        
        # Validation: check if detection makes sense
        if detected_n != n_unique and abs(detected_n - n_unique) > 1:
            print(f"[WARNING] Detection may be incorrect: detected={detected_n}, unique_count={n_unique}")
            # Use the more conservative estimate
            detected_n = n_unique
        
        return detected_n
    else:
        print(f"[WARNING] No questions found in {files_scanned} files scanned")
        return 0


# Add validation function
def validate_detection_result(dataset_name, detected_n, data_dir):
    """Validate the detection result by checking consistency."""
    dataset_path = os.path.join(data_dir, dataset_name)
    
    # Check Q-matrix consistency if available
    qmatrix_files = [
        f"{dataset_name}_qid_sid",
        f"{dataset_name}_Qmatrix.csv", 
        "Qmatrix.csv",
        f"conceptname_question_id_{dataset_name}.csv",
        "conceptname_question_id.csv"
    ]
    
    for qfile in qmatrix_files:
        qpath = os.path.join(dataset_path, qfile)
        if os.path.exists(qpath):
            try:
                if qfile.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(qpath)
                    if 'question_id' in df.columns:
                        qmatrix_max = df['question_id'].max()
                    else:
                        qmatrix_max = df.iloc[:, -1].max()
                else:
                    qmatrix_max = 0
                    with open(qpath, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                try:
                                    q_id = int(parts[0])
                                    qmatrix_max = max(qmatrix_max, q_id)
                                except ValueError:
                                    continue
                
                print(f"[VALIDATION] Q-matrix {qfile}: max_question_id = {qmatrix_max}")
                
                # Check consistency
                if abs(qmatrix_max - detected_n) > 1:
                    print(f"[WARNING] Q-matrix inconsistency: qmatrix_max={qmatrix_max}, detected={detected_n}")
                    # For significant mismatches, trust the data over Q-matrix
                    if qmatrix_max > detected_n * 2:
                        print(f"[WARNING] Q-matrix seems to contain phantom questions, ignoring")
                    else:
                        print(f"[INFO] Using Q-matrix value as it's more reliable")
                        return qmatrix_max
                
            except Exception as e:
                print(f"[WARNING] Could not validate with Q-matrix {qfile}: {e}")
    
    return detected_n



def _get_n_questions_from_qmatrix(qmatrix_path):
    """Extract n_questions from Q-matrix file."""
    try:
        if qmatrix_path.endswith('.csv'):
            # Handle CSV Q-matrix
            df = pd.read_csv(qmatrix_path)
            if 'question_id' in df.columns:
                return df['question_id'].max()
            elif len(df.columns) >= 2:
                # Assume first column is question_id
                return df.iloc[:, 0].max()
            else:
                return 0
        else:
            # Handle text-based Q-matrix (e.g., qid_sid format)
            max_q_id = 0
            with open(qmatrix_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            q_id = int(parts[0])
                            max_q_id = max(max_q_id, q_id)
                        except ValueError:
                            continue
            return max_q_id
    except Exception:
        return 0


def _get_n_questions_from_data_fixed(dataset_name, data_dir):
    """Comprehensive n_questions detection from all data files."""
    dataset_path = os.path.join(data_dir, dataset_name)
    all_question_ids = []
    
    # Detect format first
    format_info = _detect_dataset_format(dataset_name, dataset_path)
    
    if format_info['type'] == 'presplit':
        # Analyze pre-split files comprehensively
        files_to_check = []
        
        # Add all train/valid files for all folds
        for fold in range(5):
            train_file = f"{dataset_name}_train{fold}.csv"
            valid_file = f"{dataset_name}_valid{fold}.csv"
            files_to_check.extend([train_file, valid_file])
        
        # Add test file
        test_file = f"{dataset_name}_test.csv"
        files_to_check.append(test_file)
        
        for filename in files_to_check:
            filepath = os.path.join(dataset_path, filename)
            if os.path.exists(filepath):
                try:
                    q_ids = _extract_question_ids_presplit(filepath)
                    all_question_ids.extend(q_ids)
                except Exception as e:
                    print(f"Warning: Could not parse {filename}: {e}")
    
    elif format_info['type'] == 'single_file':
        # Analyze single files
        train_path = os.path.join(dataset_path, format_info['train_file'])
        test_path = os.path.join(dataset_path, format_info['test_file'])
        
        for filepath in [train_path, test_path]:
            if os.path.exists(filepath):
                try:
                    if filepath.endswith('.csv'):
                        if "builder" in filepath:
                            q_ids = _extract_question_ids_builder_csv(filepath)
                        else:
                            q_ids = _extract_question_ids_csv(filepath)
                    else:
                        q_ids = _extract_question_ids_txt(filepath)
                    all_question_ids.extend(q_ids)
                except Exception as e:
                    print(f"Warning: Could not parse {filepath}: {e}")
    
    # Return max question ID found
    if all_question_ids:
        return max(all_question_ids)
    else:
        return 0


def _detect_dataset_format(dataset_name, dataset_path):
    """Detect dataset format and return file information."""
    # Check for pre-split format (Yeung-style)
    presplit_patterns = [
        f"{dataset_name}_train0.csv", f"{dataset_name}_valid0.csv", f"{dataset_name}_test.csv"
    ]
    
    has_presplit = all(os.path.exists(os.path.join(dataset_path, f)) for f in presplit_patterns)
    
    if has_presplit:
        return {'type': 'presplit'}
    
    # Check for single file format (orig-style)
    single_file_patterns = [
        (f"{dataset_name}_train.txt", f"{dataset_name}_test.txt"),
        (f"{dataset_name}_train.csv", f"{dataset_name}_test.csv"),
        ("builder_train.csv", "builder_test.csv"),  # assist2009 special case
        (f"{dataset_name.replace('2011', '')}_train.txt", f"{dataset_name.replace('2011', '')}_test.txt"),  # statics2011
        (f"{dataset_name.replace('statics', 'static')}_train.txt", f"{dataset_name.replace('statics', 'static')}_test.txt")
    ]
    
    for train_pattern, test_pattern in single_file_patterns:
        train_path = os.path.join(dataset_path, train_pattern)
        test_path = os.path.join(dataset_path, test_pattern)
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            return {
                'type': 'single_file',
                'train_file': train_pattern,
                'test_file': test_pattern
            }
    
    return {'type': 'unknown'}


def _extract_question_ids_presplit(filepath):
    """Extract question IDs from pre-split CSV format."""
    q_ids = []
    
    with open(filepath, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            # Skip sequence length line
            if line_idx % 3 == 0:
                continue
            # Handle question line
            elif line_idx % 3 == 1:
                try:
                    q_seq = [int(x) for x in line.split(',') if x.strip()]
                    q_ids.extend(q_seq)
                except ValueError:
                    continue
    
    return q_ids


def _extract_question_ids_csv(filepath):
    """Extract question IDs from standard CSV format."""
    q_ids = []
    
    try:
        df = pd.read_csv(filepath)
        
        # Try different column names for question IDs
        question_columns = ['question_id', 'item_id', 'problem_id', 'skill_id']
        
        for col in question_columns:
            if col in df.columns:
                q_ids = df[col].dropna().astype(int).tolist()
                break
        
        # If no standard column found, try first column
        if not q_ids and len(df.columns) > 0:
            try:
                q_ids = df.iloc[:, 0].dropna().astype(int).tolist()
            except:
                pass
    
    except Exception as e:
        print(f"Error reading CSV {filepath}: {e}")
    
    return q_ids


def _extract_question_ids_builder_csv(filepath):
    """Extract question IDs from assist2009 builder CSV format."""
    q_ids = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Skip sequence length line
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
                    q_line = q_line[:-1]
                q_seq = [int(x) + 1 for x in q_line.split(',') if x.strip()]  # +1 for 1-based indexing
                q_ids.extend(q_seq)
                i += 1
            except:
                i += 1
                continue
            
            # Skip answer line
            i += 1
    
    except Exception as e:
        print(f"Error reading builder CSV {filepath}: {e}")
    
    return q_ids


def _extract_question_ids_txt(filepath):
    """Extract question IDs from TXT format."""
    q_ids = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Skip sequence length line
            try:
                seq_len = int(lines[i].strip())
                i += 1
            except:
                i += 1
                continue
            
            # Parse question sequence
            try:
                q_seq = [int(x) + 1 for x in lines[i].strip().split(',') if x.strip()]  # +1 for 1-based indexing
                q_ids.extend(q_seq)
                i += 1
            except:
                i += 1
                continue
            
            # Skip answer line
            i += 1
    
    except Exception as e:
        print(f"Error reading TXT {filepath}: {e}")
    
    return q_ids


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