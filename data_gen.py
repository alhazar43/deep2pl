#!/usr/bin/env python3
"""
GPCM Synthetic Data Generator

Usage:
    python data_gen.py --format PC --categories 4
    python data_gen.py --format OC --categories 3
    python data_gen.py --format both --categories 5
"""

import argparse
import numpy as np
from pathlib import Path
import json


class GpcmGen:
    def __init__(self, n_students=800, n_questions=50, n_cats=3, seq_len_range=(10, 50)):
        self.n_students = n_students
        self.n_questions = n_questions  
        self.n_cats = n_cats
        self.seq_len_range = seq_len_range
        
        self.theta = np.random.normal(0, 1, n_students)  # abilities
        self.alpha = np.random.lognormal(0, 0.3, n_questions)  # discrimination
        
        # Generate ordered thresholds for each question
        self.beta = np.zeros((n_questions, n_cats - 1))
        for q in range(n_questions):
            base_diff = np.random.normal(0, 1)
            thresh = np.sort(np.random.normal(base_diff, 0.5, n_cats - 1))
            self.beta[q] = thresh
    
    def gpcm_prob(self, theta, alpha, betas):
        K = len(betas) + 1
        cum_logits = np.zeros(K)
        cum_logits[0] = 0
        
        for k in range(1, K):
            cum_logits[k] = np.sum([alpha * (theta - betas[h]) for h in range(k)])
        
        exp_logits = np.exp(cum_logits - np.max(cum_logits))
        return exp_logits / np.sum(exp_logits)
    
    def gen_response(self, student_id, question_id):
        theta = self.theta[student_id]
        alpha = self.alpha[question_id]
        betas = self.beta[question_id]
        
        probs = self.gpcm_prob(theta, alpha, betas)
        return np.random.choice(self.n_cats, p=probs)
    
    def gen_seq_data(self):
        sequences = []
        
        for student_id in range(self.n_students):
            seq_len = np.random.randint(*self.seq_len_range)
            q_seq = np.random.choice(self.n_questions, size=seq_len, replace=True)
            
            r_seq = []
            for q_id in q_seq:
                response = self.gen_response(student_id, q_id)
                r_seq.append(response)
            
            sequences.append({
                'student_id': student_id,
                'seq_len': seq_len,
                'questions': q_seq,
                'responses': r_seq
            })
        
        return sequences
    
    def save_oc_format(self, sequences, data_dir, split_ratio=0.8):
        # Save ordered categorical format
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        n_train = int(len(sequences) * split_ratio)
        train_seqs = sequences[:n_train]
        test_seqs = sequences[n_train:]
        
        for split_name, split_seqs in [('train', train_seqs), ('test', test_seqs)]:
            filename = data_dir / f"synthetic_oc_{split_name}.txt"
            
            with open(filename, 'w') as f:
                for seq in split_seqs:
                    f.write(f"{seq['seq_len']}\n")
                    
                    q_str = ','.join(map(str, seq['questions']))
                    f.write(f"{q_str}\n")
                    
                    r_str = ','.join(map(str, seq['responses']))
                    f.write(f"{r_str}\n")
        
        metadata = {
            'n_students': self.n_students,
            'n_questions': self.n_questions,
            'n_cats': self.n_cats,
            'response_type': 'ordered_categorical',
            'format': 'OC',
            'description': f'Synthetic {self.n_cats} ordered categories',
            'train_students': n_train,
            'test_students': len(sequences) - n_train,
            'seq_len_range': self.seq_len_range
        }
        
        with open(data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved Ordered Categories format to {data_dir}")
        print(f"   {self.n_cats} categories: {list(range(self.n_cats))}")
        print(f"   Train: {n_train} students, Test: {len(sequences) - n_train} students")
    
    def save_pc_format(self, sequences, data_dir, split_ratio=0.8):
        # Save partial credit format  
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        n_train = int(len(sequences) * split_ratio)
        train_seqs = sequences[:n_train]
        test_seqs = sequences[n_train:]
        
        def cat_to_pc(response, n_cats):
            return response / (n_cats - 1)
        
        for split_name, split_seqs in [('train', train_seqs), ('test', test_seqs)]:
            filename = data_dir / f"synthetic_pc_{split_name}.txt"
            
            with open(filename, 'w') as f:
                for seq in split_seqs:
                    f.write(f"{seq['seq_len']}\n")
                    
                    q_str = ','.join(map(str, seq['questions']))
                    f.write(f"{q_str}\n")
                    
                    pc_scores = [cat_to_pc(r, self.n_cats) for r in seq['responses']]
                    r_str = ','.join(f"{score:.3f}" for score in pc_scores)
                    f.write(f"{r_str}\n")
        
        metadata = {
            'n_students': self.n_students,
            'n_questions': self.n_questions,
            'n_cats': self.n_cats,
            'response_type': 'partial_credit',
            'format': 'PC',
            'description': f'Synthetic partial credit scores in [0,1]',
            'score_range': [0.0, 1.0],
            'discretization': f'{self.n_cats} levels: {[round(k/(self.n_cats-1), 3) for k in range(self.n_cats)]}',
            'train_students': n_train,
            'test_students': len(sequences) - n_train,
            'seq_len_range': self.seq_len_range
        }
        
        with open(data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved Partial Credit format to {data_dir}")
        print(f"   {self.n_cats} levels: {[round(k/(self.n_cats-1), 3) for k in range(self.n_cats)]}")
        print(f"   Train: {n_train} students, Test: {len(sequences) - n_train} students")
    
    def save_irt_params(self, data_dir):
        data_dir = Path(data_dir)
        
        irt_params = {
            'student_abilities': {
                'theta': self.theta.tolist(),
                'mean': float(np.mean(self.theta)),
                'std': float(np.std(self.theta))
            },
            'question_params': {
                'discrimination': {
                    'alpha': self.alpha.tolist(),
                    'mean': float(np.mean(self.alpha)),
                    'std': float(np.std(self.alpha))
                },
                'difficulties': {
                    'beta': self.beta.tolist(),
                    'shape': list(self.beta.shape),
                    'description': f'{self.n_cats-1} thresholds per question'
                }
            },
            'model_info': {
                'model_type': 'GPCM',
                'n_cats': self.n_cats,
                'param_gen': 'synthetic'
            }
        }
        
        with open(data_dir / 'true_irt_parameters.json', 'w') as f:
            json.dump(irt_params, f, indent=2)
        
        print(f"   Saved true IRT parameters to {data_dir}/true_irt_parameters.json")
    
    def gen_and_save(self, base_data_dir="./data", formats=["OC", "PC"]):
        print(f"Generating GPCM synthetic data...")
        print(f"   Students: {self.n_students}")
        print(f"   Questions: {self.n_questions}")  
        print(f"   Categories: {self.n_cats}")
        print(f"   Sequence length: {self.seq_len_range[0]}-{self.seq_len_range[1]}")
        
        sequences = self.gen_seq_data()
        
        if "OC" in formats:
            oc_dir = Path(base_data_dir) / "synthetic_OC"
            self.save_oc_format(sequences, oc_dir)
            self.save_irt_params(oc_dir)
            
        if "PC" in formats:
            pc_dir = Path(base_data_dir) / "synthetic_PC"
            self.save_pc_format(sequences, pc_dir)
            self.save_irt_params(pc_dir)
        
        print(f"Generation complete!")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic GPCM data')
    parser.add_argument('--format', choices=['PC', 'OC', 'both'], default='both',
                       help='Format to generate: PC (Partial Credit), OC (Ordered Categories), or both')
    parser.add_argument('--categories', type=int, default=4,
                       help='Number of response categories (K)')
    parser.add_argument('--students', type=int, default=800,
                       help='Number of students')
    parser.add_argument('--questions', type=int, default=50,
                       help='Number of questions')
    parser.add_argument('--min_seq', type=int, default=10,
                       help='Minimum sequence length')
    parser.add_argument('--max_seq', type=int, default=50,
                       help='Maximum sequence length')
    parser.add_argument('--output_dir', default='./data',
                       help='Output directory for generated data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    if args.format == 'both':
        formats = ['PC', 'OC']
    else:
        formats = [args.format]
    
    gen = GpcmGen(
        n_students=args.students,
        n_questions=args.questions,
        n_cats=args.categories,
        seq_len_range=(args.min_seq, args.max_seq)
    )
    
    gen.gen_and_save(base_data_dir=args.output_dir, formats=formats)


if __name__ == "__main__":
    main()