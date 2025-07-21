#!/usr/bin/env python3
"""
Synthetic Data Generator for GPCM (Generalized Partial Credit Model)

Generates two formats:
1. synthetic_PC: Partial Credit with decimal scores in [0,1] 
2. synthetic_OC: Ordered Categories with explicit K-1 categories

Usage:
    python generate_synthetic_gpcm.py --format PC --categories 4
    python generate_synthetic_gpcm.py --format OC --categories 3
    python generate_synthetic_gpcm.py --format both --categories 5
"""

import argparse
import numpy as np
import pandas as pd
import os
from pathlib import Path
import json


class GpcmSyntheticGenerator:
    def __init__(self, n_students=800, n_questions=50, n_categories=3, sequence_length_range=(10, 50)):
        """
        GPCM Synthetic Data Generator
        
        Args:
            n_students: Number of students
            n_questions: Number of questions 
            n_categories: Number of response categories (K)
            sequence_length_range: (min, max) sequence length per student
        """
        self.n_students = n_students
        self.n_questions = n_questions  
        self.n_categories = n_categories
        self.sequence_length_range = sequence_length_range
        
        # Generate synthetic IRT parameters
        self.student_abilities = np.random.normal(0, 1, n_students)  # theta ~ N(0, 1)
        self.question_discrimination = np.random.lognormal(0, 0.3, n_questions)  # alpha > 0
        
        # Generate difficulty thresholds for each category (K-1 thresholds per question)
        # Ensure thresholds are ordered: beta_1 < beta_2 < ... < beta_{K-1}
        self.question_difficulties = np.zeros((n_questions, n_categories - 1))
        for q in range(n_questions):
            # Generate ordered thresholds
            base_difficulty = np.random.normal(0, 1)  # Overall question difficulty
            thresholds = np.sort(np.random.normal(base_difficulty, 0.5, n_categories - 1))
            self.question_difficulties[q] = thresholds
    
    def gpcm_probability(self, theta, alpha, betas):
        """
        Calculate GPCM response probabilities for a student-question pair
        
        Args:
            theta: Student ability
            alpha: Question discrimination
            betas: Question difficulty thresholds (K-1 values)
            
        Returns:
            probabilities: Array of length K with P(response = k) for k = 0, ..., K-1
        """
        K = len(betas) + 1
        
        # Calculate cumulative logits: sum from h=0 to k-1 of alpha*(theta - beta_h)
        cumulative_logits = np.zeros(K)
        cumulative_logits[0] = 0  # First category has no threshold
        
        for k in range(1, K):
            cumulative_logits[k] = np.sum([alpha * (theta - betas[h]) for h in range(k)])
        
        # Convert to probabilities using softmax
        exp_logits = np.exp(cumulative_logits - np.max(cumulative_logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        return probabilities
    
    def generate_response(self, student_id, question_id):
        """Generate a single response using GPCM"""
        theta = self.student_abilities[student_id]
        alpha = self.question_discrimination[question_id]
        betas = self.question_difficulties[question_id]
        
        probs = self.gpcm_probability(theta, alpha, betas)
        response = np.random.choice(self.n_categories, p=probs)
        
        return response
    
    def generate_sequence_data(self):
        """Generate sequence data for all students"""
        sequences = []
        
        for student_id in range(self.n_students):
            # Random sequence length for this student
            seq_len = np.random.randint(*self.sequence_length_range)
            
            # Random question sequence (with possible repeats)
            question_sequence = np.random.choice(self.n_questions, size=seq_len, replace=True)
            
            # Generate responses
            response_sequence = []
            for q_id in question_sequence:
                response = self.generate_response(student_id, q_id)
                response_sequence.append(response)
            
            sequences.append({
                'student_id': student_id,
                'sequence_length': seq_len,
                'questions': question_sequence,
                'responses': response_sequence
            })
        
        return sequences
    
    def save_ordered_categories_format(self, sequences, data_dir, split_ratio=0.8):
        """
        Save data in Ordered Categories format (synthetic_OC)
        Format: Explicit K-category responses {0, 1, 2, ..., K-1}
        """
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into train/test
        n_train = int(len(sequences) * split_ratio)
        train_sequences = sequences[:n_train]
        test_sequences = sequences[n_train:]
        
        # Save in DKVMN format (sequence length, questions, responses)
        for split_name, split_sequences in [('train', train_sequences), ('test', test_sequences)]:
            filename = data_dir / f"synthetic_oc_{split_name}.txt"
            
            with open(filename, 'w') as f:
                for seq in split_sequences:
                    # Write sequence length
                    f.write(f"{seq['sequence_length']}\n")
                    
                    # Write questions (0-indexed)
                    questions_str = ','.join(map(str, seq['questions']))
                    f.write(f"{questions_str}\n")
                    
                    # Write responses (K-category: 0, 1, ..., K-1)
                    responses_str = ','.join(map(str, seq['responses']))
                    f.write(f"{responses_str}\n")
        
        # Save metadata
        metadata = {
            'n_students': self.n_students,
            'n_questions': self.n_questions,
            'n_categories': self.n_categories,
            'response_type': 'ordered_categorical',
            'format': 'OC',
            'description': f'Synthetic GPCM data with {self.n_categories} ordered categories',
            'train_students': n_train,
            'test_students': len(sequences) - n_train,
            'sequence_length_range': self.sequence_length_range
        }
        
        with open(data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✅ Saved Ordered Categories format to {data_dir}")
        print(f"   📊 {self.n_categories} categories: {list(range(self.n_categories))}")
        print(f"   📁 Train: {n_train} students, Test: {len(sequences) - n_train} students")
    
    def save_partial_credit_format(self, sequences, data_dir, split_ratio=0.8):
        """
        Save data in Partial Credit format (synthetic_PC)
        Format: Decimal scores in [0,1] representing partial credit
        """
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into train/test
        n_train = int(len(sequences) * split_ratio)
        train_sequences = sequences[:n_train]
        test_sequences = sequences[n_train:]
        
        # Convert K-category responses to [0,1] decimal scores
        def category_to_partial_credit(response, n_categories):
            """Convert category k to partial credit score k/(K-1)"""
            return response / (n_categories - 1)
        
        # Save in DKVMN format with partial credit scores
        for split_name, split_sequences in [('train', train_sequences), ('test', test_sequences)]:
            filename = data_dir / f"synthetic_pc_{split_name}.txt"
            
            with open(filename, 'w') as f:
                for seq in split_sequences:
                    # Write sequence length
                    f.write(f"{seq['sequence_length']}\n")
                    
                    # Write questions (0-indexed)
                    questions_str = ','.join(map(str, seq['questions']))
                    f.write(f"{questions_str}\n")
                    
                    # Write responses as partial credit scores [0, 1]
                    pc_scores = [category_to_partial_credit(r, self.n_categories) for r in seq['responses']]
                    responses_str = ','.join(f"{score:.3f}" for score in pc_scores)
                    f.write(f"{responses_str}\n")
        
        # Save metadata
        metadata = {
            'n_students': self.n_students,
            'n_questions': self.n_questions,
            'n_categories': self.n_categories,
            'response_type': 'partial_credit',
            'format': 'PC',
            'description': f'Synthetic GPCM data with partial credit scores in [0,1]',
            'score_range': [0.0, 1.0],
            'discretization': f'{self.n_categories} levels: {[round(k/(self.n_categories-1), 3) for k in range(self.n_categories)]}',
            'train_students': n_train,
            'test_students': len(sequences) - n_train,
            'sequence_length_range': self.sequence_length_range
        }
        
        with open(data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved Partial Credit format to {data_dir}")
        print(f"   📊 {self.n_categories} levels: {[round(k/(self.n_categories-1), 3) for k in range(self.n_categories)]}")
        print(f"   📁 Train: {n_train} students, Test: {len(sequences) - n_train} students")
    
    def save_irt_parameters(self, data_dir):
        """Save the true IRT parameters used for generation"""
        data_dir = Path(data_dir)
        
        irt_params = {
            'student_abilities': {
                'theta': self.student_abilities.tolist(),
                'mean': float(np.mean(self.student_abilities)),
                'std': float(np.std(self.student_abilities))
            },
            'question_parameters': {
                'discrimination': {
                    'alpha': self.question_discrimination.tolist(),
                    'mean': float(np.mean(self.question_discrimination)),
                    'std': float(np.std(self.question_discrimination))
                },
                'difficulties': {
                    'beta': self.question_difficulties.tolist(),
                    'shape': list(self.question_difficulties.shape),
                    'description': f'{self.n_categories-1} thresholds per question'
                }
            },
            'model_info': {
                'model_type': 'GPCM',
                'n_categories': self.n_categories,
                'parameter_generation': 'synthetic'
            }
        }
        
        with open(data_dir / 'true_irt_parameters.json', 'w') as f:
            json.dump(irt_params, f, indent=2)
        
        print(f"   💾 Saved true IRT parameters to {data_dir}/true_irt_parameters.json")
    
    def generate_and_save(self, base_data_dir="./data", formats=["OC", "PC"]):
        """Generate and save data in specified formats"""
        print(f"🔄 Generating GPCM synthetic data...")
        print(f"   👥 Students: {self.n_students}")
        print(f"   ❓ Questions: {self.n_questions}")  
        print(f"   📊 Categories: {self.n_categories}")
        print(f"   📏 Sequence length: {self.sequence_length_range[0]}-{self.sequence_length_range[1]}")
        
        # Generate sequences
        sequences = self.generate_sequence_data()
        
        # Save in requested formats
        if "OC" in formats:
            oc_dir = Path(base_data_dir) / "synthetic_OC"
            self.save_ordered_categories_format(sequences, oc_dir)
            self.save_irt_parameters(oc_dir)
            
        if "PC" in formats:
            pc_dir = Path(base_data_dir) / "synthetic_PC"
            self.save_partial_credit_format(sequences, pc_dir)
            self.save_irt_parameters(pc_dir)
        
        print(f"✨ Generation complete!")


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
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Determine formats to generate
    if args.format == 'both':
        formats = ['PC', 'OC']
    else:
        formats = [args.format]
    
    # Create generator
    generator = GpcmSyntheticGenerator(
        n_students=args.students,
        n_questions=args.questions,
        n_categories=args.categories,
        sequence_length_range=(args.min_seq, args.max_seq)
    )
    
    # Generate and save data
    generator.generate_and_save(base_data_dir=args.output_dir, formats=formats)


if __name__ == "__main__":
    main()