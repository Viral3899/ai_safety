#!/usr/bin/env python3
"""
Prepare comprehensive training data for AI safety models.
Combines real data from Kaggle with synthetic data generation.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

def load_real_data() -> Dict[str, Any]:
    """Load real data from Kaggle downloads."""
    real_data = {
        'abuse_detector': [],
        'crisis_detector': [],
        'content_filter': []
    }
    
    # Load mental health survey data for crisis detection
    survey_file = "kaggle_data/survey.csv"
    if os.path.exists(survey_file):
        print("📊 Loading mental health survey data...")
        df = pd.read_csv(survey_file)
        
        # Process survey data for crisis detection
        for _, row in df.iterrows():
            # Create text from survey responses
            text_parts = []
            if pd.notna(row.get('comments')):
                text_parts.append(str(row['comments']))
            
            # Add context from other fields
            context = []
            if pd.notna(row.get('treatment')) and row['treatment'] == 'Yes':
                context.append("seeking treatment")
            if pd.notna(row.get('work_interfere')) and row['work_interfere'] != 'Never':
                context.append("work interference")
            if pd.notna(row.get('family_history')) and row['family_history'] == 'Yes':
                context.append("family history")
            
            if text_parts or context:
                text = " ".join(text_parts + context)
                
                # Determine crisis level based on survey responses
                crisis_indicators = 0
                if row.get('treatment') == 'Yes':
                    crisis_indicators += 2
                if row.get('work_interfere') in ['Often', 'Sometimes']:
                    crisis_indicators += 1
                if row.get('family_history') == 'Yes':
                    crisis_indicators += 1
                if row.get('mental_health_consequence') == 'Yes':
                    crisis_indicators += 1
                
                if crisis_indicators >= 3:
                    label = 2  # crisis
                    category = "crisis"
                elif crisis_indicators >= 1:
                    label = 1  # help_seeking
                    category = "help_seeking"
                else:
                    label = 0  # safe
                    category = "safe"
                
                real_data['crisis_detector'].append({
                    'text': text,
                    'label': label,
                    'category': category,
                    'severity': 'high' if label == 2 else 'medium' if label == 1 else 'low'
                })
        
        print(f"✅ Loaded {len(real_data['crisis_detector'])} crisis detection samples from survey")
    
    return real_data

def load_synthetic_data() -> Dict[str, Any]:
    """Load synthetic data."""
    synthetic_data = {
        'abuse_detector': [],
        'crisis_detector': [],
        'content_filter': []
    }
    
    # Load combined synthetic data
    combined_file = "data/synthetic/combined_data.json"
    if os.path.exists(combined_file):
        print("📊 Loading synthetic data...")
        with open(combined_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for model_type in synthetic_data.keys():
            if model_type in data:
                synthetic_data[model_type] = data[model_type]
                print(f"✅ Loaded {len(synthetic_data[model_type])} {model_type} samples")
    
    return synthetic_data

def combine_datasets(real_data: Dict[str, Any], synthetic_data: Dict[str, Any]) -> Dict[str, Any]:
    """Combine real and synthetic data."""
    combined = {}
    
    for model_type in ['abuse_detector', 'crisis_detector', 'content_filter']:
        combined[model_type] = []
        
        # Add real data first (if available)
        if real_data[model_type]:
            combined[model_type].extend(real_data[model_type])
            print(f"📈 Added {len(real_data[model_type])} real samples for {model_type}")
        
        # Add synthetic data
        if synthetic_data[model_type]:
            combined[model_type].extend(synthetic_data[model_type])
            print(f"📈 Added {len(synthetic_data[model_type])} synthetic samples for {model_type}")
        
        print(f"📊 Total {model_type} samples: {len(combined[model_type])}")
    
    return combined

def create_train_test_split(data: Dict[str, Any], test_size: float = 0.2) -> Dict[str, Any]:
    """Create train/test splits for each model."""
    splits = {}
    
    for model_type, samples in data.items():
        if not samples:
            print(f"⚠️  No data available for {model_type}")
            continue
        
        # Convert to DataFrame for easier splitting
        df = pd.DataFrame(samples)
        
        # Stratified split to maintain label distribution
        if 'label' in df.columns:
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                stratify=df['label'],
                random_state=42
            )
        else:
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=42
            )
        
        splits[model_type] = {
            'train': train_df.to_dict('records'),
            'test': test_df.to_dict('records')
        }
        
        print(f"📊 {model_type}: {len(train_df)} train, {len(test_df)} test samples")
    
    return splits

def save_processed_data(splits: Dict[str, Any], output_dir: str = "data/processed"):
    """Save processed data splits."""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_type, split_data in splits.items():
        model_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save train data
        train_file = os.path.join(model_dir, "train.json")
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(split_data['train'], f, indent=2, ensure_ascii=False)
        
        # Save test data
        test_file = os.path.join(model_dir, "test.json")
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(split_data['test'], f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {model_type} data to {model_dir}")

def generate_data_summary(splits: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics for the prepared data."""
    summary = {
        'total_models': len(splits),
        'models': {}
    }
    
    for model_type, split_data in splits.items():
        train_samples = len(split_data['train'])
        test_samples = len(split_data['test'])
        total_samples = train_samples + test_samples
        
        # Calculate label distribution
        train_df = pd.DataFrame(split_data['train'])
        label_dist = {}
        if 'label' in train_df.columns:
            label_dist = train_df['label'].value_counts().to_dict()
        
        summary['models'][model_type] = {
            'total_samples': total_samples,
            'train_samples': train_samples,
            'test_samples': test_samples,
            'label_distribution': label_dist
        }
    
    return summary

def main():
    """Main data preparation pipeline."""
    print("🚀 AI Safety Models - Data Preparation")
    print("=" * 50)
    
    # Load data
    print("\n📥 Loading datasets...")
    real_data = load_real_data()
    synthetic_data = load_synthetic_data()
    
    # Combine datasets
    print("\n🔄 Combining datasets...")
    combined_data = combine_datasets(real_data, synthetic_data)
    
    # Create train/test splits
    print("\n✂️  Creating train/test splits...")
    splits = create_train_test_split(combined_data)
    
    # Save processed data
    print("\n💾 Saving processed data...")
    save_processed_data(splits)
    
    # Generate summary
    print("\n📊 Generating data summary...")
    summary = generate_data_summary(splits)
    
    # Save summary
    summary_file = "data/processed/data_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Data preparation completed!")
    print(f"📁 Processed data saved to: data/processed/")
    print(f"📋 Summary saved to: {summary_file}")
    
    # Print summary
    print(f"\n📊 Data Summary:")
    print(f"  Total models: {summary['total_models']}")
    for model_type, stats in summary['models'].items():
        print(f"  {model_type}:")
        print(f"    Total samples: {stats['total_samples']}")
        print(f"    Train: {stats['train_samples']}, Test: {stats['test_samples']}")
        if stats['label_distribution']:
            print(f"    Label distribution: {stats['label_distribution']}")

if __name__ == "__main__":
    main()
