#!/usr/bin/env python3
"""
Robust data pipeline for AI safety models.
Works with available data and generates additional synthetic data as needed.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import random

def generate_additional_synthetic_data() -> Dict[str, List[Dict]]:
    """Generate additional synthetic data to supplement existing datasets."""
    
    # Abuse detection patterns
    abuse_patterns = {
        'severe_abuse': [
            "You're a complete idiot and should die",
            "This is the worst thing I've ever seen, you're garbage",
            "I hate you so much, you're worthless",
            "You're such a loser, nobody likes you",
            "This is absolutely terrible, you're a failure"
        ],
        'mild_abuse': [
            "This is pretty bad, not impressed",
            "You're not very good at this",
            "This could be better, honestly",
            "I'm not a fan of this approach",
            "This doesn't work well for me"
        ],
        'safe': [
            "This is a great idea, I like it",
            "Thanks for sharing this information",
            "I appreciate your help with this",
            "This looks really good to me",
            "I think this is a solid approach"
        ]
    }
    
    # Crisis detection patterns
    crisis_patterns = {
        'crisis': [
            "I don't want to live anymore, everything is hopeless",
            "I'm thinking about ending it all, nothing matters",
            "I can't go on like this, I want to die",
            "Life isn't worth living, I'm done",
            "I'm planning to hurt myself, nobody cares"
        ],
        'help_seeking': [
            "I've been feeling really down lately, need help",
            "I'm struggling with depression and anxiety",
            "I need someone to talk to about my problems",
            "I'm having a hard time coping with stress",
            "I think I need professional help"
        ],
        'mild_concern': [
            "I've been feeling a bit stressed lately",
            "Work has been really overwhelming",
            "I'm having trouble sleeping",
            "I feel anxious about the future",
            "I'm not sure how to handle this situation"
        ],
        'safe': [
            "I'm doing well today, feeling good",
            "Life is going pretty smoothly right now",
            "I'm happy with how things are going",
            "I feel confident about the future",
            "Everything seems to be working out fine"
        ]
    }
    
    # Content filtering patterns
    content_patterns = {
        'adult': [
            "This contains explicit sexual content",
            "Adult themes and mature content ahead",
            "Not suitable for children or minors",
            "Contains graphic violence and language",
            "Mature audiences only content here"
        ],
        'moderate': [
            "This has some mild language and themes",
            "Contains some adult references",
            "May not be suitable for all audiences",
            "Has some mature content elements",
            "Mixed audience content with some warnings"
        ],
        'safe': [
            "This is family-friendly content",
            "Suitable for all ages and audiences",
            "Clean and appropriate content here",
            "Safe for children and adults alike",
            "Educational and wholesome material"
        ]
    }
    
    # Generate additional samples
    additional_data = {
        'abuse_detector': [],
        'crisis_detector': [],
        'content_filter': []
    }
    
    # Generate 1000 additional samples for each category
    for category, patterns in abuse_patterns.items():
        for _ in range(200):
            base_text = random.choice(patterns)
            # Add variations
            variations = [
                f"{base_text}",
                f"Honestly, {base_text.lower()}",
                f"Well, {base_text.lower()}",
                f"I think {base_text.lower()}",
                f"Personally, {base_text.lower()}"
            ]
            text = random.choice(variations)
            
            label = 1 if category != 'safe' else 0
            severity = 'high' if category == 'severe_abuse' else 'medium' if category == 'mild_abuse' else 'low'
            
            additional_data['abuse_detector'].append({
                'text': text,
                'label': label,
                'category': category,
                'severity': severity
            })
    
    for category, patterns in crisis_patterns.items():
        for _ in range(250):
            base_text = random.choice(patterns)
            variations = [
                f"{base_text}",
                f"Lately, {base_text.lower()}",
                f"Recently, {base_text.lower()}",
                f"I've been thinking, {base_text.lower()}",
                f"Sometimes {base_text.lower()}"
            ]
            text = random.choice(variations)
            
            if category == 'crisis':
                label = 2
            elif category == 'help_seeking':
                label = 1
            elif category == 'mild_concern':
                label = 1
            else:
                label = 0
            
            additional_data['crisis_detector'].append({
                'text': text,
                'label': label,
                'category': category,
                'severity': 'high' if label == 2 else 'medium' if label == 1 else 'low'
            })
    
    for category, patterns in content_patterns.items():
        for _ in range(200):
            base_text = random.choice(patterns)
            variations = [
                f"{base_text}",
                f"Warning: {base_text.lower()}",
                f"Note: {base_text.lower()}",
                f"Please be aware: {base_text.lower()}",
                f"Content notice: {base_text.lower()}"
            ]
            text = random.choice(variations)
            
            label = 1 if category != 'safe' else 0
            
            additional_data['content_filter'].append({
                'text': text,
                'label': label,
                'category': category,
                'severity': 'high' if category == 'adult' else 'medium' if category == 'moderate' else 'low'
            })
    
    return additional_data

def load_existing_data() -> Dict[str, List[Dict]]:
    """Load existing processed data."""
    data = {
        'abuse_detector': [],
        'crisis_detector': [],
        'content_filter': []
    }
    
    for model_type in data.keys():
        train_file = f"data/processed/{model_type}/train.json"
        if os.path.exists(train_file):
            with open(train_file, 'r', encoding='utf-8') as f:
                data[model_type] = json.load(f)
            print(f"✅ Loaded {len(data[model_type])} existing {model_type} samples")
    
    return data

def create_enhanced_dataset() -> Dict[str, Any]:
    """Create enhanced dataset with additional synthetic data."""
    print("🔄 Creating enhanced dataset...")
    
    # Load existing data
    existing_data = load_existing_data()
    
    # Generate additional synthetic data
    additional_data = generate_additional_synthetic_data()
    
    # Combine datasets
    enhanced_data = {}
    for model_type in ['abuse_detector', 'crisis_detector', 'content_filter']:
        enhanced_data[model_type] = existing_data[model_type] + additional_data[model_type]
        print(f"📊 {model_type}: {len(existing_data[model_type])} existing + {len(additional_data[model_type])} additional = {len(enhanced_data[model_type])} total")
    
    return enhanced_data

def create_final_splits(data: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Create final train/test/validation splits."""
    splits = {}
    
    for model_type, samples in data.items():
        if not samples:
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(samples)
        
        # Create train/test split (80/20)
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df['label'] if 'label' in df.columns else None,
            random_state=42
        )
        
        # Create train/validation split (80/20 of training data)
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.2,
            stratify=train_df['label'] if 'label' in train_df.columns else None,
            random_state=42
        )
        
        splits[model_type] = {
            'train': train_df.to_dict('records'),
            'validation': val_df.to_dict('records'),
            'test': test_df.to_dict('records')
        }
        
        print(f"📊 {model_type}: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
    
    return splits

def save_enhanced_data(splits: Dict[str, Any], output_dir: str = "data/enhanced"):
    """Save enhanced dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_type, split_data in splits.items():
        model_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        for split_name, data in split_data.items():
            file_path = os.path.join(model_dir, f"{split_name}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {model_type} enhanced data to {model_dir}")

def main():
    """Main enhanced data pipeline."""
    print("🚀 AI Safety Models - Enhanced Data Pipeline")
    print("=" * 50)
    
    # Create enhanced dataset
    enhanced_data = create_enhanced_dataset()
    
    # Create final splits
    print("\n✂️  Creating final data splits...")
    splits = create_final_splits(enhanced_data)
    
    # Save enhanced data
    print("\n💾 Saving enhanced data...")
    save_enhanced_data(splits)
    
    # Generate final summary
    print("\n📊 Final Dataset Summary:")
    total_samples = 0
    for model_type, split_data in splits.items():
        model_total = sum(len(data) for data in split_data.values())
        total_samples += model_total
        print(f"  {model_type}: {model_total} total samples")
        for split_name, data in split_data.items():
            print(f"    {split_name}: {len(data)} samples")
    
    print(f"\n✅ Enhanced data pipeline completed!")
    print(f"📁 Enhanced data saved to: data/enhanced/")
    print(f"📊 Total samples across all models: {total_samples}")

if __name__ == "__main__":
    main()
