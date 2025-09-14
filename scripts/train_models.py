#!/usr/bin/env python3
"""
Train AI safety models with synthetic data.
"""

import sys
import os
import argparse
import json
from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from safety_system.safety_manager import SafetyManager
from core.base_model import ModelConfig
from utils.data_generator import DataGenerator


def generate_comprehensive_synthetic_data() -> Dict[str, List[Dict[str, Any]]]:
    """Generate comprehensive synthetic training data for all models."""
    print("Generating comprehensive synthetic training data...")
    
    # Initialize data generator
    data_gen = DataGenerator()
    
    # Generate datasets for each model
    datasets = {}
    
    # 1. Abuse Detection Dataset (5000 samples)
    print("  📊 Generating abuse detection dataset (5000 samples)...")
    abuse_data = generate_enhanced_abuse_dataset(5000)
    datasets['abuse_detector'] = abuse_data
    
    # 2. Crisis Detection Dataset (3000 samples)
    print("  📊 Generating crisis detection dataset (3000 samples)...")
    crisis_data = generate_enhanced_crisis_dataset(3000)
    datasets['crisis_detector'] = crisis_data
    
    # 3. Content Filtering Dataset (4000 samples)
    print("  📊 Generating content filtering dataset (4000 samples)...")
    content_data = generate_enhanced_content_dataset(4000)
    datasets['content_filter'] = content_data
    
    # 4. Escalation Detection Dataset (2000 samples)
    print("  📊 Generating escalation detection dataset (2000 samples)...")
    escalation_data = generate_enhanced_escalation_dataset(2000)
    datasets['escalation_detector'] = escalation_data
    
    print(f"✅ Generated {sum(len(data) for data in datasets.values())} total training samples")
    return datasets


def generate_enhanced_abuse_dataset(count: int) -> List[Dict[str, Any]]:
    """Generate enhanced abuse detection dataset with diverse examples."""
    data = []
    
    # Safe content templates (40%)
    safe_templates = [
        "Hello, how are you today?",
        "Thank you for your help!",
        "Have a wonderful day!",
        "I appreciate your assistance.",
        "This is a great conversation.",
        "What a beautiful day it is!",
        "I'm excited about the project.",
        "Thanks for the information.",
        "That's very helpful.",
        "I'm looking forward to it.",
        "Great job on the presentation!",
        "I love this new feature.",
        "The weather is nice today.",
        "I'm feeling optimistic.",
        "This is exactly what I needed."  
    ]
    
    # Mild abuse templates (30%)
    mild_abuse_templates = [
        "This is damn frustrating!",
        "What the hell is going on?",
        "This is so stupid!",
        "I can't believe this crap!",
        "This is ridiculous!",
        "That's bullshit!",
        "This sucks!",
        "What a waste of time!",
        "This is annoying as hell!",
        "I'm so pissed off!",
        "This is garbage!",
        "What the f*** is this?",
        "This is a joke!",
        "I hate this!",
        "This is terrible!"
    ]
    
    # Severe abuse templates (20%)
    severe_abuse_templates = [
        "I hate you, you stupid idiot!",
        "You are a worthless piece of shit!",
        "I want to hurt you badly!",
        "You are a fucking moron!",
        "I despise everything about you!",
        "You're a complete asshole!",
        "I wish you would die!",
        "You're a fucking loser!",
        "I want to kill you!",
        "You're a piece of garbage!",
        "I hate your guts!",
        "You're a fucking waste of space!",
        "I want to destroy you!",
        "You're a fucking idiot!",
        "I despise you completely!"
    ]
    
    # Multilingual abuse (10%)
    multilingual_abuse = [
        "Je te hais, espèce d'idiot!",  # French
        "Te odio, estúpido!",  # Spanish
        "Ich hasse dich, du Idiot!",  # German
        "Je te déteste!",  # French
        "Te odio completamente!",  # Spanish
        "Du bist ein Idiot!",  # German
        "Je veux te faire mal!",  # French
        "Quiero hacerte daño!",  # Spanish
        "Ich will dir wehtun!",  # German
        "Je te méprise!",  # French
        "Te desprecio!",  # Spanish
        "Du bist wertlos!",  # German
        "Je te déteste, crétin!",  # French
        "Te odio, imbécil!",  # Spanish
        "Du bist ein Trottel!"  # German
    ]
    
    # Generate safe content (40%)
    safe_count = int(count * 0.4)
    for _ in range(safe_count):
        text = random.choice(safe_templates)
        data.append({
            'text': text,
            'label': 0,
            'category': 'safe',
            'language': 'en'
        })
    
    # Generate mild abuse (30%)
    mild_count = int(count * 0.3)
    for _ in range(mild_count):
        text = random.choice(mild_abuse_templates)
        data.append({
            'text': text,
            'label': 1,
            'category': 'mild_abuse',
            'language': 'en'
        })
    
    # Generate severe abuse (20%)
    severe_count = int(count * 0.2)
    for _ in range(severe_count):
        text = random.choice(severe_abuse_templates)
        data.append({
            'text': text,
            'label': 1,
            'category': 'severe_abuse',
            'language': 'en'
        })
    
    # Generate multilingual abuse (10%)
    multi_count = count - safe_count - mild_count - severe_count
    for _ in range(multi_count):
        text = random.choice(multilingual_abuse)
        data.append({
            'text': text,
            'label': 1,
            'category': 'multilingual_abuse',
            'language': random.choice(['fr', 'es', 'de'])
        })
    
    random.shuffle(data)
    return data


def generate_enhanced_crisis_dataset(count: int) -> List[Dict[str, Any]]:
    """Generate enhanced crisis detection dataset with comprehensive patterns."""
    data = []
    
    # Safe content (50%)
    safe_templates = [
        "I am feeling great today!",
        "Life is wonderful!",
        "I am so happy!",
        "Today is amazing!",
        "I love my life!",
        "Everything is going well!",
        "I'm excited about the future!",
        "I feel optimistic!",
        "I'm grateful for everything!",
        "I'm looking forward to tomorrow!",
        "I feel confident!",
        "I'm proud of my achievements!",
        "I feel blessed!",
        "I'm content with my life!",
        "I feel peaceful!"
    ]
    
    # Crisis indicators (50%)
    crisis_templates = [
        # Immediate threat
        "I want to kill myself.",
        "I want to end it all.",
        "I want to die.",
        "I want to hurt myself.",
        "I want to cut myself.",
        "I want to overdose.",
        "I want to jump off a bridge.",
        "I want to end my life.",
        "I want to take pills.",
        "I want to poison myself.",
        
        # Severe distress
        "I can't go on anymore.",
        "Everything is hopeless.",
        "I want to die.",
        "I hate my life.",
        "I'm worthless.",
        "Nobody cares about me.",
        "I'm alone and isolated.",
        "I'm a burden to everyone.",
        "I want to disappear.",
        "I can't take it anymore.",
        
        # Emotional crisis
        "I'm having a breakdown.",
        "I'm falling apart.",
        "I'm losing my mind.",
        "I can't cope anymore.",
        "I'm overwhelmed.",
        "I'm drowning in problems.",
        "I'm suffocating.",
        "I'm having a panic attack.",
        "I'm in crisis.",
        "I need help urgently.",
        
        # Substance crisis
        "I'm drunk and want to die.",
        "I'm high and feeling suicidal.",
        "I'm addicted and hopeless.",
        "I'm overdosing on purpose.",
        "I'm drinking to forget.",
        "I'm using drugs to escape.",
        "I'm addicted and want to die.",
        "I'm drunk and suicidal.",
        "I'm high and depressed.",
        "I'm using to numb the pain.",
        
        # Relationship crisis
        "My partner left me and I want to die.",
        "I'm being abused and want to end it.",
        "I'm being bullied and can't take it.",
        "I'm rejected by everyone.",
        "I'm betrayed and want to die.",
        "I'm alone and abandoned.",
        "I'm being harassed and want to die.",
        "I'm being threatened and scared.",
        "I'm being hurt and want to end it.",
        "I'm being manipulated and want to die."
    ]
    
    # Generate safe content (50%)
    safe_count = int(count * 0.5)
    for _ in range(safe_count):
        text = random.choice(safe_templates)
        data.append({
            'text': text,
            'label': 0,
            'category': 'safe',
            'language': 'en'
        })
    
    # Generate crisis content (50%)
    crisis_count = count - safe_count
    for _ in range(crisis_count):
        text = random.choice(crisis_templates)
        data.append({
            'text': text,
            'label': 1,
            'category': 'crisis',
            'language': 'en'
        })
    
    random.shuffle(data)
    return data


def generate_enhanced_content_dataset(count: int) -> List[Dict[str, Any]]:
    """Generate enhanced content filtering dataset with age-appropriate content."""
    data = []
    
    # Safe content (70%)
    safe_templates = [
        "Hello, how are you today?",
        "Thank you for your help!",
        "Have a wonderful day!",
        "I appreciate your assistance.",
        "This is a great conversation.",
        "What a beautiful day it is!",
        "I'm excited about the project.",
        "Thanks for the information.",
        "That's very helpful.",
        "I'm looking forward to it.",
        "Great job on the presentation!",
        "I love this new feature.",
        "The weather is nice today.",
        "I'm feeling optimistic.",
        "This is exactly what I needed."
    ]
    
    # Adult content (30%)
    adult_templates = [
        "This contains explicit sexual content.",
        "Graphic violence and gore.",
        "This has strong language throughout.",
        "This contains drug use and abuse.",
        "This has disturbing imagery.",
        "This contains adult themes.",
        "This has mature content.",
        "This contains profanity.",
        "This has sexual references.",
        "This contains violence.",
        "This has drug references.",
        "This contains disturbing content.",
        "This has explicit language.",
        "This contains adult situations.",
        "This has mature themes."
    ]
    
    # Generate safe content (70%)
    safe_count = int(count * 0.7)
    for _ in range(safe_count):
        text = random.choice(safe_templates)
        data.append({
            'text': text,
            'label': 0,
            'category': 'safe',
            'age_group': random.choice(['child', 'teen', 'adult']),
            'language': 'en'
        })
    
    # Generate adult content (30%)
    adult_count = count - safe_count
    for _ in range(adult_count):
        text = random.choice(adult_templates)
        data.append({
            'text': text,
            'label': 1,
            'category': 'adult',
            'age_group': random.choice(['child', 'teen']),  # Inappropriate for these age groups
            'language': 'en'
        })
    
    random.shuffle(data)
    return data


def generate_enhanced_escalation_dataset(count: int) -> List[Dict[str, Any]]:
    """Generate enhanced escalation detection dataset with conversation patterns."""
    data = []
    
    # Non-escalating conversations (60%)
    safe_conversations = [
        "Hello, how are you?",
        "I'm doing well, thank you!",
        "That's great to hear.",
        "How was your day?",
        "It was good, thanks for asking.",
        "I'm glad to hear that.",
        "What are you up to?",
        "Just working on some projects.",
        "That sounds interesting.",
        "Yes, it's quite engaging."
    ]
    
    # Escalating conversations (40%)
    escalating_conversations = [
        "I'm frustrated with this situation.",
        "I'm getting really angry now.",
        "This is making me furious!",
        "I can't take this anymore!",
        "I'm about to explode!",
        "I'm so mad I could scream!",
        "This is driving me crazy!",
        "I'm losing my temper!",
        "I'm getting really upset!",
        "I'm about to lose it!"
    ]
    
    # Generate non-escalating content (60%)
    safe_count = int(count * 0.6)
    for _ in range(safe_count):
        text = random.choice(safe_conversations)
        data.append({
            'text': text,
            'label': 0,
            'category': 'non_escalating',
            'language': 'en'
        })
    
    # Generate escalating content (40%)
    escalating_count = count - safe_count
    for _ in range(escalating_count):
        text = random.choice(escalating_conversations)
        data.append({
            'text': text,
            'label': 1,
            'category': 'escalating',
            'language': 'en'
        })
    
    random.shuffle(data)
    return data


def load_training_data(data_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load training data from JSON file."""
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"❌ Training data file not found: {data_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in training data: {e}")
        sys.exit(1)


def load_synthetic_data(synthetic_data_dir: str = "data/synthetic") -> Dict[str, List[Dict[str, Any]]]:
    """Load existing synthetic training data from individual model files."""
    print("📊 Loading existing synthetic training data...")
    
    datasets = {}
    synthetic_files = {
        'abuse_detector': 'abuse_detector_data.json',
        'crisis_detector': 'crisis_detector_data.json', 
        'content_filter': 'content_filter_data.json',
        'escalation_detector': 'escalation_detector_data.json'
    }
    
    for model_name, filename in synthetic_files.items():
        file_path = os.path.join(synthetic_data_dir, filename)
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                datasets[model_name] = data
                print(f"  ✅ Loaded {len(data)} samples for {model_name}")
            else:
                print(f"  ⚠️  File not found: {file_path}")
                datasets[model_name] = []
        except json.JSONDecodeError as e:
            print(f"  ❌ Error loading {filename}: {e}")
            datasets[model_name] = []
    
    total_samples = sum(len(data) for data in datasets.values())
    print(f"✅ Loaded {total_samples} total synthetic training samples")
    return datasets


def train_single_model(model_name: str, model, train_data: List[Dict[str, Any]], 
                      val_data: List[Dict[str, Any]] = None) -> Dict[str, float]:
    """Train a single model and return metrics."""
    print(f"🏋️  Training {model_name}...")
    
    try:
        # Check if model has train method
        if hasattr(model, 'train'):
            metrics = model.train(train_data, val_data)
        else:
            # For models without train method, use fit
            if hasattr(model, 'fit'):
                # Extract features and labels
                X = [item['text'] for item in train_data]
                y = [item['label'] for item in train_data]
                model.fit(X, y)
                metrics = {'status': 'trained'}
            else:
                metrics = {'status': 'no_training_method'}
        
        print(f"✅ {model_name} training completed")
        return metrics
    except Exception as e:
        print(f"❌ Error training {model_name}: {e}")
        return {'error': str(e)}


def split_data(data: List[Dict[str, Any]], train_ratio: float = 0.8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data into training and validation sets."""
    np.random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def train_models(data_path: str = None, output_path: str = "trained_models", 
                use_synthetic_data: bool = False) -> Dict[str, Any]:
    """Train all AI Safety Models."""
    print("🚀 Enhanced AI Safety Models Training")
    print("=" * 40)
    
    # Load or generate training data
    if use_synthetic_data or data_path is None:
        print("📊 Loading existing synthetic training data...")
        training_data = load_synthetic_data()
    else:
        print(f"📊 Loading training data from: {data_path}")
        training_data = load_training_data(data_path)
    
    # Initialize safety manager with enhanced models
    safety_manager = SafetyManager()
    
    # Training results
    results = {
        'timestamp': datetime.now().isoformat(),
        'models': {},
        'overall_metrics': {},
        'data_summary': {}
    }
    
    # Data summary
    for model_name, data in training_data.items():
        results['data_summary'][model_name] = {
            'total_samples': len(data),
            'positive_samples': sum(1 for item in data if item['label'] == 1),
            'negative_samples': sum(1 for item in data if item['label'] == 0)
        }
    
    # Train each model
    for model_name, model in safety_manager.models.items():
        if model_name in training_data:
            print(f"\n🔧 Training {model_name}...")
            
            # Get training data for this model
            model_data = training_data[model_name]
            
            # Split into train/validation
            train_data, val_data = split_data(model_data.copy())
            
            print(f"  📚 Training samples: {len(train_data)}")
            print(f"  📊 Validation samples: {len(val_data)}")
            
            # Train the model
            metrics = train_single_model(model_name, model, train_data, val_data)
            results['models'][model_name] = metrics
            
            # Save the trained model
            model_output_path = os.path.join(output_path, f"{model_name}.pkl")
            os.makedirs(output_path, exist_ok=True)
            
            # Check if model has save_model method
            if hasattr(model, 'save_model'):
                model.save_model(model_output_path)
                print(f"  💾 Model saved to: {model_output_path}")
            else:
                print(f"  ⚠️  Model {model_name} doesn't have save_model method")
        else:
            print(f"⚠️  No training data found for {model_name}, skipping...")
            results['models'][model_name] = {'skipped': True}
    
    # Calculate overall metrics
    all_metrics = []
    for model_name, metrics in results['models'].items():
        if 'error' not in metrics and 'skipped' not in metrics:
            if 'train_accuracy' in metrics:
                all_metrics.append(metrics['train_accuracy'])
    
    if all_metrics:
        results['overall_metrics'] = {
            'average_accuracy': np.mean(all_metrics),
            'min_accuracy': np.min(all_metrics),
            'max_accuracy': np.max(all_metrics),
            'models_trained': len(all_metrics)
        }
    
    # Save training results
    results_path = os.path.join(output_path, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📈 Training Summary:")
    print(f"  Models trained: {results['overall_metrics'].get('models_trained', 0)}")
    print(f"  Average accuracy: {results['overall_metrics'].get('average_accuracy', 0):.3f}")
    print(f"  Results saved to: {results_path}")
    
    # Print data summary
    print(f"\n📊 Data Summary:")
    for model_name, summary in results['data_summary'].items():
        print(f"  {model_name}: {summary['total_samples']} samples ({summary['positive_samples']} positive, {summary['negative_samples']} negative)")
    
    return results


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(description='Train Enhanced AI Safety Models')
    parser.add_argument('--data-path', type=str,
                       help='Path to training data JSON file')
    parser.add_argument('--output-path', type=str, default='trained_models',
                       help='Output directory for trained models')
    parser.add_argument('--use-synthetic-data', action='store_true',
                       help='Use existing synthetic training data from data/synthetic/ directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        results = train_models(
            data_path=args.data_path,
            output_path=args.output_path,
            use_synthetic_data=args.use_synthetic_data
        )
        
        print("\n✅ Training completed successfully!")
        
        if args.verbose:
            print("\n📊 Detailed Results:")
            print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()