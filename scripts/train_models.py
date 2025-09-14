#!/usr/bin/env python3
"""
Train AI Safety Models.

This script provides functionality to train the AI Safety Models on custom datasets.
"""

import sys
import os
import argparse
import json
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from safety_system.safety_manager import SafetyManager
from core.base_model import ModelConfig


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


def generate_sample_data() -> Dict[str, List[Dict[str, Any]]]:
    """Generate sample training data for demonstration."""
    sample_data = {
        'abuse_detector': [
            {'text': 'Hello, how are you?', 'label': 0},
            {'text': 'This is great!', 'label': 0},
            {'text': 'I hate you!', 'label': 1},
            {'text': 'You are stupid!', 'label': 1},
            {'text': 'What a wonderful day!', 'label': 0},
            {'text': 'I want to hurt you!', 'label': 1},
            {'text': 'Thanks for your help!', 'label': 0},
            {'text': 'You are an idiot!', 'label': 1},
            {'text': 'Have a nice day!', 'label': 0},
            {'text': 'I hate everything!', 'label': 1}
        ],
        'crisis_detector': [
            {'text': 'I am feeling great today!', 'label': 0},
            {'text': 'Life is wonderful!', 'label': 0},
            {'text': 'I want to kill myself.', 'label': 1},
            {'text': 'I can\'t go on anymore.', 'label': 1},
            {'text': 'Everything is hopeless.', 'label': 1},
            {'text': 'I need help.', 'label': 1},
            {'text': 'I am so happy!', 'label': 0},
            {'text': 'I want to die.', 'label': 1},
            {'text': 'Today is amazing!', 'label': 0},
            {'text': 'I hate my life.', 'label': 1}
        ]
    }
    return sample_data


def train_single_model(model_name: str, model, train_data: List[Dict[str, Any]], 
                      val_data: List[Dict[str, Any]] = None) -> Dict[str, float]:
    """Train a single model and return metrics."""
    print(f"🏋️  Training {model_name}...")
    
    try:
        metrics = model.train(train_data, val_data)
        print(f"✅ {model_name} training completed")
        return metrics
    except Exception as e:
        print(f"❌ Error training {model_name}: {e}")
        return {'error': str(e)}


def split_data(data: List[Dict[str, Any]], train_ratio: float = 0.8) -> tuple:
    """Split data into training and validation sets."""
    np.random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def train_models(data_path: str = None, output_path: str = "models", 
                use_sample_data: bool = False) -> Dict[str, Any]:
    """Train all AI Safety Models."""
    print("🚀 AI Safety Models Training")
    print("=" * 35)
    
    # Load or generate training data
    if use_sample_data or data_path is None:
        print("📊 Using sample training data...")
        training_data = generate_sample_data()
    else:
        print(f"📊 Loading training data from: {data_path}")
        training_data = load_training_data(data_path)
    
    # Initialize safety manager
    safety_manager = SafetyManager()
    
    # Training results
    results = {
        'timestamp': datetime.now().isoformat(),
        'models': {},
        'overall_metrics': {}
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
            model.save_model(model_output_path)
            print(f"  💾 Model saved to: {model_output_path}")
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
    
    return results


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(description='Train AI Safety Models')
    parser.add_argument('--data-path', type=str,
                       help='Path to training data JSON file')
    parser.add_argument('--output-path', type=str, default='models',
                       help='Output directory for trained models')
    parser.add_argument('--use-sample-data', action='store_true',
                       help='Use sample training data for demonstration')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        results = train_models(
            data_path=args.data_path,
            output_path=args.output_path,
            use_sample_data=args.use_sample_data
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