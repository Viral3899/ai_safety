#!/usr/bin/env python3
"""
Generate synthetic data for AI safety models.
"""

import sys
import os
import argparse
import json
import random
from typing import Dict, Any, List
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def generate_abuse_data(count: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic abuse detection data."""
    safe_templates = [
        "Hello, how are you today?",
        "Thank you for your help!",
        "Have a wonderful day!",
        "I appreciate your assistance.",
        "This is a great conversation.",
        "I'm looking forward to our meeting.",
        "The weather is nice today.",
        "I enjoyed our discussion.",
        "Please let me know if you need anything.",
        "I hope you have a good weekend!"
    ]
    
    mild_abuse_templates = [
        "This is damn frustrating!",
        "What the hell is going on?",
        "This is so stupid!",
        "I can't believe this crap!",
        "This is ridiculous!",
        "What the heck is this?",
        "This is annoying as hell!",
        "I'm so pissed off!",
        "This is complete bullshit!",
        "What a stupid idea!"
    ]
    
    severe_abuse_templates = [
        "I hate you, you stupid idiot!",
        "You are a worthless piece of shit!",
        "I want to hurt you badly!",
        "You are a fucking moron!",
        "I despise everything about you!",
        "You are a pathetic loser!",
        "I want to destroy you!",
        "You are a complete asshole!",
        "I hate your fucking guts!",
        "You are a worthless waste of space!"
    ]
    
    data = []
    
    # Generate safe content
    safe_count = int(count * 0.6)
    for _ in range(safe_count):
        text = random.choice(safe_templates)
        data.append({
            'text': text,
            'label': 0,
            'category': 'safe',
            'severity': 'none'
        })
    
    # Generate mild abuse
    mild_count = int(count * 0.25)
    for _ in range(mild_count):
        text = random.choice(mild_abuse_templates)
        data.append({
            'text': text,
            'label': 1,
            'category': 'mild_abuse',
            'severity': 'mild'
        })
    
    # Generate severe abuse
    severe_count = count - safe_count - mild_count
    for _ in range(severe_count):
        text = random.choice(severe_abuse_templates)
        data.append({
            'text': text,
            'label': 1,
            'category': 'severe_abuse',
            'severity': 'severe'
        })
    
    random.shuffle(data)
    return data


def generate_crisis_data(count: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic crisis detection data."""
    safe_templates = [
        "I am feeling great today!",
        "Life is wonderful!",
        "I love my family!",
        "Everything is going well!",
        "I'm excited about the future!",
        "Today is a beautiful day!",
        "I'm grateful for everything!",
        "I feel happy and content!",
        "Life is full of possibilities!",
        "I'm looking forward to tomorrow!"
    ]
    
    mild_concern_templates = [
        "I am feeling a bit down today.",
        "I have been struggling lately.",
        "I'm going through a tough time.",
        "I feel overwhelmed sometimes.",
        "I've been feeling sad recently.",
        "I'm having a hard time coping.",
        "I feel stressed and anxious.",
        "I'm not feeling like myself.",
        "I've been having negative thoughts.",
        "I feel hopeless sometimes."
    ]
    
    crisis_templates = [
        "I want to kill myself.",
        "I can't go on anymore.",
        "I want to end it all.",
        "Nobody would miss me if I was gone.",
        "I want to hurt myself.",
        "I don't see any point in living.",
        "I want to die.",
        "I'm planning to end my life.",
        "I have nothing left to live for.",
        "I want to disappear forever."
    ]
    
    help_seeking_templates = [
        "I need help with my mental health.",
        "Can someone talk to me?",
        "I'm reaching out for support.",
        "I need someone to listen.",
        "I'm looking for professional help.",
        "I need to talk to someone.",
        "I'm asking for help.",
        "I need emotional support.",
        "I'm seeking counseling.",
        "I need mental health resources."
    ]
    
    data = []
    
    # Generate safe content
    safe_count = int(count * 0.5)
    for _ in range(safe_count):
        text = random.choice(safe_templates)
        data.append({
            'text': text,
            'label': 0,
            'category': 'safe',
            'severity': 'none'
        })
    
    # Generate mild concern
    mild_count = int(count * 0.2)
    for _ in range(mild_count):
        text = random.choice(mild_concern_templates)
        data.append({
            'text': text,
            'label': 1,
            'category': 'mild_concern',
            'severity': 'mild'
        })
    
    # Generate crisis indicators
    crisis_count = int(count * 0.2)
    for _ in range(crisis_count):
        text = random.choice(crisis_templates)
        data.append({
            'text': text,
            'label': 1,
            'category': 'crisis',
            'severity': 'severe'
        })
    
    # Generate help seeking
    help_count = count - safe_count - mild_count - crisis_count
    for _ in range(help_count):
        text = random.choice(help_seeking_templates)
        data.append({
            'text': text,
            'label': 1,
            'category': 'help_seeking',
            'severity': 'moderate'
        })
    
    random.shuffle(data)
    return data


def generate_content_filter_data(count: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic content filtering data."""
    child_safe_templates = [
        "Let's learn about animals!",
        "I love my family!",
        "School is fun!",
        "I like to play with my friends!",
        "Learning is exciting!",
        "I love my pet dog!",
        "My teacher is nice!",
        "I enjoy reading books!",
        "Playing outside is fun!",
        "I love my parents!"
    ]
    
    teen_appropriate_templates = [
        "This movie has some action scenes.",
        "There is mild violence in this content.",
        "This contains some scary elements.",
        "There are some intense moments.",
        "This has some dramatic content.",
        "There is some mild profanity.",
        "This contains some mature themes.",
        "There are some emotional scenes.",
        "This has some suspenseful moments.",
        "There is some conflict in the story."
    ]
    
    adult_content_templates = [
        "This contains explicit sexual content.",
        "Graphic violence and gore.",
        "This has strong language throughout.",
        "This contains drug use and abuse.",
        "This has disturbing imagery.",
        "This contains graphic sexual scenes.",
        "This has extreme violence.",
        "This contains disturbing themes.",
        "This has explicit content.",
        "This contains mature sexual content."
    ]
    
    data = []
    
    # Generate child-safe content
    child_count = int(count * 0.4)
    for _ in range(child_count):
        text = random.choice(child_safe_templates)
        data.append({
            'text': text,
            'label': 0,
            'age_group': 'child',
            'category': 'safe',
            'rating': 'G'
        })
    
    # Generate teen-appropriate content
    teen_count = int(count * 0.3)
    for _ in range(teen_count):
        text = random.choice(teen_appropriate_templates)
        # Some teen content might be flagged
        label = 1 if random.random() < 0.3 else 0
        data.append({
            'text': text,
            'label': label,
            'age_group': 'teen',
            'category': 'moderate',
            'rating': 'PG-13' if label else 'PG'
        })
    
    # Generate adult content
    adult_count = count - child_count - teen_count
    for _ in range(adult_count):
        text = random.choice(adult_content_templates)
        # Adult content should be flagged for teens and children
        age_group = random.choice(['teen', 'child'])
        data.append({
            'text': text,
            'label': 1,
            'age_group': age_group,
            'category': 'adult',
            'rating': 'R'
        })
    
    random.shuffle(data)
    return data


def generate_synthetic_data(output_path: str = "synthetic_data", 
                          total_count: int = 10000) -> Dict[str, Any]:
    """Generate comprehensive synthetic data for all models."""
    print("🚀 Generating Synthetic Data for AI Safety Models")
    print("=" * 50)
    
    # Calculate counts per model
    per_model_count = total_count // 3
    
    print(f"📊 Generating {total_count} total samples:")
    print(f"  - Abuse Detection: {per_model_count} samples")
    print(f"  - Crisis Detection: {per_model_count} samples")
    print(f"  - Content Filtering: {per_model_count} samples")
    
    # Generate data for each model
    synthetic_data = {
        'abuse_detector': generate_abuse_data(per_model_count),
        'crisis_detector': generate_crisis_data(per_model_count),
        'content_filter': generate_content_filter_data(per_model_count)
    }
    
    # Add metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'total_samples': total_count,
        'models': {
            model_name: len(data) for model_name, data in synthetic_data.items()
        },
        'categories': {
            'abuse_detector': list(set(item['category'] for item in synthetic_data['abuse_detector'])),
            'crisis_detector': list(set(item['category'] for item in synthetic_data['crisis_detector'])),
            'content_filter': list(set(item['category'] for item in synthetic_data['content_filter']))
        }
    }
    
    # Save data
    os.makedirs(output_path, exist_ok=True)
    
    # Save individual model data
    for model_name, data in synthetic_data.items():
        model_file = os.path.join(output_path, f"{model_name}_data.json")
        with open(model_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  💾 Saved {model_name}: {len(data)} samples to {model_file}")
    
    # Save combined data
    combined_file = os.path.join(output_path, "combined_data.json")
    with open(combined_file, 'w') as f:
        json.dump(synthetic_data, f, indent=2)
    
    # Save metadata
    metadata_file = os.path.join(output_path, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📈 Generation Summary:")
    print(f"  Total samples: {total_count}")
    print(f"  Models: {len(synthetic_data)}")
    print(f"  Output directory: {output_path}")
    print(f"  Combined data: {combined_file}")
    print(f"  Metadata: {metadata_file}")
    
    return synthetic_data


def main():
    """Main entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(description='Generate Synthetic Data for AI Safety Models')
    parser.add_argument('--output-path', type=str, default='synthetic_data',
                       help='Output directory for generated data')
    parser.add_argument('--count', type=int, default=10000,
                       help='Total number of samples to generate')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        data = generate_synthetic_data(
            output_path=args.output_path,
            total_count=args.count
        )
        
        print("\n✅ Synthetic data generation completed successfully!")
        
        if args.verbose:
            print("\n📊 Data Statistics:")
            for model_name, model_data in data.items():
                print(f"\n  {model_name}:")
                categories = {}
                for item in model_data:
                    cat = item.get('category', 'unknown')
                    categories[cat] = categories.get(cat, 0) + 1
                
                for category, count in categories.items():
                    print(f"    {category}: {count} samples")
        
    except Exception as e:
        print(f"\n❌ Data generation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()