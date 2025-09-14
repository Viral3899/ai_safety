"""
Data generator for AI safety models.
"""

import random
import json
from typing import Dict, Any, List
from datetime import datetime


class DataGenerator:
    """Data generator for AI Safety Models."""
    
    def __init__(self):
        """Initialize the data generator."""
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load text templates for data generation."""
        return {
            'safe_content': [
                "Hello, how are you today?",
                "Thank you for your help!",
                "Have a wonderful day!",
                "I appreciate your assistance.",
                "This is a great conversation."
            ],
            'mild_abuse': [
                "This is damn frustrating!",
                "What the hell is going on?",
                "This is so stupid!",
                "I can't believe this crap!",
                "This is ridiculous!"
            ],
            'severe_abuse': [
                "I hate you, you stupid idiot!",
                "You are a worthless piece of shit!",
                "I want to hurt you badly!",
                "You are a fucking moron!",
                "I despise everything about you!"
            ],
            'crisis_indicators': [
                "I want to kill myself.",
                "I can't go on anymore.",
                "I want to end it all.",
                "Nobody would miss me if I was gone.",
                "I want to hurt myself."
            ],
            'adult_content': [
                "This contains explicit sexual content.",
                "Graphic violence and gore.",
                "This has strong language throughout.",
                "This contains drug use and abuse.",
                "This has disturbing imagery."
            ]
        }
    
    def generate_dataset(self, dataset_type: str, count: int = 1000) -> List[Dict[str, Any]]:
        """Generate a dataset of specified type and count."""
        if dataset_type == 'abuse_detection':
            return self._generate_abuse_dataset(count)
        elif dataset_type == 'crisis_detection':
            return self._generate_crisis_dataset(count)
        elif dataset_type == 'content_filtering':
            return self._generate_content_filtering_dataset(count)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _generate_abuse_dataset(self, count: int) -> List[Dict[str, Any]]:
        """Generate abuse detection dataset."""
        data = []
        
        # 60% safe content
        safe_count = int(count * 0.6)
        for _ in range(safe_count):
            text = random.choice(self.templates['safe_content'])
            data.append({
                'text': text,
                'label': 0,
                'category': 'safe'
            })
        
        # 25% mild abuse
        mild_count = int(count * 0.25)
        for _ in range(mild_count):
            text = random.choice(self.templates['mild_abuse'])
            data.append({
                'text': text,
                'label': 1,
                'category': 'mild_abuse'
            })
        
        # 15% severe abuse
        severe_count = count - safe_count - mild_count
        for _ in range(severe_count):
            text = random.choice(self.templates['severe_abuse'])
            data.append({
                'text': text,
                'label': 1,
                'category': 'severe_abuse'
            })
        
        random.shuffle(data)
        return data
    
    def _generate_crisis_dataset(self, count: int) -> List[Dict[str, Any]]:
        """Generate crisis detection dataset."""
        data = []
        
        # 50% safe content
        safe_count = int(count * 0.5)
        for _ in range(safe_count):
            text = random.choice(self.templates['safe_content'])
            data.append({
                'text': text,
                'label': 0,
                'category': 'safe'
            })
        
        # 50% crisis indicators
        crisis_count = count - safe_count
        for _ in range(crisis_count):
            text = random.choice(self.templates['crisis_indicators'])
            data.append({
                'text': text,
                'label': 1,
                'category': 'crisis'
            })
        
        random.shuffle(data)
        return data
    
    def _generate_content_filtering_dataset(self, count: int) -> List[Dict[str, Any]]:
        """Generate content filtering dataset."""
        data = []
        
        # 70% safe content
        safe_count = int(count * 0.7)
        for _ in range(safe_count):
            text = random.choice(self.templates['safe_content'])
            data.append({
                'text': text,
                'label': 0,
                'category': 'safe',
                'age_group': random.choice(['child', 'teen', 'adult'])
            })
        
        # 30% adult content
        adult_count = count - safe_count
        for _ in range(adult_count):
            text = random.choice(self.templates['adult_content'])
            data.append({
                'text': text,
                'label': 1,
                'category': 'adult',
                'age_group': random.choice(['child', 'teen'])
            })
        
        random.shuffle(data)
        return data
    
    def save_dataset(self, data: List[Dict[str, Any]], output_path: str):
        """Save dataset to file."""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_dataset(self, input_path: str) -> List[Dict[str, Any]]:
        """Load dataset from file."""
        with open(input_path, 'r') as f:
            return json.load(f)