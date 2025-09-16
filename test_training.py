#!/usr/bin/env python3
"""
Test training with the enhanced dataset.
"""

import os
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_data_loading():
    """Test loading the enhanced dataset."""
    print("🧪 Testing Enhanced Dataset Loading")
    print("=" * 40)
    
    models = ['abuse_detector', 'crisis_detector', 'content_filter']
    
    for model in models:
        print(f"\n📊 Testing {model}:")
        
        for split in ['train', 'validation', 'test']:
            file_path = f"data/enhanced/{model}/{split}.json"
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"  {split}: {len(data)} samples")
                
                # Show sample data
                if data:
                    sample = data[0]
                    print(f"    Sample: {sample.get('text', 'N/A')[:50]}...")
                    print(f"    Label: {sample.get('label', 'N/A')}")
                    print(f"    Category: {sample.get('category', 'N/A')}")
            else:
                print(f"  {split}: File not found")

def test_model_imports():
    """Test importing the model classes."""
    print("\n🔧 Testing Model Imports")
    print("=" * 40)
    
    try:
        from src.models.abuse_detector import AbuseDetector
        print("✅ AbuseDetector imported successfully")
    except Exception as e:
        print(f"❌ AbuseDetector import failed: {e}")
    
    try:
        from src.models.crisis_detector import CrisisDetector
        print("✅ CrisisDetector imported successfully")
    except Exception as e:
        print(f"❌ CrisisDetector import failed: {e}")
    
    try:
        from src.models.content_filter import ContentFilter
        print("✅ ContentFilter imported successfully")
    except Exception as e:
        print(f"❌ ContentFilter import failed: {e}")

def test_training_setup():
    """Test basic training setup."""
    print("\n🏋️  Testing Training Setup")
    print("=" * 40)
    
    try:
        # Test loading training data
        train_file = "data/enhanced/abuse_detector/train.json"
        if os.path.exists(train_file):
            with open(train_file, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            
            print(f"✅ Training data loaded: {len(train_data)} samples")
            
            # Check data format
            if train_data:
                sample = train_data[0]
                required_fields = ['text', 'label']
                missing_fields = [field for field in required_fields if field not in sample]
                
                if missing_fields:
                    print(f"❌ Missing required fields: {missing_fields}")
                else:
                    print("✅ Data format looks correct")
        else:
            print("❌ Training data file not found")
    
    except Exception as e:
        print(f"❌ Training setup failed: {e}")

def main():
    """Main test function."""
    print("🚀 AI Safety Models - Training Test")
    print("=" * 50)
    
    # Test data loading
    test_data_loading()
    
    # Test model imports
    test_model_imports()
    
    # Test training setup
    test_training_setup()
    
    print("\n✅ Testing completed!")
    print("\n📝 Next steps:")
    print("1. Run: python scripts/train_models.py")
    print("2. Run: python scripts/evaluate_models.py")
    print("3. Run: python run_demo.py")

if __name__ == "__main__":
    main()
