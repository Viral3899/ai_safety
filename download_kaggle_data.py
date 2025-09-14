#!/usr/bin/env python3
"""
Download Real Training Data from Kaggle for AI Safety Models.

This script downloads real datasets from Kaggle to replace synthetic data.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any
import requests
import zipfile
from pathlib import Path

# Note: This script requires kaggle API setup
# pip install kaggle
# Setup API credentials: https://www.kaggle.com/docs/api

def setup_kaggle_api():
    """Setup Kaggle API credentials."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Instructions for user
    print("🔑 Kaggle API Setup Required:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Create API token (kaggle.json)")
    print("3. Place kaggle.json in ~/.kaggle/ directory")
    print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    print()
    
    # Check if credentials exist
    kaggle_file = kaggle_dir / 'kaggle.json'
    if kaggle_file.exists():
        print("✅ Kaggle credentials found")
        return True
    else:
        print("❌ Kaggle credentials not found")
        return False

def download_dataset(dataset_name: str, output_dir: str):
    """Download a dataset from Kaggle."""
    try:
        import kaggle
        print(f"📥 Downloading {dataset_name}...")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name, 
            path=output_dir, 
            unzip=True
        )
        
        print(f"✅ Downloaded {dataset_name} successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {dataset_name}: {e}")
        return False

def download_abuse_detection_data(output_dir: str):
    """Download abuse detection dataset."""
    datasets = [
        "jigsaw-toxic-comment-classification-challenge",
        "jigsaw-unintended-bias-in-toxicity-classification",
        "detecting-insults-in-social-commentary"
    ]
    
    for dataset in datasets:
        if download_dataset(dataset, output_dir):
            break
    else:
        print("⚠️  Could not download abuse detection data, using synthetic data")

def download_crisis_detection_data(output_dir: str):
    """Download crisis detection dataset."""
    datasets = [
        "suicide-and-depression-detection",
        "mental-health-in-tech-survey",
        "reddit-mental-health-posts"
    ]
    
    for dataset in datasets:
        if download_dataset(dataset, output_dir):
            break
    else:
        print("⚠️  Could not download crisis detection data, using synthetic data")

def download_content_filtering_data(output_dir: str):
    """Download content filtering dataset."""
    datasets = [
        "common-crawl-news-comments",
        "youtube-comments-classification",
        "social-media-content-moderation"
    ]
    
    for dataset in datasets:
        if download_dataset(dataset, output_dir):
            break
    else:
        print("⚠️  Could not download content filtering data, using synthetic data")

def process_jigsaw_data(data_dir: str) -> Dict[str, Any]:
    """Process Jigsaw toxic comment classification data."""
    try:
        train_file = os.path.join(data_dir, "train.csv")
        test_file = os.path.join(data_dir, "test.csv")
        
        if not os.path.exists(train_file):
            return {}
        
        # Load training data
        df = pd.read_csv(train_file)
        
        # Process for abuse detection
        processed_data = []
        
        for _, row in df.iterrows():
            text = row['comment_text']
            
            # Create binary label (toxic or not)
            is_toxic = row['toxic'] == 1
            
            processed_data.append({
                'text': text,
                'label': 1 if is_toxic else 0,
                'category': 'toxic' if is_toxic else 'safe',
                'severity': 'high' if row.get('severe_toxic', 0) == 1 else 'medium'
            })
        
        # Save processed data
        output_file = os.path.join(data_dir, "processed_abuse_data.json")
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"✅ Processed {len(processed_data)} abuse detection samples")
        return {
            'dataset': 'jigsaw_toxic_comments',
            'samples': len(processed_data),
            'file': output_file
        }
        
    except Exception as e:
        print(f"❌ Error processing Jigsaw data: {e}")
        return {}

def create_data_manifest(data_dir: str):
    """Create a manifest of available datasets."""
    manifest = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'datasets': [],
        'total_samples': 0
    }
    
    # Check for processed datasets
    processed_files = [
        'processed_abuse_data.json',
        'processed_crisis_data.json', 
        'processed_content_data.json'
    ]
    
    for file_name in processed_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            manifest['datasets'].append({
                'file': file_name,
                'samples': len(data),
                'type': file_name.replace('processed_', '').replace('.json', '')
            })
            manifest['total_samples'] += len(data)
    
    # Save manifest
    manifest_file = os.path.join(data_dir, 'data_manifest.json')
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"📋 Created data manifest: {manifest_file}")
    return manifest

def main():
    """Main function to download and process Kaggle datasets."""
    print("🚀 AI Safety Models - Real Data Download")
    print("=" * 45)
    
    # Setup output directory
    data_dir = "kaggle_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Check Kaggle API setup
    if not setup_kaggle_api():
        print("⚠️  Skipping Kaggle downloads, using synthetic data")
        return
    
    # Download datasets
    print("\n📥 Downloading datasets from Kaggle...")
    
    # Download abuse detection data
    print("\n🔍 Downloading Abuse Detection Data:")
    download_abuse_detection_data(data_dir)
    
    # Download crisis detection data
    print("\n🚨 Downloading Crisis Detection Data:")
    download_crisis_detection_data(data_dir)
    
    # Download content filtering data
    print("\n👶 Downloading Content Filtering Data:")
    download_content_filtering_data(data_dir)
    
    # Process downloaded data
    print("\n⚙️  Processing downloaded data...")
    
    # Process Jigsaw data if available
    jigsaw_result = process_jigsaw_data(data_dir)
    if jigsaw_result:
        print(f"✅ Processed Jigsaw dataset: {jigsaw_result['samples']} samples")
    
    # Create data manifest
    manifest = create_data_manifest(data_dir)
    
    print(f"\n📊 Data Download Summary:")
    print(f"  Total samples: {manifest['total_samples']}")
    print(f"  Datasets: {len(manifest['datasets'])}")
    print(f"  Output directory: {data_dir}")
    
    print("\n✅ Real data download completed!")
    print("📝 Next steps:")
    print("  1. Review downloaded datasets")
    print("  2. Run train_all_models.py with real data")
    print("  3. Evaluate model performance on real data")

if __name__ == "__main__":
    main()
