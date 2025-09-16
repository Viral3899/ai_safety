#!/usr/bin/env python3
"""
Process downloaded Kaggle data for AI safety models.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

def process_mental_health_survey(data_dir: str) -> Dict[str, Any]:
    """Process the mental health survey data for crisis detection."""
    try:
        survey_file = os.path.join(data_dir, "survey.csv")
        
        if not os.path.exists(survey_file):
            return {}
        
        # Load the survey data
        df = pd.read_csv(survey_file)
        
        # Process for crisis detection
        processed_data = []
        
        for _, row in df.iterrows():
            # Extract relevant information for crisis detection
            age = row.get('Age', '')
            gender = row.get('Gender', '')
            treatment = row.get('treatment', '')
            work_interfere = row.get('work_interfere', '')
            family_history = row.get('family_history', '')
            comments = row.get('comments', '')
            
            # Create text representation
            text_parts = []
            
            # Add demographic info
            if pd.notna(age) and str(age).isdigit():
                text_parts.append(f"I am {age} years old")
            
            if pd.notna(gender) and str(gender).strip():
                text_parts.append(f"and I am {gender}")
            
            # Add mental health indicators
            if pd.notna(family_history) and str(family_history).strip():
                text_parts.append(f"My family history of mental illness is: {family_history}")
            
            if pd.notna(treatment) and str(treatment).strip():
                text_parts.append(f"I have sought treatment: {treatment}")
            
            if pd.notna(work_interfere) and str(work_interfere).strip():
                text_parts.append(f"Mental health interferes with work: {work_interfere}")
            
            # Add comments if available
            if pd.notna(comments) and str(comments).strip() and str(comments) != 'NA':
                text_parts.append(f"Additional comments: {comments}")
            
            # Create the text
            text = ". ".join(text_parts) + "."
            
            # Determine crisis level based on indicators
            crisis_score = 0
            
            # High risk indicators
            if str(treatment).lower() == 'yes':
                crisis_score += 2
            if str(family_history).lower() == 'yes':
                crisis_score += 1
            if str(work_interfere).lower() in ['often', 'sometimes']:
                crisis_score += 2
            if pd.notna(comments) and any(word in str(comments).lower() for word in 
                                        ['depressed', 'anxious', 'suicidal', 'help', 'crisis', 'emergency']):
                crisis_score += 3
            
            # Categorize crisis level
            if crisis_score >= 4:
                category = 'high_risk'
                label = 1
            elif crisis_score >= 2:
                category = 'medium_risk'
                label = 1
            else:
                category = 'low_risk'
                label = 0
            
            processed_data.append({
                'text': text,
                'label': label,
                'category': category,
                'crisis_score': crisis_score,
                'source': 'mental_health_survey'
            })
        
        # Save processed data
        output_file = os.path.join(data_dir, "processed_crisis_data.json")
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"✅ Processed {len(processed_data)} crisis detection samples from mental health survey")
        return {
            'dataset': 'mental_health_survey',
            'samples': len(processed_data),
            'file': output_file,
            'high_risk': len([d for d in processed_data if d['category'] == 'high_risk']),
            'medium_risk': len([d for d in processed_data if d['category'] == 'medium_risk']),
            'low_risk': len([d for d in processed_data if d['category'] == 'low_risk'])
        }
        
    except Exception as e:
        print(f"❌ Error processing mental health survey: {e}")
        return {}

def create_enhanced_data_manifest(data_dir: str):
    """Create an enhanced manifest of available datasets."""
    manifest = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'datasets': [],
        'total_samples': 0,
        'crisis_detection': {
            'total_samples': 0,
            'high_risk': 0,
            'medium_risk': 0,
            'low_risk': 0
        }
    }
    
    # Check for processed datasets
    processed_files = [
        'processed_crisis_data.json',
        'processed_abuse_data.json',
        'processed_content_data.json'
    ]
    
    for file_name in processed_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            dataset_info = {
                'file': file_name,
                'samples': len(data),
                'type': file_name.replace('processed_', '').replace('.json', '')
            }
            
            # Add crisis detection specific info
            if 'crisis' in file_name:
                high_risk = len([d for d in data if d.get('category') == 'high_risk'])
                medium_risk = len([d for d in data if d.get('category') == 'medium_risk'])
                low_risk = len([d for d in data if d.get('category') == 'low_risk'])
                
                dataset_info.update({
                    'high_risk': high_risk,
                    'medium_risk': medium_risk,
                    'low_risk': low_risk
                })
                
                manifest['crisis_detection'].update({
                    'total_samples': len(data),
                    'high_risk': high_risk,
                    'medium_risk': medium_risk,
                    'low_risk': low_risk
                })
            
            manifest['datasets'].append(dataset_info)
            manifest['total_samples'] += len(data)
    
    # Save enhanced manifest
    manifest_file = os.path.join(data_dir, 'enhanced_data_manifest.json')
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"📋 Created enhanced data manifest: {manifest_file}")
    return manifest

def main():
    """Main function to process downloaded Kaggle datasets."""
    print("🔄 Processing Downloaded Kaggle Data")
    print("=" * 40)
    
    # Setup data directory
    data_dir = "kaggle_data"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory {data_dir} not found")
        return
    
    # Process mental health survey data
    print("\n🧠 Processing Mental Health Survey Data:")
    survey_result = process_mental_health_survey(data_dir)
    
    if survey_result:
        print(f"✅ Processed {survey_result['samples']} samples:")
        print(f"   - High Risk: {survey_result['high_risk']}")
        print(f"   - Medium Risk: {survey_result['medium_risk']}")
        print(f"   - Low Risk: {survey_result['low_risk']}")
    
    # Create enhanced data manifest
    print("\n📊 Creating enhanced data manifest...")
    manifest = create_enhanced_data_manifest(data_dir)
    
    print(f"\n📈 Data Processing Summary:")
    print(f"  Total samples: {manifest['total_samples']}")
    print(f"  Datasets: {len(manifest['datasets'])}")
    
    if manifest['crisis_detection']['total_samples'] > 0:
        print(f"\n🚨 Crisis Detection Data:")
        print(f"  Total samples: {manifest['crisis_detection']['total_samples']}")
        print(f"  High risk: {manifest['crisis_detection']['high_risk']}")
        print(f"  Medium risk: {manifest['crisis_detection']['medium_risk']}")
        print(f"  Low risk: {manifest['crisis_detection']['low_risk']}")
    
    print(f"\n✅ Data processing completed!")
    print("📝 Next steps:")
    print("  1. Use processed data for training crisis detection models")
    print("  2. Run model training with real data")
    print("  3. Evaluate model performance")

if __name__ == "__main__":
    main()
