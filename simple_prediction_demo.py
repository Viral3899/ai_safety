#!/usr/bin/env python3
"""
Simple Prediction Demo for AI Safety Models.

This script provides a simplified interface to test the AI safety models
without complex dependencies.
"""

import sys
import os
import json
import re
from typing import Dict, Any, List
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class SimpleSafetyAnalyzer:
    """Simple safety analyzer using rule-based patterns."""
    
    def __init__(self):
        """Initialize the analyzer with safety patterns."""
        self.crisis_patterns = {
            'immediate_threat': [
                r'\b(kill myself|suicide|end it all|not worth living|end my life)\b',
                r'\b(harm myself|hurt myself|cut myself|self harm)\b',
                r'\b(overdose|take pills|poison|overdose on)\b',
                r'\b(jump off|jump from|fall from|jump in front of)\b',
                r'\b(final goodbye|last message|never see me again|goodbye forever)\b'
            ],
            'severe_distress': [
                r'\b(can\'t go on|can\'t take it|giving up|give up)\b',
                r'\b(hopeless|worthless|useless|pointless)\b',
                r'\b(nobody cares|no one loves me|alone|lonely|isolated)\b',
                r'\b(burden|better off without me|everyone hates me)\b',
                r'\b(want to die|wish I was dead|wish I could die)\b'
            ],
            'emotional_crisis': [
                r'\b(breakdown|falling apart|losing it|losing my mind)\b',
                r'\b(can\'t cope|overwhelmed|drowning|suffocating)\b',
                r'\b(panic attack|anxiety attack|panic|anxiety)\b',
                r'\b(crisis|emergency|help me|need help)\b',
                r'\b(desperate|urgent|immediate help|can\'t handle)\b'
            ]
        }
        
        self.abuse_patterns = {
            'mild_abuse': [
                r'\b(damn|hell|stupid|idiot|moron)\b',
                r'\b(crap|bullshit|sucks|annoying)\b',
                r'\b(frustrating|ridiculous|terrible)\b'
            ],
            'severe_abuse': [
                r'\b(hate you|despise|worthless|piece of shit)\b',
                r'\b(fucking|asshole|loser|garbage)\b',
                r'\b(want to hurt|kill you|destroy you)\b'
            ]
        }
        
        self.content_patterns = {
            'adult_content': [
                r'\b(explicit|sexual|violence|gore)\b',
                r'\b(drugs|alcohol|pills|medication)\b',
                r'\b(profanity|mature|adult)\b'
            ]
        }
        
        self.escalation_patterns = [
            r'\b(frustrated|angry|mad|furious)\b',
            r'\b(can\'t take this|about to explode|losing my temper)\b',
            r'\b(driving me crazy|about to lose it)\b'
        ]
    
    def analyze_crisis(self, text: str) -> Dict[str, Any]:
        """Analyze text for crisis indicators."""
        text_lower = text.lower()
        scores = {}
        
        for category, patterns in self.crisis_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                count += len(matches)
            scores[category] = count
        
        # Calculate overall crisis score
        immediate_score = min(scores['immediate_threat'] * 0.8, 1.0)
        if scores['immediate_threat'] > 0:
            immediate_score = 1.0  # Any immediate threat = critical
        
        severe_score = min(scores['severe_distress'] * 0.4, 0.9)
        emotional_score = min(scores['emotional_crisis'] * 0.3, 0.7)
        
        overall_score = max(immediate_score, severe_score, emotional_score)
        
        if overall_score > 0.8:
            level = "critical"
            label = "critical_crisis"
        elif overall_score > 0.6:
            level = "high"
            label = "severe_crisis"
        elif overall_score > 0.4:
            level = "medium"
            label = "moderate_crisis"
        elif overall_score > 0.2:
            level = "low"
            label = "mild_concern"
        else:
            level = "minimal"
            label = "safe"
        
        return {
            'label': label,
            'score': overall_score,
            'level': level,
            'scores': scores,
            'confidence': overall_score
        }
    
    def analyze_abuse(self, text: str) -> Dict[str, Any]:
        """Analyze text for abuse indicators."""
        text_lower = text.lower()
        scores = {}
        
        for category, patterns in self.abuse_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                count += len(matches)
            scores[category] = count
        
        # Calculate overall abuse score
        mild_score = min(scores['mild_abuse'] * 0.3, 0.6)
        severe_score = min(scores['severe_abuse'] * 0.8, 1.0)
        
        overall_score = max(mild_score, severe_score)
        
        if overall_score > 0.8:
            level = "high"
            label = "severe_abuse"
        elif overall_score > 0.5:
            level = "medium"
            label = "moderate_abuse"
        elif overall_score > 0.2:
            level = "low"
            label = "mild_abuse"
        else:
            level = "minimal"
            label = "safe"
        
        return {
            'label': label,
            'score': overall_score,
            'level': level,
            'scores': scores,
            'confidence': overall_score
        }
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze text for inappropriate content."""
        text_lower = text.lower()
        scores = {}
        
        for category, patterns in self.content_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                count += len(matches)
            scores[category] = count
        
        # Calculate overall content score
        adult_score = min(scores['adult_content'] * 0.5, 1.0)
        
        if adult_score > 0.7:
            level = "high"
            label = "inappropriate_content"
        elif adult_score > 0.4:
            level = "medium"
            label = "mature_content"
        elif adult_score > 0.1:
            level = "low"
            label = "mild_content"
        else:
            level = "minimal"
            label = "safe"
        
        return {
            'label': label,
            'score': adult_score,
            'level': level,
            'scores': scores,
            'confidence': adult_score
        }
    
    def analyze_escalation(self, text: str) -> Dict[str, Any]:
        """Analyze text for escalation patterns."""
        text_lower = text.lower()
        count = 0
        
        for pattern in self.escalation_patterns:
            matches = re.findall(pattern, text_lower)
            count += len(matches)
        
        # Calculate escalation score
        escalation_score = min(count * 0.3, 1.0)
        
        if escalation_score > 0.7:
            level = "high"
            label = "escalating"
        elif escalation_score > 0.4:
            level = "medium"
            label = "moderate_escalation"
        elif escalation_score > 0.1:
            level = "low"
            label = "mild_escalation"
        else:
            level = "minimal"
            label = "stable"
        
        return {
            'label': label,
            'score': escalation_score,
            'level': level,
            'count': count,
            'confidence': escalation_score
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive safety analysis."""
        print(f"🔍 Analyzing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Run all analyses
        crisis_result = self.analyze_crisis(text)
        abuse_result = self.analyze_abuse(text)
        content_result = self.analyze_content(text)
        escalation_result = self.analyze_escalation(text)
        
        # Calculate overall risk
        scores = [
            crisis_result['score'],
            abuse_result['score'],
            content_result['score'],
            escalation_result['score']
        ]
        
        overall_score = max(scores)
        
        if overall_score > 0.8:
            overall_risk = "critical"
        elif overall_score > 0.6:
            overall_risk = "high"
        elif overall_score > 0.4:
            overall_risk = "medium"
        elif overall_score > 0.2:
            overall_risk = "low"
        else:
            overall_risk = "minimal"
        
        return {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'overall_risk': overall_risk,
            'overall_score': overall_score,
            'models': {
                'crisis_detector': crisis_result,
                'abuse_detector': abuse_result,
                'content_filter': content_result,
                'escalation_detector': escalation_result
            }
        }

def print_analysis_results(results: Dict[str, Any]):
    """Print analysis results in a formatted way."""
    print("\n" + "="*60)
    print("📊 AI SAFETY ANALYSIS RESULTS")
    print("="*60)
    
    # Overall assessment
    overall_risk = results['overall_risk']
    overall_score = results['overall_score']
    
    risk_emoji = {
        'minimal': '🟢',
        'low': '🟡', 
        'medium': '🟠',
        'high': '🔴',
        'critical': '🚨'
    }
    
    print(f"{risk_emoji.get(overall_risk, '❓')} Overall Risk: {overall_risk.upper()}")
    print(f"📊 Overall Score: {overall_score:.3f}")
    
    # Individual model results
    print(f"\n📋 Model Results:")
    print("-" * 40)
    
    for model_name, model_result in results['models'].items():
        print(f"\n🔧 {model_name.replace('_', ' ').title()}:")
        print(f"   Label: {model_result['label']}")
        print(f"   Score: {model_result['score']:.3f}")
        print(f"   Level: {model_result['level']}")
        print(f"   Confidence: {model_result['confidence']:.3f}")
    
    # Recommendations
    print(f"\n⚠️  Recommendations:")
    print("-" * 20)
    
    if overall_risk == "critical":
        print("🚨 IMMEDIATE INTERVENTION REQUIRED")
        print("   - Contact crisis intervention team")
        print("   - Provide emergency resources")
        print("   - Monitor closely")
    elif overall_risk == "high":
        print("🔴 HIGH PRIORITY MONITORING")
        print("   - Escalate to human moderator")
        print("   - Provide support resources")
        print("   - Consider intervention")
    elif overall_risk == "medium":
        print("🟠 MEDIUM PRIORITY")
        print("   - Monitor conversation")
        print("   - Provide gentle guidance")
        print("   - Flag for review")
    elif overall_risk == "low":
        print("🟡 LOW PRIORITY")
        print("   - Continue monitoring")
        print("   - Provide positive reinforcement")
    else:
        print("🟢 SAFE")
        print("   - No intervention needed")
        print("   - Continue normal interaction")

def run_test_cases():
    """Run predefined test cases."""
    print("\n🧪 RUNNING TEST CASES")
    print("="*30)
    
    analyzer = SimpleSafetyAnalyzer()
    
    test_cases = [
        {
            'name': 'Safe Content',
            'text': 'Hello, how are you today? I hope you have a great day!',
            'expected_risk': 'minimal'
        },
        {
            'name': 'Mild Profanity',
            'text': 'This is so damn frustrating!',
            'expected_risk': 'low'
        },
        {
            'name': 'Escalating Anger',
            'text': 'I hate this! I hate everything! You are so stupid!',
            'expected_risk': 'high'
        },
        {
            'name': 'Crisis Indicators',
            'text': 'I want to kill myself. I can\'t go on anymore.',
            'expected_risk': 'critical'
        },
        {
            'name': 'Adult Content',
            'text': 'This movie has explicit sexual content and violence.',
            'expected_risk': 'medium'
        }
    ]
    
    print(f"Running {len(test_cases)} test cases...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Text: '{test_case['text']}'")
        
        results = analyzer.analyze(test_case['text'])
        overall_risk = results['overall_risk']
        expected_risk = test_case['expected_risk']
        
        if overall_risk == expected_risk:
            print(f"✅ PASS - Risk: {overall_risk}")
        else:
            print(f"⚠️  PARTIAL - Expected: {expected_risk}, Got: {overall_risk}")
        
        print()

def interactive_mode():
    """Run interactive analysis mode."""
    print("\n🔍 INTERACTIVE SAFETY ANALYSIS")
    print("="*40)
    print("Enter text to analyze (type 'quit' to exit, 'test' for test cases)")
    
    analyzer = SimpleSafetyAnalyzer()
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            
            if text.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif text.lower() == 'test':
                run_test_cases()
                continue
            elif not text:
                print("❌ No text provided.")
                continue
            
            results = analyzer.analyze(text)
            print_analysis_results(results)
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main entry point."""
    print("🤖 AI Safety Models - Simple Prediction Demo")
    print("="*50)
    print("This demo provides rule-based safety analysis for:")
    print("• Crisis Detection")
    print("• Abuse Language Detection") 
    print("• Content Filtering")
    print("• Escalation Detection")
    print("="*50)
    
    while True:
        print("\n📋 Options:")
        print("1. Interactive analysis")
        print("2. Run test cases")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            interactive_mode()
        elif choice == '2':
            run_test_cases()
        elif choice == '3':
            print("👋 Thank you for using AI Safety Models!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-3.")

if __name__ == "__main__":
    main()
