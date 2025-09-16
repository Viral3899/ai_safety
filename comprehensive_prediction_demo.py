#!/usr/bin/env python3
"""
Comprehensive Prediction Demo for AI Safety Models.

This script demonstrates the AI safety models with detailed analysis,
visualization, and comprehensive testing capabilities.
"""

import sys
import os
import json
import re
from typing import Dict, Any, List, Tuple
from datetime import datetime
import random

class ComprehensiveSafetyAnalyzer:
    """Comprehensive safety analyzer with advanced pattern matching."""
    
    def __init__(self):
        """Initialize the analyzer with comprehensive safety patterns."""
        self.crisis_patterns = {
            'immediate_threat': [
                r'\b(kill myself|suicide|end it all|not worth living|end my life)\b',
                r'\b(harm myself|hurt myself|cut myself|self harm)\b',
                r'\b(overdose|take pills|poison|overdose on)\b',
                r'\b(jump off|jump from|fall from|jump in front of)\b',
                r'\b(final goodbye|last message|never see me again|goodbye forever)\b',
                r'\b(hanging|hang myself|strangle|choke)\b',
                r'\b(bleed out|cut wrists|slit wrists)\b',
                r'\b(drown|drowning|water|drown myself)\b'
            ],
            'severe_distress': [
                r'\b(can\'t go on|can\'t take it|giving up|give up)\b',
                r'\b(hopeless|worthless|useless|pointless)\b',
                r'\b(nobody cares|no one loves me|alone|lonely|isolated)\b',
                r'\b(burden|better off without me|everyone hates me)\b',
                r'\b(want to die|wish I was dead|wish I could die)\b',
                r'\b(no point|what\'s the point|nothing matters)\b',
                r'\b(empty|numb|dead inside|feel nothing)\b',
                r'\b(can\'t feel|can\'t feel anything|emotionally dead)\b'
            ],
            'emotional_crisis': [
                r'\b(breakdown|falling apart|losing it|losing my mind)\b',
                r'\b(can\'t cope|overwhelmed|drowning|suffocating)\b',
                r'\b(panic attack|anxiety attack|panic|anxiety)\b',
                r'\b(crisis|emergency|help me|need help)\b',
                r'\b(desperate|urgent|immediate help|can\'t handle)\b',
                r'\b(mental breakdown|nervous breakdown)\b',
                r'\b(psychotic|hallucinating|hearing voices)\b',
                r'\b(manic|mania|bipolar|depression)\b'
            ],
            'substance_crisis': [
                r'\b(drunk|drinking|alcohol|booze)\b',
                r'\b(drugs|high|stoned|overdose)\b',
                r'\b(pills|medication|prescription)\b',
                r'\b(addiction|addicted|withdrawal)\b'
            ],
            'relationship_crisis': [
                r'\b(breakup|divorce|separated|abandoned)\b',
                r'\b(abuse|abused|violence|violent)\b',
                r'\b(bullied|harassed|threatened)\b',
                r'\b(rejected|betrayed|lied to)\b'
            ]
        }
        
        self.abuse_patterns = {
            'mild_abuse': [
                r'\b(damn|hell|stupid|idiot|moron)\b',
                r'\b(crap|bullshit|sucks|annoying)\b',
                r'\b(frustrating|ridiculous|terrible|awful)\b',
                r'\b(hate this|hate it|disgusting)\b'
            ],
            'severe_abuse': [
                r'\b(hate you|despise|worthless|piece of shit)\b',
                r'\b(fucking|asshole|loser|garbage)\b',
                r'\b(want to hurt|kill you|destroy you)\b',
                r'\b(you\'re a|you are a)\s+(idiot|moron|loser|failure)\b'
            ],
            'threats': [
                r'\b(i\'ll hurt|i will hurt|going to hurt)\b',
                r'\b(i\'ll kill|i will kill|going to kill)\b',
                r'\b(i\'ll destroy|i will destroy|going to destroy)\b',
                r'\b(you\'ll pay|you will pay|going to pay)\b'
            ]
        }
        
        self.content_patterns = {
            'explicit_sexual': [
                r'\b(sex|sexual|porn|pornography|nude|naked)\b',
                r'\b(orgasm|masturbat|fuck|fucking|fucked)\b',
                r'\b(penis|vagina|breast|ass|butt)\b'
            ],
            'violence': [
                r'\b(kill|murder|death|die|dying|dead)\b',
                r'\b(violence|violent|fight|fighting|beat|beating)\b',
                r'\b(blood|gore|gory|torture|torturing)\b',
                r'\b(weapon|gun|knife|bomb|explosive)\b'
            ],
            'drugs': [
                r'\b(drugs|drug|marijuana|cannabis|weed)\b',
                r'\b(cocaine|crack|heroin|meth|methamphetamine)\b',
                r'\b(ecstasy|lsd|acid|mushrooms|shrooms)\b',
                r'\b(overdose|overdosing|high|stoned|wasted)\b'
            ],
            'profanity': [
                r'\b(fuck|shit|bitch|asshole|damn|hell)\b',
                r'\b(cunt|pussy|dick|cock|bastard)\b',
                r'\b(motherfucker|fucking|fucked|fucker)\b'
            ]
        }
        
        self.escalation_patterns = [
            r'\b(frustrated|angry|mad|furious|pissed)\b',
            r'\b(can\'t take this|about to explode|losing my temper)\b',
            r'\b(driving me crazy|about to lose it|losing it)\b',
            r'\b(getting worse|escalating|out of control)\b',
            r'\b(had enough|can\'t handle|too much)\b'
        ]
        
        # Support-seeking patterns (positive indicators)
        self.support_patterns = [
            r'\b(help|support|therapy|counseling|counselor)\b',
            r'\b(talk to someone|reach out|call|contact)\b',
            r'\b(hotline|crisis line|emergency|911)\b',
            r'\b(doctor|psychiatrist|therapist|mental health)\b',
            r'\b(friend|family|parent|someone to talk to)\b',
            r'\b(need help|asking for help|please help)\b'
        ]
        
        # Protective factors (reduce risk)
        self.protective_patterns = [
            r'\b(future|tomorrow|next week|plans)\b',
            r'\b(family|children|kids|loved ones)\b',
            r'\b(religion|faith|god|prayer)\b',
            r'\b(hopeful|hope|better|improving)\b',
            r'\b(medication|treatment|therapy|getting help)\b'
        ]
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive features from text."""
        text_lower = text.lower()
        features = {}
        
        # Crisis features
        for category, patterns in self.crisis_patterns.items():
            count = 0
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower)
                count += len(found)
                matches.extend(found)
            features[f'crisis_{category}'] = count
            features[f'crisis_{category}_matches'] = matches
        
        # Abuse features
        for category, patterns in self.abuse_patterns.items():
            count = 0
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower)
                count += len(found)
                matches.extend(found)
            features[f'abuse_{category}'] = count
            features[f'abuse_{category}_matches'] = matches
        
        # Content features
        for category, patterns in self.content_patterns.items():
            count = 0
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower)
                count += len(found)
                matches.extend(found)
            features[f'content_{category}'] = count
            features[f'content_{category}_matches'] = matches
        
        # Escalation features
        escalation_count = 0
        escalation_matches = []
        for pattern in self.escalation_patterns:
            found = re.findall(pattern, text_lower)
            escalation_count += len(found)
            escalation_matches.extend(found)
        features['escalation'] = escalation_count
        features['escalation_matches'] = escalation_matches
        
        # Support-seeking features
        support_count = 0
        support_matches = []
        for pattern in self.support_patterns:
            found = re.findall(pattern, text_lower)
            support_count += len(found)
            support_matches.extend(found)
        features['support_seeking'] = support_count
        features['support_matches'] = support_matches
        
        # Protective factors
        protective_count = 0
        protective_matches = []
        for pattern in self.protective_patterns:
            found = re.findall(pattern, text_lower)
            protective_count += len(found)
            protective_matches.extend(found)
        features['protective'] = protective_count
        features['protective_matches'] = protective_matches
        
        # Text characteristics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))
        features['ellipsis_count'] = text.count('...')
        
        return features
    
    def analyze_crisis(self, text: str) -> Dict[str, Any]:
        """Analyze text for crisis indicators with detailed scoring."""
        features = self.extract_features(text)
        
        # Calculate individual crisis scores
        immediate_score = min(features['crisis_immediate_threat'] * 0.8, 1.0)
        if features['crisis_immediate_threat'] > 0:
            immediate_score = 1.0  # Any immediate threat = critical
        
        severe_score = min(features['crisis_severe_distress'] * 0.4, 0.9)
        emotional_score = min(features['crisis_emotional_crisis'] * 0.3, 0.7)
        substance_score = min(features['crisis_substance_crisis'] * 0.3, 0.6)
        relationship_score = min(features['crisis_relationship_crisis'] * 0.2, 0.5)
        
        # Support-seeking bonus (reduces risk)
        support_bonus = min(features['support_seeking'] * 0.1, 0.3)
        
        # Protective factors bonus (reduces risk)
        protective_bonus = min(features['protective'] * 0.05, 0.2)
        
        # Calculate individual scores
        scores = {
            'immediate_threat': immediate_score,
            'severe_distress': max(0, severe_score - support_bonus - protective_bonus),
            'emotional_crisis': max(0, emotional_score - support_bonus - protective_bonus),
            'substance_crisis': max(0, substance_score - support_bonus),
            'relationship_crisis': max(0, relationship_score - support_bonus)
        }
        
        # Overall crisis score
        overall_score = max(scores.values())
        
        # Determine level and label
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
            'confidence': overall_score,
            'features': features,
            'support_seeking': support_bonus,
            'protective_factors': protective_bonus
        }
    
    def analyze_abuse(self, text: str) -> Dict[str, Any]:
        """Analyze text for abuse indicators with detailed scoring."""
        features = self.extract_features(text)
        
        # Calculate abuse scores
        mild_score = min(features['abuse_mild_abuse'] * 0.3, 0.6)
        severe_score = min(features['abuse_severe_abuse'] * 0.8, 1.0)
        threat_score = min(features['abuse_threats'] * 1.0, 1.0)
        
        # Overall abuse score
        overall_score = max(mild_score, severe_score, threat_score)
        
        # Determine level and label
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
            'scores': {
                'mild_abuse': mild_score,
                'severe_abuse': severe_score,
                'threats': threat_score
            },
            'confidence': overall_score,
            'features': features
        }
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze text for inappropriate content with detailed scoring."""
        features = self.extract_features(text)
        
        # Calculate content scores
        sexual_score = min(features['content_explicit_sexual'] * 0.8, 1.0)
        violence_score = min(features['content_violence'] * 0.7, 1.0)
        drugs_score = min(features['content_drugs'] * 0.6, 1.0)
        profanity_score = min(features['content_profanity'] * 0.4, 1.0)
        
        # Overall content score
        overall_score = max(sexual_score, violence_score, drugs_score, profanity_score)
        
        # Determine level and label
        if overall_score > 0.8:
            level = "high"
            label = "inappropriate_content"
        elif overall_score > 0.5:
            level = "medium"
            label = "mature_content"
        elif overall_score > 0.2:
            level = "low"
            label = "mild_content"
        else:
            level = "minimal"
            label = "safe"
        
        return {
            'label': label,
            'score': overall_score,
            'level': level,
            'scores': {
                'explicit_sexual': sexual_score,
                'violence': violence_score,
                'drugs': drugs_score,
                'profanity': profanity_score
            },
            'confidence': overall_score,
            'features': features
        }
    
    def analyze_escalation(self, text: str) -> Dict[str, Any]:
        """Analyze text for escalation patterns with detailed scoring."""
        features = self.extract_features(text)
        
        # Calculate escalation score
        escalation_score = min(features['escalation'] * 0.3, 1.0)
        
        # Additional intensity factors
        intensity_factors = 0
        if features['repeated_chars'] > 2:
            intensity_factors += 0.1
        if features['ellipsis_count'] > 1:
            intensity_factors += 0.05
        if features['exclamation_count'] > 3:
            intensity_factors += 0.1
        
        overall_score = min(escalation_score + intensity_factors, 1.0)
        
        # Determine level and label
        if overall_score > 0.7:
            level = "high"
            label = "escalating"
        elif overall_score > 0.4:
            level = "medium"
            label = "moderate_escalation"
        elif overall_score > 0.1:
            level = "low"
            label = "mild_escalation"
        else:
            level = "minimal"
            label = "stable"
        
        return {
            'label': label,
            'score': overall_score,
            'level': level,
            'confidence': overall_score,
            'features': features,
            'intensity_factors': intensity_factors
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

def print_detailed_results(results: Dict[str, Any]):
    """Print detailed analysis results."""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE AI SAFETY ANALYSIS RESULTS")
    print("="*80)
    
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
    
    print(f"{risk_emoji.get(overall_risk, '❓')} OVERALL RISK: {overall_risk.upper()}")
    print(f"📊 Overall Score: {overall_score:.3f}")
    print(f"⏰ Analysis Time: {results['timestamp']}")
    
    # Individual model results
    print(f"\n📋 DETAILED MODEL RESULTS:")
    print("-" * 60)
    
    for model_name, model_result in results['models'].items():
        print(f"\n🔧 {model_name.replace('_', ' ').title()}:")
        print(f"   Label: {model_result['label']}")
        print(f"   Score: {model_result['score']:.3f}")
        print(f"   Level: {model_result['level']}")
        print(f"   Confidence: {model_result['confidence']:.3f}")
        
        # Show detailed scores if available
        if 'scores' in model_result:
            print(f"   Detailed Scores:")
            for score_name, score_value in model_result['scores'].items():
                print(f"     - {score_name}: {score_value:.3f}")
        
        # Show key features
        if 'features' in model_result:
            features = model_result['features']
            key_features = []
            for key, value in features.items():
                if isinstance(value, list) and value:
                    key_features.append(f"{key}: {value}")
                elif isinstance(value, (int, float)) and value > 0:
                    key_features.append(f"{key}: {value}")
            
            if key_features:
                print(f"   Key Features: {', '.join(key_features[:5])}")
    
    # Recommendations
    print(f"\n⚠️  INTERVENTION RECOMMENDATIONS:")
    print("-" * 40)
    
    if overall_risk == "critical":
        print("🚨 IMMEDIATE INTERVENTION REQUIRED")
        print("   - Contact crisis intervention team immediately")
        print("   - Provide emergency resources and hotlines")
        print("   - Monitor conversation closely")
        print("   - Consider escalating to human moderator")
    elif overall_risk == "high":
        print("🔴 HIGH PRIORITY MONITORING")
        print("   - Escalate to human moderator")
        print("   - Provide support resources")
        print("   - Consider intervention")
        print("   - Flag for immediate review")
    elif overall_risk == "medium":
        print("🟠 MEDIUM PRIORITY")
        print("   - Monitor conversation closely")
        print("   - Provide gentle guidance")
        print("   - Flag for review")
        print("   - Consider support resources")
    elif overall_risk == "low":
        print("🟡 LOW PRIORITY")
        print("   - Continue monitoring")
        print("   - Provide positive reinforcement")
        print("   - Flag for periodic review")
    else:
        print("🟢 SAFE")
        print("   - No intervention needed")
        print("   - Continue normal interaction")
        print("   - Maintain positive engagement")

def run_comprehensive_tests():
    """Run comprehensive test cases."""
    print("\n🧪 COMPREHENSIVE TEST SUITE")
    print("="*40)
    
    analyzer = ComprehensiveSafetyAnalyzer()
    
    test_cases = [
        {
            'name': 'Safe Content',
            'text': 'Hello, how are you today? I hope you have a great day!',
            'expected_risk': 'minimal',
            'description': 'Normal, positive conversation'
        },
        {
            'name': 'Mild Profanity',
            'text': 'This is so damn frustrating! I can\'t figure this out.',
            'expected_risk': 'low',
            'description': 'Mild profanity and frustration'
        },
        {
            'name': 'Escalating Anger',
            'text': 'I hate this! I hate everything! You are so stupid! This is driving me crazy!',
            'expected_risk': 'high',
            'description': 'Escalating anger and frustration'
        },
        {
            'name': 'Crisis Indicators',
            'text': 'I want to kill myself. I can\'t go on anymore. Nobody cares about me.',
            'expected_risk': 'critical',
            'description': 'Clear crisis indicators and suicidal ideation'
        },
        {
            'name': 'Adult Content',
            'text': 'This movie has explicit sexual content and graphic violence.',
            'expected_risk': 'medium',
            'description': 'Adult content references'
        },
        {
            'name': 'Support Seeking',
            'text': 'I need help. Can someone talk to me? I\'m feeling really down.',
            'expected_risk': 'low',
            'description': 'Support-seeking behavior (positive indicator)'
        },
        {
            'name': 'Mixed Signals',
            'text': 'I hate my life but I have my family and I\'m getting therapy.',
            'expected_risk': 'medium',
            'description': 'Negative thoughts with protective factors'
        }
    ]
    
    print(f"Running {len(test_cases)} comprehensive test cases...\n")
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Text: '{test_case['text']}'")
        
        analysis = analyzer.analyze(test_case['text'])
        overall_risk = analysis['overall_risk']
        expected_risk = test_case['expected_risk']
        
        # Determine test result
        if overall_risk == expected_risk:
            result = "✅ PASS"
        elif abs(ord(overall_risk[0]) - ord(expected_risk[0])) <= 1:  # Adjacent risk levels
            result = "⚠️  PARTIAL"
        else:
            result = "❌ FAIL"
        
        print(f"{result} - Expected: {expected_risk}, Got: {overall_risk}")
        
        # Store results
        results.append({
            'test': test_case['name'],
            'expected': expected_risk,
            'actual': overall_risk,
            'result': result,
            'score': analysis['overall_score']
        })
        
        print("-" * 60)
    
    # Summary
    passed = sum(1 for r in results if "PASS" in r['result'])
    partial = sum(1 for r in results if "PARTIAL" in r['result'])
    failed = sum(1 for r in results if "FAIL" in r['result'])
    
    print(f"\n📊 TEST SUMMARY:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ⚠️  Partial: {partial}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {((passed + partial * 0.5) / len(results) * 100):.1f}%")
    
    return results

def interactive_analysis():
    """Run interactive analysis mode."""
    print("\n🔍 INTERACTIVE SAFETY ANALYSIS")
    print("="*40)
    print("Enter text to analyze (type 'quit' to exit, 'test' for test cases)")
    
    analyzer = ComprehensiveSafetyAnalyzer()
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            
            if text.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif text.lower() == 'test':
                run_comprehensive_tests()
                continue
            elif not text:
                print("❌ No text provided.")
                continue
            
            results = analyzer.analyze(text)
            print_detailed_results(results)
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main entry point."""
    print("🤖 AI Safety Models - Comprehensive Prediction Demo")
    print("="*60)
    print("This demo provides comprehensive safety analysis including:")
    print("• Advanced Crisis Detection with multiple categories")
    print("• Detailed Abuse Language Detection")
    print("• Comprehensive Content Filtering")
    print("• Escalation Pattern Recognition")
    print("• Support-seeking and Protective Factor Analysis")
    print("="*60)
    
    while True:
        print("\n📋 Options:")
        print("1. Interactive analysis")
        print("2. Run comprehensive test cases")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            interactive_analysis()
        elif choice == '2':
            run_comprehensive_tests()
        elif choice == '3':
            print("👋 Thank you for using AI Safety Models!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-3.")

if __name__ == "__main__":
    main()
