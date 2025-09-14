#!/usr/bin/env python3
"""
Run AI Safety Models POC Demo.

This script provides a comprehensive demonstration of the AI Safety Models system.
"""

import sys
import os
import argparse
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from safety_system.safety_manager import SafetyManager
from models.content_filter import AgeGroup


def run_comprehensive_demo():
    """Run a comprehensive demonstration of all safety models."""
    print("🚀 AI Safety Models POC - Comprehensive Demo")
    print("=" * 50)
    
    # Initialize safety manager
    safety_manager = SafetyManager()
    
    # Demo test cases
    test_cases = [
        {
            'name': 'Safe Content',
            'text': 'Hello, how are you today? I hope you have a great day!',
            'age_group': AgeGroup.ADULT,
            'expected_risk': 'minimal'
        },
        {
            'name': 'Mild Profanity',
            'text': 'This is so damn frustrating!',
            'age_group': AgeGroup.ADULT,
            'expected_risk': 'low'
        },
        {
            'name': 'Escalating Anger',
            'text': 'I hate this! I hate everything! You are so stupid!',
            'age_group': AgeGroup.ADULT,
            'expected_risk': 'medium'
        },
        {
            'name': 'Crisis Indicators',
            'text': 'I want to kill myself. I can\'t go on anymore.',
            'age_group': AgeGroup.ADULT,
            'expected_risk': 'critical'
        },
        {
            'name': 'Adult Content for Teen',
            'text': 'This movie has explicit sexual content and violence.',
            'age_group': AgeGroup.TEEN,
            'expected_risk': 'high'
        },
        {
            'name': 'Educational Content for Child',
            'text': 'Let\'s learn about science and math together!',
            'age_group': AgeGroup.CHILD,
            'expected_risk': 'minimal'
        }
    ]
    
    print(f"Running {len(test_cases)} comprehensive test cases...\n")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🧪 Test {i}: {test_case['name']}")
        print(f"📝 Text: '{test_case['text']}'")
        print(f"👤 Age Group: {test_case['age_group'].value}")
        
        try:
            # Perform analysis
            analysis_result = safety_manager.analyze(
                text=test_case['text'],
                user_id=f"demo_user_{i}",
                session_id=f"demo_session_{i}",
                age_group=test_case['age_group']
            )
            
            overall_risk = analysis_result['overall_assessment']['overall_risk']
            intervention_level = analysis_result['overall_assessment']['intervention_level']
            
            print(f"📊 Overall Risk: {overall_risk.upper()}")
            print(f"🔧 Intervention Level: {intervention_level}")
            
            # Check if result matches expectation
            if overall_risk == test_case['expected_risk']:
                print("✅ PASS - Risk assessment matches expectation")
                status = "PASS"
            else:
                print(f"⚠️  PARTIAL - Expected: {test_case['expected_risk']}, Got: {overall_risk}")
                status = "PARTIAL"
            
            # Show model breakdown
            print("📋 Model Results:")
            for model_name, model_result in analysis_result['models'].items():
                result = model_result['result']
                risk_level = model_result['risk_level']
                print(f"  • {model_name.replace('_', ' ').title()}: {result.label} ({risk_level})")
            
            # Show interventions if any
            if analysis_result['intervention_recommendations']:
                print("⚠️  Interventions:")
                for rec in analysis_result['intervention_recommendations'][:2]:  # Show first 2
                    print(f"  - [{rec['priority'].upper()}] {rec['action']}")
            
            results.append({
                'test_name': test_case['name'],
                'status': status,
                'expected': test_case['expected_risk'],
                'actual': overall_risk,
                'intervention_level': intervention_level
            })
            
        except Exception as e:
            print(f"❌ FAIL - Error: {e}")
            results.append({
                'test_name': test_case['name'],
                'status': 'FAIL',
                'error': str(e)
            })
        
        print("-" * 50)
    
    # Summary
    print("\n📈 Demo Summary:")
    print("=" * 30)
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    partial = sum(1 for r in results if r['status'] == 'PARTIAL')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Partial: {partial}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(results)}")
    
    if passed > 0:
        print(f"\n🎉 Demo completed successfully! {passed} tests passed.")
    else:
        print("\n⚠️  Demo completed with some issues. Check the results above.")
    
    return results


def run_interactive_demo():
    """Run an interactive demo session."""
    print("🎮 AI Safety Models POC - Interactive Demo")
    print("=" * 45)
    
    safety_manager = SafetyManager()
    
    print("Enter text to analyze (type 'quit' to exit):")
    
    user_id = "interactive_user"
    session_id = "interactive_session"
    age_group = AgeGroup.ADULT
    
    while True:
        try:
            text = input("\n📝 Text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text:
                print("❌ Please enter some text.")
                continue
            
            print("⏳ Analyzing...")
            
            result = safety_manager.analyze(
                text=text,
                user_id=user_id,
                session_id=session_id,
                age_group=age_group
            )
            
            overall = result['overall_assessment']
            print(f"\n📊 Results:")
            print(f"  Risk Level: {overall['overall_risk'].upper()}")
            print(f"  Intervention: {overall['intervention_level']}")
            print(f"  Max Score: {overall['max_score']:.3f}")
            
            # Show model results
            for model_name, model_result in result['models'].items():
                result_data = model_result['result']
                print(f"  {model_name.replace('_', ' ').title()}: {result_data.label} ({result_data.score:.3f})")
            
            # Show interventions
            if result['intervention_recommendations']:
                print("\n⚠️  Recommendations:")
                for rec in result['intervention_recommendations']:
                    print(f"  - {rec['action']}")
            
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point for the POC demo."""
    parser = argparse.ArgumentParser(description='AI Safety Models POC Demo')
    parser.add_argument('--mode', choices=['comprehensive', 'interactive'], 
                       default='comprehensive',
                       help='Demo mode to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("🤖 AI Safety Models Proof of Concept")
    print("=====================================")
    
    try:
        if args.mode == 'comprehensive':
            run_comprehensive_demo()
        elif args.mode == 'interactive':
            run_interactive_demo()
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()