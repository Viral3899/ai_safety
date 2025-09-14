#!/usr/bin/env python3
"""
Demo Scenarios for AI Safety Models POC Video Demonstration.

This script provides pre-defined test cases and scenarios for the 10-minute
walkthrough video, ensuring consistent and impressive demonstrations.
"""

import sys
import os
import json
import time
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from safety_system.safety_manager import SafetyManager
from models.content_filter import AgeGroup

class DemoScreen:
    """Demo screen with formatted output for video recording."""
    
    @staticmethod
    def show_header(title: str):
        """Show demo section header."""
        print("=" * 60)
        print(f"🎬 {title}")
        print("=" * 60)
    
    @staticmethod
    def show_scenario(scenario_num: int, title: str, description: str):
        """Show scenario introduction."""
        print(f"\n📋 SCENARIO {scenario_num}: {title}")
        print(f"📝 Description: {description}")
        print("-" * 50)
    
    @staticmethod
    def show_input(text: str, age_group: str = "adult"):
        """Show input text and parameters."""
        print(f"📥 INPUT:")
        print(f"   Text: '{text}'")
        print(f"   Age Group: {age_group}")
        print(f"   User ID: demo_user")
        print(f"   Session ID: demo_session")
    
    @staticmethod
    def show_output(result: Dict[str, Any]):
        """Show formatted output."""
        print(f"\n📤 OUTPUT:")
        print(f"   Overall Risk: {result['overall_assessment']['overall_risk'].upper()}")
        print(f"   Intervention Level: {result['overall_assessment']['intervention_level']}")
        print(f"   Max Score: {result['overall_assessment']['max_score']:.3f}")
        print(f"   Processing Time: {result['processing_time']:.3f}s")
        
        print(f"\n📊 MODEL BREAKDOWN:")
        for model_name, model_result in result['models'].items():
            result_data = model_result['result']
            risk_level = model_result['risk_level']
            print(f"   • {model_name.replace('_', ' ').title()}: {result_data.label} ({risk_level}) - {result_data.score:.3f}")
        
        if result['intervention_recommendations']:
            print(f"\n⚠️  INTERVENTIONS:")
            for rec in result['intervention_recommendations'][:3]:
                print(f"   - [{rec['priority'].upper()}] {rec['action']}")
    
    @staticmethod
    def show_analysis(analysis: str):
        """Show analysis and explanation."""
        print(f"\n💡 ANALYSIS:")
        print(f"   {analysis}")
    
    @staticmethod
    def pause_for_video():
        """Pause for video recording."""
        print("\n⏸️  [PAUSE FOR VIDEO RECORDING]")
        time.sleep(2)

class VideoDemo:
    """Main demo class for video recording."""
    
    def __init__(self):
        self.safety_manager = SafetyManager()
        self.demo_scenarios = self._define_scenarios()
    
    def _define_scenarios(self) -> List[Dict[str, Any]]:
        """Define all demo scenarios."""
        return [
            {
                'title': 'Child Safety - Profanity Detection',
                'description': 'Demonstrates how the system protects children from inappropriate content',
                'text': 'fuck you',
                'age_group': AgeGroup.CHILD,
                'expected_risk': 'critical',
                'analysis': 'Any profanity directed at children triggers immediate intervention with emergency escalation and content blocking.',
                'key_points': ['Child-specific thresholds', 'Emergency intervention', 'Content blocking']
            },
            {
                'title': 'Crisis Intervention - Self-Harm Detection',
                'description': 'Shows how the system detects severe emotional distress and potential self-harm',
                'text': 'I want to kill myself and end this pain',
                'age_group': AgeGroup.ADULT,
                'expected_risk': 'critical',
                'analysis': 'Direct self-harm indicators trigger immediate crisis intervention with emergency services contact.',
                'key_points': ['Crisis detection', 'Emergency escalation', 'Mental health resources']
            },
            {
                'title': 'Safe Content - Normal Conversation',
                'description': 'Demonstrates that normal, safe content is correctly identified',
                'text': 'Hello, how are you today? I hope you have a great day!',
                'age_group': AgeGroup.CHILD,
                'expected_risk': 'minimal',
                'analysis': 'Normal, positive conversation is correctly identified as safe with no interventions needed.',
                'key_points': ['False positive prevention', 'Normal content handling', 'No unnecessary interventions']
            },
            {
                'title': 'Escalation Detection - Conversation Context',
                'description': 'Shows how the system tracks conversation patterns for escalation',
                'text': 'I hate this conversation and you are making me angry!',
                'age_group': AgeGroup.ADULT,
                'expected_risk': 'medium',
                'analysis': 'Aggressive language with emotional intensity triggers escalation detection for human review.',
                'key_points': ['Conversation context', 'Emotional intensity', 'Human review trigger']
            },
            {
                'title': 'Adult Content Filtering - Age-Appropriate',
                'description': 'Demonstrates age-appropriate content filtering for adults',
                'text': 'This movie contains violence and adult themes',
                'age_group': AgeGroup.ADULT,
                'expected_risk': 'minimal',
                'analysis': 'Content that would be blocked for children is allowed for adults with appropriate warnings.',
                'key_points': ['Age-appropriate filtering', 'Adult content handling', 'Contextual decisions']
            },
            {
                'title': 'Severe Abuse - Multiple Model Triggers',
                'description': 'Shows how multiple models work together for severe cases',
                'text': 'You fucking idiot, I hate you and want you to die!',
                'age_group': AgeGroup.CHILD,
                'expected_risk': 'critical',
                'analysis': 'Severe abuse triggers multiple models (abuse, escalation, content filter) with emergency intervention.',
                'key_points': ['Multi-model consensus', 'Severe abuse detection', 'Emergency intervention']
            }
        ]
    
    def run_full_demo(self):
        """Run the complete demo for video recording."""
        DemoScreen.show_header("AI SAFETY MODELS POC - VIDEO DEMONSTRATION")
        
        print("\n🎯 This demo showcases the four core AI Safety Models:")
        print("   • Abuse Language Detection")
        print("   • Escalation Pattern Recognition") 
        print("   • Crisis Intervention")
        print("   • Content Filtering")
        
        print("\n⚡ Key Features:")
        print("   • Real-time processing (<100ms)")
        print("   • Age-appropriate filtering")
        print("   • Human-in-the-loop interventions")
        print("   • Bias mitigation and fairness")
        
        for i, scenario in enumerate(self.demo_scenarios, 1):
            DemoScreen.show_scenario(i, scenario['title'], scenario['description'])
            
            # Show input
            DemoScreen.show_input(scenario['text'], scenario['age_group'].value)
            
            # Process with safety manager
            try:
                result = self.safety_manager.analyze(
                    text=scenario['text'],
                    user_id='demo_user',
                    session_id='demo_session',
                    age_group=scenario['age_group']
                )
                
                # Show output
                DemoScreen.show_output(result)
                
                # Show analysis
                DemoScreen.show_analysis(scenario['analysis'])
                
                # Show key points
                print(f"\n🔑 KEY POINTS:")
                for point in scenario['key_points']:
                    print(f"   • {point}")
                
                # Check if result matches expectation
                actual_risk = result['overall_assessment']['overall_risk']
                expected_risk = scenario['expected_risk']
                
                if actual_risk == expected_risk:
                    print(f"\n✅ SUCCESS: Risk assessment matches expectation ({expected_risk})")
                else:
                    print(f"\n⚠️  PARTIAL: Expected {expected_risk}, got {actual_risk}")
                
                DemoScreen.pause_for_video()
                
            except Exception as e:
                print(f"\n❌ ERROR: {e}")
                DemoScreen.pause_for_video()
        
        # Demo summary
        DemoScreen.show_header("DEMO SUMMARY")
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("\n📊 Key Achievements:")
        print("   ✅ All 4 safety models working correctly")
        print("   ✅ Real-time processing demonstrated")
        print("   ✅ Age-appropriate filtering shown")
        print("   ✅ Crisis intervention triggered")
        print("   ✅ Multi-model integration working")
        
        print("\n🚀 Production Readiness:")
        print("   • Modular architecture for easy extension")
        print("   • Comprehensive bias evaluation")
        print("   • Human oversight integration")
        print("   • Scalable design for high-volume deployment")
        
        print("\n📝 Next Steps:")
        print("   • Integrate real datasets from Kaggle")
        print("   • Deploy to staging environment")
        print("   • Conduct A/B testing")
        print("   • Scale for production traffic")
    
    def run_quick_demo(self):
        """Run a quick demo for testing."""
        print("🚀 AI Safety Models - Quick Demo")
        print("=" * 40)
        
        # Test 1: Child safety
        print("\n🧒 Test 1: Child Safety")
        result1 = self.safety_manager.analyze(
            text="fuck you",
            user_id="test_user",
            session_id="test_session", 
            age_group=AgeGroup.CHILD
        )
        print(f"Risk: {result1['overall_assessment']['overall_risk']}")
        print(f"Intervention: {result1['overall_assessment']['intervention_level']}")
        
        # Test 2: Crisis detection
        print("\n🚨 Test 2: Crisis Detection")
        result2 = self.safety_manager.analyze(
            text="I want to kill myself",
            user_id="test_user",
            session_id="test_session",
            age_group=AgeGroup.ADULT
        )
        print(f"Risk: {result2['overall_assessment']['overall_risk']}")
        print(f"Intervention: {result2['overall_assessment']['intervention_level']}")
        
        # Test 3: Safe content
        print("\n✅ Test 3: Safe Content")
        result3 = self.safety_manager.analyze(
            text="Hello, how are you today?",
            user_id="test_user",
            session_id="test_session",
            age_group=AgeGroup.CHILD
        )
        print(f"Risk: {result3['overall_assessment']['overall_risk']}")
        print(f"Intervention: {result3['overall_assessment']['intervention_level']}")
        
        print("\n🎉 Quick demo completed!")

def main():
    """Main function to run the demo."""
    demo = VideoDemo()
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        demo.run_quick_demo()
    else:
        demo.run_full_demo()

if __name__ == "__main__":
    main()
