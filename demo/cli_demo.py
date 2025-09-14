#!/usr/bin/env python3
"""
CLI Demo for AI Safety Models POC.

This script provides a command-line interface to demonstrate the AI Safety Models
system with interactive text analysis and real-time safety assessment.
"""

import sys
import os
import json
from typing import Dict, Any
import argparse

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from safety_system.safety_manager import SafetyManager
    from models.content_filter import AgeGroup
except ImportError:
    # Fallback for different import structures
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from src.safety_system.safety_manager import SafetyManager
        from src.models.content_filter import AgeGroup
    except ImportError:
        print("Error: Could not import safety modules. Please ensure the package is properly installed.")
        print("Try running: pip install -e .")
        sys.exit(1)


class CLIDemo:
    """Command-line interface for AI Safety Models demo."""
    
    def __init__(self):
        """Initialize the CLI demo."""
        self.safety_manager = SafetyManager()
        self.current_user_id = "demo_user"
        self.current_session_id = "demo_session"
        self.current_age_group = AgeGroup.ADULT
        
    def print_header(self):
        """Print demo header."""
        print("=" * 60)
        print("🤖 AI Safety Models POC - Command Line Demo")
        print("=" * 60)
        print("This demo showcases real-time AI safety analysis including:")
        print("• Abuse Language Detection")
        print("• Escalation Pattern Recognition") 
        print("• Crisis Intervention Detection")
        print("• Age-Appropriate Content Filtering")
        print("=" * 60)
        print()
    
    def print_menu(self):
        """Print main menu options."""
        print("\n📋 Available Commands:")
        print("1. Analyze text for safety issues")
        print("2. Set user age group")
        print("3. View conversation summary")
        print("4. Clear conversation history")
        print("5. Show model status")
        print("6. Run predefined test cases")
        print("7. Exit")
        print()
    
    def analyze_text(self):
        """Interactive text analysis."""
        print("\n🔍 Text Safety Analysis")
        print("-" * 30)
        
        text = input("Enter text to analyze: ").strip()
        if not text:
            print("❌ No text provided.")
            return
        
        print(f"\n⏳ Analyzing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            # Perform comprehensive safety analysis
            results = self.safety_manager.analyze(
                text=text,
                user_id=self.current_user_id,
                session_id=self.current_session_id,
                age_group=self.current_age_group
            )
            
            self._display_analysis_results(results)
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
    
    def _display_analysis_results(self, results: Dict[str, Any]):
        """Display analysis results in a formatted way."""
        print("\n📊 Analysis Results:")
        print("=" * 40)
        
        # Overall assessment
        overall = results.get('overall_assessment', {})
        overall_risk = overall.get('overall_risk', 'unknown')
        intervention_level = overall.get('intervention_level', 'none')
        
        risk_emoji = {
            'minimal': '🟢',
            'low': '🟡', 
            'medium': '🟠',
            'high': '🔴',
            'critical': '🚨'
        }
        
        print(f"{risk_emoji.get(overall_risk, '❓')} Overall Risk: {overall_risk.upper()}")
        print(f"🔧 Intervention Level: {intervention_level}")
        
        # Individual model results
        models = results.get('models', {})
        print(f"\n📋 Model Results:")
        
        for model_name, model_result in models.items():
            result = model_result['result']
            risk_level = model_result['risk_level']
            
            print(f"  • {model_name.replace('_', ' ').title()}:")
            print(f"    - Label: {result.label}")
            print(f"    - Score: {result.score:.3f}")
            print(f"    - Risk Level: {risk_level}")
            print(f"    - Safety Level: {result.safety_level.value}")
        
        # Intervention recommendations
        recommendations = results.get('intervention_recommendations', [])
        if recommendations:
            print(f"\n⚠️  Intervention Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                priority = rec.get('priority', 'unknown')
                action = rec.get('action', 'No action specified')
                reason = rec.get('reason', 'No reason provided')
                
                priority_emoji = {
                    'critical': '🚨',
                    'high': '🔴',
                    'medium': '🟠',
                    'low': '🟡'
                }
                
                print(f"  {i}. {priority_emoji.get(priority, '🔵')} [{priority.upper()}] {action}")
                print(f"     Reason: {reason}")
    
    def set_age_group(self):
        """Set user age group for content filtering."""
        print("\n👤 Set User Age Group")
        print("-" * 25)
        print("Available age groups:")
        print("1. Child (5-12 years)")
        print("2. Teen (13-17 years)")
        print("3. Young Adult (18-21 years)")
        print("4. Adult (22+ years)")
        
        choice = input("\nSelect age group (1-4): ").strip()
        
        age_groups = {
            '1': AgeGroup.CHILD,
            '2': AgeGroup.TEEN,
            '3': AgeGroup.YOUNG_ADULT,
            '4': AgeGroup.ADULT
        }
        
        if choice in age_groups:
            self.current_age_group = age_groups[choice]
            print(f"✅ Age group set to: {self.current_age_group.value}")
        else:
            print("❌ Invalid choice.")
    
    def view_conversation_summary(self):
        """View conversation summary."""
        print("\n📈 Conversation Summary")
        print("-" * 25)
        
        try:
            summary = self.safety_manager.get_conversation_summary(
                self.current_user_id, 
                self.current_session_id
            )
            
            if 'error' in summary:
                print(f"❌ {summary['error']}")
                return
            
            print(f"Message Count: {summary.get('message_count', 0)}")
            print(f"Escalation Risk: {summary.get('escalation_risk', 'unknown')}")
            print(f"Average Negative Score: {summary.get('avg_negative_score', 0):.3f}")
            
            last_message = summary.get('last_message_time')
            if last_message:
                print(f"Last Message: {last_message}")
                
        except Exception as e:
            print(f"❌ Error getting conversation summary: {e}")
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        print("\n🗑️  Clear Conversation History")
        print("-" * 30)
        
        confirm = input("Are you sure you want to clear the conversation history? (y/N): ").strip().lower()
        
        if confirm == 'y':
            self.safety_manager.clear_conversation_history(
                self.current_user_id, 
                self.current_session_id
            )
            print("✅ Conversation history cleared.")
        else:
            print("❌ Operation cancelled.")
    
    def show_model_status(self):
        """Show status of all models."""
        print("\n🔧 Model Status")
        print("-" * 15)
        
        try:
            status = self.safety_manager.get_model_status()
            
            for model_name, model_status in status.items():
                print(f"\n{model_name.replace('_', ' ').title()}:")
                
                if model_status.get('initialized', False):
                    print(f"  ✅ Initialized: {model_status.get('model_type', 'unknown')}")
                    print(f"  📚 Trained: {model_status.get('trained', False)}")
                    print(f"  🎯 Threshold: {model_status.get('threshold', 'unknown')}")
                else:
                    print(f"  ❌ Not initialized")
                    if 'error' in model_status:
                        print(f"  Error: {model_status['error']}")
                        
        except Exception as e:
            print(f"❌ Error getting model status: {e}")
    
    def run_test_cases(self):
        """Run predefined test cases."""
        print("\n🧪 Running Test Cases")
        print("-" * 20)
        
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
                'expected_risk': 'medium'
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
            
            try:
                results = self.safety_manager.analyze(
                    text=test_case['text'],
                    user_id=f"test_user_{i}",
                    session_id=f"test_session_{i}",
                    age_group=AgeGroup.ADULT
                )
                
                overall_risk = results.get('overall_assessment', {}).get('overall_risk', 'unknown')
                expected_risk = test_case['expected_risk']
                
                if overall_risk == expected_risk:
                    print(f"✅ PASS - Risk: {overall_risk}")
                else:
                    print(f"⚠️  PARTIAL - Expected: {expected_risk}, Got: {overall_risk}")
                
            except Exception as e:
                print(f"❌ FAIL - Error: {e}")
            
            print()
    
    def run(self):
        """Run the CLI demo."""
        self.print_header()
        
        while True:
            self.print_menu()
            
            try:
                choice = input("Enter your choice (1-7): ").strip()
                
                if choice == '1':
                    self.analyze_text()
                elif choice == '2':
                    self.set_age_group()
                elif choice == '3':
                    self.view_conversation_summary()
                elif choice == '4':
                    self.clear_conversation_history()
                elif choice == '5':
                    self.show_model_status()
                elif choice == '6':
                    self.run_test_cases()
                elif choice == '7':
                    print("\n👋 Thank you for trying the AI Safety Models POC!")
                    print("For more information, visit: https://github.com/your-repo/ai-safety-poc")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-7.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")


def main():
    """Main entry point for CLI demo."""
    parser = argparse.ArgumentParser(description='AI Safety Models POC - CLI Demo')
    parser.add_argument('--version', action='version', version='AI Safety POC v1.0.0')
    
    args = parser.parse_args()
    
    try:
        demo = CLIDemo()
        demo.run()
    except Exception as e:
        print(f"❌ Failed to start demo: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
