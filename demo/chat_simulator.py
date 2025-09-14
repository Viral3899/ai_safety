"""
Chat Simulator for AI Safety Models Demo.

This module provides a chat simulator to demonstrate the AI Safety Models in action.
"""

import sys
import os
import time
import random
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from safety_system.safety_manager import SafetyManager
from models.content_filter import AgeGroup


class ChatSimulator:
    """Chat simulator for AI Safety Models demonstration."""
    
    def __init__(self):
        """Initialize the chat simulator."""
        self.safety_manager = SafetyManager()
        self.chat_history = []
        self.current_user = None
        self.current_age_group = AgeGroup.ADULT
        
        # Sample conversation scenarios
        self.scenarios = {
            'normal_conversation': [
                "Hello, how are you today?",
                "I'm doing great, thank you for asking!",
                "What are your plans for the weekend?",
                "I'm planning to go hiking with some friends.",
                "That sounds like fun! Have a great time!"
            ],
            'escalating_anger': [
                "This is so frustrating!",
                "I can't believe this is happening again.",
                "This is damn annoying!",
                "I hate this stupid system!",
                "I want to hurt someone!"
            ],
            'crisis_situation': [
                "I'm feeling really down today.",
                "Nothing seems to be going right.",
                "I don't see any point in continuing.",
                "I want to kill myself.",
                "I can't go on anymore."
            ],
            'inappropriate_content': [
                "This movie has explicit sexual content.",
                "There's graphic violence and gore.",
                "This contains strong language throughout.",
                "There are disturbing themes here.",
                "This is not suitable for children."
            ]
        }
    
    def start_chat(self, user_id: str = "demo_user", age_group: AgeGroup = AgeGroup.ADULT):
        """Start a new chat session."""
        self.current_user = user_id
        self.current_age_group = age_group
        self.chat_history = []
        
        print(f"💬 Chat Simulator Started")
        print(f"👤 User: {user_id}")
        print(f"👶 Age Group: {age_group.value}")
        print("=" * 40)
    
    def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message and get safety analysis."""
        if not self.current_user:
            raise ValueError("No active chat session")
        
        # Add message to history
        self.chat_history.append({
            'message': message,
            'timestamp': time.time(),
            'user': self.current_user
        })
        
        # Analyze message
        result = self.safety_manager.analyze(
            text=message,
            user_id=self.current_user,
            session_id="chat_simulator",
            age_group=self.current_age_group
        )
        
        # Add to result
        result['chat_info'] = {
            'message_count': len(self.chat_history),
            'timestamp': time.time()
        }
        
        return result
    
    def run_scenario(self, scenario_name: str) -> List[Dict[str, Any]]:
        """Run a predefined conversation scenario."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        print(f"🎭 Running Scenario: {scenario_name.replace('_', ' ').title()}")
        print("-" * 40)
        
        results = []
        messages = self.scenarios[scenario_name]
        
        for i, message in enumerate(messages, 1):
            print(f"\n📝 Message {i}: {message}")
            
            result = self.send_message(message)
            results.append(result)
            
            # Display safety analysis
            overall_assessment = result['overall_assessment']
            print(f"📊 Risk Level: {overall_assessment['overall_risk'].upper()}")
            print(f"🔧 Intervention: {overall_assessment['intervention_level']}")
            
            # Show model breakdown
            for model_name, model_result in result['models'].items():
                risk_level = model_result['risk_level']
                score = model_result['result'].score
                print(f"  • {model_name.replace('_', ' ').title()}: {risk_level} ({score:.3f})")
            
            # Show interventions if any
            if result['intervention_recommendations']:
                print("⚠️  Interventions:")
                for rec in result['intervention_recommendations'][:2]:
                    print(f"  - {rec['action']}")
            
            # Add delay for realism
            time.sleep(1)
        
        return results
    
    def interactive_chat(self):
        """Start interactive chat mode."""
        print("🎮 Interactive Chat Mode")
        print("Type messages to test the safety models (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            try:
                message = input("\n👤 You: ").strip()
                
                if message.lower() in ['quit', 'exit', 'q']:
                    print("👋 Chat ended. Goodbye!")
                    break
                
                if not message:
                    print("❌ Please enter a message.")
                    continue
                
                result = self.send_message(message)
                
                # Display results
                overall_assessment = result['overall_assessment']
                print(f"🤖 Safety Analysis:")
                print(f"  Risk: {overall_assessment['overall_risk'].upper()}")
                print(f"  Intervention: {overall_assessment['intervention_level']}")
                
                # Show any interventions
                if result['intervention_recommendations']:
                    print("⚠️  Recommendations:")
                    for rec in result['intervention_recommendations']:
                        print(f"  - {rec['action']}")
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def get_chat_summary(self) -> Dict[str, Any]:
        """Get summary of current chat session."""
        if not self.chat_history:
            return {'message_count': 0}
        
        # Analyze all messages for patterns
        high_risk_count = 0
        intervention_count = 0
        
        for message_data in self.chat_history:
            # This would normally analyze each message
            # For demo purposes, we'll simulate
            pass
        
        return {
            'message_count': len(self.chat_history),
            'session_duration': time.time() - self.chat_history[0]['timestamp'],
            'user': self.current_user,
            'age_group': self.current_age_group.value
        }


def demo_chat_simulator():
    """Demonstrate the chat simulator."""
    print("🚀 AI Safety Models - Chat Simulator Demo")
    print("=" * 45)
    
    simulator = ChatSimulator()
    simulator.start_chat("demo_user", AgeGroup.ADULT)
    
    # Run different scenarios
    scenarios_to_run = ['normal_conversation', 'escalating_anger', 'crisis_situation']
    
    for scenario in scenarios_to_run:
        try:
            results = simulator.run_scenario(scenario)
            
            # Summary for this scenario
            high_risk_count = sum(1 for r in results 
                                if r['overall_assessment']['overall_risk'] in ['high', 'critical'])
            
            print(f"\n📈 Scenario Summary:")
            print(f"  Messages: {len(results)}")
            print(f"  High Risk Messages: {high_risk_count}")
            print(f"  Safety Triggered: {'Yes' if high_risk_count > 0 else 'No'}")
            
        except Exception as e:
            print(f"❌ Error running scenario {scenario}: {e}")
        
        print("\n" + "="*50)
    
    # Show chat summary
    summary = simulator.get_chat_summary()
    print(f"\n📊 Chat Session Summary:")
    print(f"  Total Messages: {summary['message_count']}")
    print(f"  User: {summary['user']}")
    print(f"  Age Group: {summary['age_group']}")


def main():
    """Main entry point for chat simulator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Safety Models Chat Simulator')
    parser.add_argument('--mode', choices=['demo', 'interactive'], default='demo',
                       help='Run mode: demo scenarios or interactive chat')
    parser.add_argument('--age-group', choices=['child', 'teen', 'adult'], default='adult',
                       help='Age group for content filtering')
    
    args = parser.parse_args()
    
    simulator = ChatSimulator()
    
    # Convert age group string to enum
    age_group_map = {
        'child': AgeGroup.CHILD,
        'teen': AgeGroup.TEEN,
        'adult': AgeGroup.ADULT
    }
    
    simulator.start_chat("interactive_user", age_group_map[args.age_group])
    
    if args.mode == 'demo':
        demo_chat_simulator()
    else:
        simulator.interactive_chat()


if __name__ == "__main__":
    main()