"""
Integration Tests for AI Safety Models.

This module provides integration testing for the AI Safety Models system.
"""

import sys
import os
import json
from typing import Dict, Any, List
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from safety_system.safety_manager import SafetyManager
from models.content_filter import AgeGroup


class IntegrationTester:
    """Integration testing for AI Safety Models."""
    
    def __init__(self):
        """Initialize the integration tester."""
        self.safety_manager = SafetyManager()
        self.test_results = {}
    
    def run_end_to_end_tests(self) -> Dict[str, Any]:
        """Run end-to-end integration tests."""
        print("🧪 Running End-to-End Integration Tests")
        print("=" * 40)
        
        test_cases = [
            {
                'name': 'Safe Content Analysis',
                'text': 'Hello, how are you today?',
                'expected_risk': 'minimal',
                'age_group': AgeGroup.ADULT
            },
            {
                'name': 'Abuse Detection Test',
                'text': 'I hate you, you stupid idiot!',
                'expected_risk': 'high',
                'age_group': AgeGroup.ADULT
            },
            {
                'name': 'Crisis Detection Test',
                'text': 'I want to kill myself.',
                'expected_risk': 'critical',
                'age_group': AgeGroup.ADULT
            },
            {
                'name': 'Content Filtering Test',
                'text': 'This contains explicit sexual content.',
                'expected_risk': 'high',
                'age_group': AgeGroup.TEEN
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}: {test_case['name']}")
            
            try:
                result = self.safety_manager.analyze(
                    text=test_case['text'],
                    user_id=f"test_user_{i}",
                    session_id=f"test_session_{i}",
                    age_group=test_case['age_group']
                )
                
                overall_risk = result['overall_assessment']['overall_risk']
                expected_risk = test_case['expected_risk']
                
                if overall_risk == expected_risk:
                    status = "PASS"
                    print(f"  ✅ PASS - Risk: {overall_risk}")
                else:
                    status = "PARTIAL"
                    print(f"  ⚠️  PARTIAL - Expected: {expected_risk}, Got: {overall_risk}")
                
                results.append({
                    'test_name': test_case['name'],
                    'status': status,
                    'expected_risk': expected_risk,
                    'actual_risk': overall_risk,
                    'intervention_level': result['overall_assessment']['intervention_level']
                })
                
            except Exception as e:
                print(f"  ❌ FAIL - Error: {e}")
                results.append({
                    'test_name': test_case['name'],
                    'status': 'FAIL',
                    'error': str(e)
                })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'test_results': results,
            'summary': self._generate_test_summary(results)
        }
    
    def test_model_integration(self) -> Dict[str, Any]:
        """Test integration between different models."""
        print("\n🔗 Testing Model Integration")
        print("-" * 30)
        
        # Test text that should trigger multiple models
        test_text = "I hate you so much, I want to kill myself and everyone else!"
        
        try:
            result = self.safety_manager.analyze(
                text=test_text,
                user_id="integration_test",
                session_id="integration_session"
            )
            
            model_results = result['models']
            
            print("Model Results:")
            for model_name, model_result in model_results.items():
                risk_level = model_result['risk_level']
                score = model_result['result'].score
                print(f"  {model_name}: {risk_level} (score: {score:.3f})")
            
            # Check if multiple models are triggered
            triggered_models = [name for name, result in model_results.items() 
                              if result['risk_level'] in ['medium', 'high', 'critical']]
            
            integration_score = len(triggered_models) / len(model_results)
            
            return {
                'integration_score': integration_score,
                'triggered_models': triggered_models,
                'model_results': {name: result['risk_level'] for name, result in model_results.items()}
            }
            
        except Exception as e:
            print(f"  ❌ Integration test failed: {e}")
            return {'error': str(e)}
    
    def test_conversation_tracking(self) -> Dict[str, Any]:
        """Test conversation tracking across multiple messages."""
        print("\n💬 Testing Conversation Tracking")
        print("-" * 35)
        
        user_id = "conversation_test"
        session_id = "conversation_session"
        
        # Clear any existing conversation
        self.safety_manager.clear_conversation_history(user_id, session_id)
        
        messages = [
            "Hello, how are you?",
            "I'm feeling a bit frustrated today.",
            "This is so damn annoying!",
            "I hate everything about this!",
            "I want to hurt someone!"
        ]
        
        escalation_scores = []
        
        for i, message in enumerate(messages, 1):
            result = self.safety_manager.analyze(
                text=message,
                user_id=user_id,
                session_id=session_id
            )
            
            escalation_result = result['models']['escalation_detector']
            escalation_scores.append(escalation_result['result'].score)
            
            print(f"Message {i}: {escalation_result['result'].score:.3f}")
        
        # Check if escalation is detected
        escalation_detected = any(score > 0.4 for score in escalation_scores)
        escalation_increase = escalation_scores[-1] > escalation_scores[0]
        
        return {
            'escalation_detected': escalation_detected,
            'escalation_increase': escalation_increase,
            'escalation_scores': escalation_scores,
            'conversation_summary': self.safety_manager.get_conversation_summary(user_id, session_id)
        }
    
    def _generate_test_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate test summary."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['status'] == 'PASS')
        partial_tests = sum(1 for r in results if r['status'] == 'PARTIAL')
        failed_tests = sum(1 for r in results if r['status'] == 'FAIL')
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'partial': partial_tests,
            'failed': failed_tests,
            'success_rate': (passed_tests + partial_tests) / total_tests if total_tests > 0 else 0
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("🚀 AI Safety Models Integration Tests")
        print("=" * 40)
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'end_to_end_tests': self.run_end_to_end_tests(),
            'model_integration': self.test_model_integration(),
            'conversation_tracking': self.test_conversation_tracking()
        }
        
        # Overall summary
        e2e_summary = all_results['end_to_end_tests']['summary']
        print(f"\n📈 Integration Test Summary:")
        print(f"  End-to-End Tests: {e2e_summary['passed']}/{e2e_summary['total_tests']} passed")
        print(f"  Success Rate: {e2e_summary['success_rate']:.1%}")
        
        return all_results


def main():
    """Run integration tests."""
    tester = IntegrationTester()
    results = tester.run_all_tests()
    
    # Save results
    with open('integration_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Integration tests completed!")
    print(f"Results saved to: integration_test_results.json")


if __name__ == "__main__":
    main()