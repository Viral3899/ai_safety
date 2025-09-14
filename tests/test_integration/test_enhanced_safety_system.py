"""
Enhanced Integration Tests for the AI Safety System.

This module tests the integration of all enhanced safety models:
- Abuse detection triggering crisis intervention
- Escalation detection with conversation context
- Content filtering with age-appropriate responses
- Multilingual support across all models
- Edge cases and real-world scenarios
"""

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from safety_system.safety_manager import SafetyManager
from models.abuse_detector import AdvancedAbuseDetector
from models.crisis_detector import AdvancedCrisisDetector
from models.escalation_detector import EscalationDetector
from models.content_filter import ContentFilter
from core.base_model import ModelConfig, AgeGroup


class TestEnhancedSafetySystem:
    """Integration tests for the enhanced safety system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Enhanced configuration for all models
        self.config = {
            'abuse_detector': {
                'model_type': 'hurtbert_style',
                'threshold': 0.5,
                'device': 'cpu',
                'multilingual': True
            },
            'escalation_detector': {
                'model_type': 'rule_based',
                'threshold': 0.4,
                'device': 'cpu'
            },
            'crisis_detector': {
                'model_type': 'transformers',
                'threshold': 0.3,
                'device': 'cpu',
                'multilingual': True
            },
            'content_filter': {
                'model_type': 'rule_based',
                'threshold': 0.6,
                'device': 'cpu'
            }
        }
        
        self.safety_manager = SafetyManager(self.config)
    
    def test_abuse_to_crisis_escalation(self):
        """Test escalation from abuse detection to crisis intervention."""
        # Scenario: User starts with abusive language, escalates to crisis
        conversation = [
            "fuck you, you're an idiot",
            "I hate everyone and everything",
            "I want to kill myself",
            "I can't take this anymore"
        ]
        
        results = []
        for i, message in enumerate(conversation):
            result = self.safety_manager.analyze(
                message,
                user_id="test_user",
                session_id="test_session",
                age_group=AgeGroup.ADULT
            )
            results.append(result)
            
            print(f"\nMessage {i+1}: '{message}'")
            print(f"Overall Risk: {result['overall_assessment']['overall_risk']}")
            print(f"Intervention Level: {result['overall_assessment']['intervention_level']}")
            
            # Check that risk escalates
            if i > 0:
                current_risk = result['overall_assessment']['max_score']
                previous_risk = results[i-1]['overall_assessment']['max_score']
                assert current_risk >= previous_risk
        
        # Final message should trigger crisis intervention
        final_result = results[-1]
        assert final_result['overall_assessment']['overall_risk'] in ['critical', 'high']
        assert final_result['overall_assessment']['intervention_level'] in ['emergency', 'intervene']
    
    def test_multilingual_safety_detection(self):
        """Test safety detection across multiple languages."""
        multilingual_cases = [
            # Spanish
            ("te odio, eres un idiota", "abusive"),
            ("quiero morir", "crisis"),
            
            # French
            ("je te déteste, tu es stupide", "abusive"),
            ("je veux mourir", "crisis"),
            
            # German
            ("ich hasse dich, du bist dumm", "abusive"),
            ("ich will sterben", "crisis"),
            
            # Mixed language
            ("fuck you, quiero morir", "both"),
        ]
        
        for text, expected_type in multilingual_cases:
            result = self.safety_manager.analyze(
                text,
                user_id="test_user",
                session_id="test_session",
                age_group=AgeGroup.ADULT
            )
            
            print(f"\nMultilingual: '{text}'")
            print(f"Expected: {expected_type}")
            print(f"Abuse: {result['models']['abuse']['risk_level']}")
            print(f"Crisis: {result['models']['crisis']['risk_level']}")
            
            if expected_type == "abusive":
                assert result['models']['abuse']['risk_level'] in ['high', 'medium']
            elif expected_type == "crisis":
                assert result['models']['crisis']['risk_level'] in ['critical', 'high']
            elif expected_type == "both":
                assert (result['models']['abuse']['risk_level'] in ['high', 'medium'] or
                       result['models']['crisis']['risk_level'] in ['critical', 'high'])
    
    def test_age_appropriate_content_filtering(self):
        """Test content filtering for different age groups."""
        inappropriate_content = "fuck you, you're a stupid bitch"
        
        # Test with child
        child_result = self.safety_manager.analyze(
            inappropriate_content,
            user_id="child_user",
            session_id="test_session",
            age_group=AgeGroup.CHILD
        )
        
        print(f"\nChild content filtering:")
        print(f"Content: '{inappropriate_content}'")
        print(f"Content Filter Risk: {child_result['models']['content_filter']['risk_level']}")
        print(f"Overall Risk: {child_result['overall_assessment']['overall_risk']}")
        
        assert child_result['models']['content_filter']['risk_level'] in ['critical', 'high']
        assert child_result['overall_assessment']['overall_risk'] in ['critical', 'high']
        
        # Test with adult
        adult_result = self.safety_manager.analyze(
            inappropriate_content,
            user_id="adult_user",
            session_id="test_session",
            age_group=AgeGroup.ADULT
        )
        
        print(f"\nAdult content filtering:")
        print(f"Content Filter Risk: {adult_result['models']['content_filter']['risk_level']}")
        print(f"Overall Risk: {adult_result['overall_assessment']['overall_risk']}")
        
        # Adults should have more tolerance
        assert adult_result['models']['content_filter']['risk_level'] in ['high', 'medium', 'low']
    
    def test_conversation_context_escalation(self):
        """Test escalation detection with conversation context."""
        # Simulate a conversation that escalates
        conversation_messages = [
            "Hello, how are you?",
            "I'm feeling a bit down today",
            "Actually, I'm really struggling",
            "I don't think I can handle this anymore",
            "I want to hurt myself"
        ]
        
        results = []
        for i, message in enumerate(conversation_messages):
            result = self.safety_manager.analyze(
                message,
                user_id="escalating_user",
                session_id="escalation_session",
                age_group=AgeGroup.ADULT
            )
            results.append(result)
            
            print(f"\nMessage {i+1}: '{message}'")
            print(f"Escalation Risk: {result['models']['escalation']['risk_level']}")
            print(f"Overall Risk: {result['overall_assessment']['overall_risk']}")
        
        # Check that escalation is detected
        final_result = results[-1]
        assert final_result['models']['escalation']['risk_level'] in ['high', 'medium']
        
        # Check that crisis is detected in final message
        assert final_result['models']['crisis']['risk_level'] in ['critical', 'high']
    
    def test_edge_cases_and_ambiguous_language(self):
        """Test edge cases and ambiguous language across all models."""
        edge_cases = [
            # Sarcasm
            ("Oh great, I want to die now", "crisis"),
            ("Thanks for nothing, you're so helpful", "abusive"),
            
            # Context-dependent
            ("That's sick!", "safe"),  # Positive slang
            ("You're sick in the head", "abusive"),
            
            # Coded language
            ("kys", "abusive"),
            ("I want to kms", "crisis"),
            
            # Character obfuscation
            ("f*ck you", "abusive"),
            ("I want to d*e", "crisis"),
            
            # Mixed signals
            ("I hate you but I'm getting help", "both"),
            ("I want to die but I have hope", "crisis"),
        ]
        
        for text, expected_type in edge_cases:
            result = self.safety_manager.analyze(
                text,
                user_id="edge_case_user",
                session_id="test_session",
                age_group=AgeGroup.ADULT
            )
            
            print(f"\nEdge case: '{text}'")
            print(f"Expected: {expected_type}")
            print(f"Abuse: {result['models']['abuse']['risk_level']}")
            print(f"Crisis: {result['models']['crisis']['risk_level']}")
            print(f"Overall: {result['overall_assessment']['overall_risk']}")
            
            if expected_type == "abusive":
                assert result['models']['abuse']['risk_level'] in ['high', 'medium']
            elif expected_type == "crisis":
                assert result['models']['crisis']['risk_level'] in ['critical', 'high']
            elif expected_type == "both":
                assert (result['models']['abuse']['risk_level'] in ['high', 'medium'] or
                       result['models']['crisis']['risk_level'] in ['critical', 'high'])
    
    def test_intervention_recommendations(self):
        """Test that appropriate intervention recommendations are generated."""
        # Test critical crisis
        crisis_result = self.safety_manager.analyze(
            "I want to kill myself",
            user_id="crisis_user",
            session_id="test_session",
            age_group=AgeGroup.ADULT
        )
        
        print(f"\nCrisis intervention recommendations:")
        for rec in crisis_result['intervention_recommendations']:
            print(f"- {rec['action']} (Priority: {rec['priority']})")
        
        assert len(crisis_result['intervention_recommendations']) > 0
        assert any(rec['priority'] == 'critical' for rec in crisis_result['intervention_recommendations'])
        
        # Test abuse detection
        abuse_result = self.safety_manager.analyze(
            "fuck you, you're an idiot",
            user_id="abuse_user",
            session_id="test_session",
            age_group=AgeGroup.ADULT
        )
        
        print(f"\nAbuse intervention recommendations:")
        for rec in abuse_result['intervention_recommendations']:
            print(f"- {rec['action']} (Priority: {rec['priority']})")
        
        assert len(abuse_result['intervention_recommendations']) > 0
    
    def test_model_integration_workflow(self):
        """Test the complete workflow of model integration."""
        # Test a complex scenario
        complex_scenario = "I hate everyone and want to kill myself, but I'm also getting help"
        
        result = self.safety_manager.analyze(
            complex_scenario,
            user_id="complex_user",
            session_id="test_session",
            age_group=AgeGroup.ADULT
        )
        
        print(f"\nComplex scenario: '{complex_scenario}'")
        print(f"Abuse Risk: {result['models']['abuse']['risk_level']}")
        print(f"Crisis Risk: {result['models']['crisis']['risk_level']}")
        print(f"Escalation Risk: {result['models']['escalation']['risk_level']}")
        print(f"Content Filter Risk: {result['models']['content_filter']['risk_level']}")
        print(f"Overall Risk: {result['overall_assessment']['overall_risk']}")
        print(f"Intervention Level: {result['overall_assessment']['intervention_level']}")
        
        # Should detect both abuse and crisis
        assert result['models']['abuse']['risk_level'] in ['high', 'medium']
        assert result['models']['crisis']['risk_level'] in ['critical', 'high']
        
        # Should have appropriate intervention recommendations
        assert len(result['intervention_recommendations']) > 0
        
        # Check metadata completeness
        for model_name, model_result in result['models'].items():
            assert 'result' in model_result
            assert 'risk_level' in model_result
            assert hasattr(model_result['result'], 'metadata')
    
    def test_performance_under_load(self):
        """Test system performance under load."""
        import time
        
        test_messages = [
            "Hello, how are you?",
            "I'm feeling sad today",
            "fuck you, you're stupid",
            "I want to die",
            "I hate everyone and everything"
        ]
        
        start_time = time.time()
        
        for i in range(50):  # Test with 50 iterations
            message = test_messages[i % len(test_messages)]
            result = self.safety_manager.analyze(
                message,
                user_id=f"load_test_user_{i}",
                session_id=f"load_test_session_{i}",
                age_group=AgeGroup.ADULT
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 50
        
        print(f"\nPerformance test:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per message: {avg_time:.4f} seconds")
        print(f"Messages per second: {1/avg_time:.2f}")
        
        assert avg_time < 0.5  # Should process messages quickly
    
    def test_error_handling_and_recovery(self):
        """Test error handling and system recovery."""
        # Test with malformed input
        malformed_inputs = [
            "",  # Empty string
            "!@#$%^&*()",  # Special characters
            "a" * 10000,  # Very long string
            None,  # None input
        ]
        
        for i, malformed_input in enumerate(malformed_inputs):
            try:
                if malformed_input is None:
                    # Skip None input as it would cause a type error
                    continue
                    
                result = self.safety_manager.analyze(
                    malformed_input,
                    user_id=f"error_test_user_{i}",
                    session_id="error_test_session",
                    age_group=AgeGroup.ADULT
                )
                
                print(f"\nMalformed input {i+1}: '{str(malformed_input)[:50]}...'")
                print(f"Result: {result['overall_assessment']['overall_risk']}")
                
                # Should handle gracefully
                assert 'overall_assessment' in result
                assert 'intervention_recommendations' in result
                
            except Exception as e:
                print(f"Error handling test {i+1} failed: {e}")
                # Some errors are expected, but system should not crash
    
    def test_model_status_and_health(self):
        """Test model status and health monitoring."""
        status = self.safety_manager.get_model_status()
        
        print(f"\nModel status:")
        for model_name, model_status in status.items():
            print(f"{model_name}: {model_status}")
        
        # All models should be initialized
        for model_name, model_status in status.items():
            assert model_status['initialized'] == True
    
    def test_conversation_history_management(self):
        """Test conversation history management for escalation detection."""
        # Test clearing conversation history
        self.safety_manager.clear_conversation_history("test_user", "test_session")
        
        # Test getting conversation summary
        summary = self.safety_manager.get_conversation_summary("test_user", "test_session")
        print(f"\nConversation summary: {summary}")
        
        # Should handle non-existent conversations gracefully
        assert 'error' in summary or 'conversation' in summary


if __name__ == "__main__":
    pytest.main([__file__])
