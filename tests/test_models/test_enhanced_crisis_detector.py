"""
Test cases for the enhanced crisis detector.
"""

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.crisis_detector import AdvancedCrisisDetector
from core.base_model import ModelConfig, SafetyLevel


class TestEnhancedCrisisDetector:
    """Test cases for the enhanced crisis detector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.configs = {
            'rule_based': ModelConfig(
                model_type='rule_based',
                threshold=0.3,
                device='cpu'
            ),
            'transformers': ModelConfig(
                model_type='transformers',
                threshold=0.3,
                device='cpu'
            ),
            'multilingual': ModelConfig(
                model_type='rule_based',
                threshold=0.3,
                device='cpu',
                multilingual=True
            )
        }
    
    def test_immediate_threat_detection(self):
        """Test detection of immediate self-harm threats."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        immediate_threats = [
            "I want to kill myself",
            "I'm going to end it all",
            "I can't go on living",
            "I want to die",
            "I'm going to hurt myself",
            "I want to cut myself",
            "I'm going to overdose",
            "I want to jump off a bridge",
            "This is my final goodbye",
            "I'll never see you again"
        ]
        
        for text in immediate_threats:
            result = detector.predict(text)
            print(f"Immediate threat: '{text}' -> {result.label} (score: {result.score:.3f})")
            assert result.label in ["critical_crisis", "severe_crisis"]
            assert result.score > 0.6
    
    def test_severe_distress_detection(self):
        """Test detection of severe emotional distress."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        severe_distress = [
            "I feel hopeless",
            "I'm worthless",
            "Nobody cares about me",
            "I'm a burden to everyone",
            "Everyone would be better off without me",
            "I can't take it anymore",
            "I'm giving up",
            "I feel so alone",
            "I have nothing left to live for",
            "I'm dead inside"
        ]
        
        for text in severe_distress:
            result = detector.predict(text)
            print(f"Severe distress: '{text}' -> {result.label} (score: {result.score:.3f})")
            assert result.label in ["severe_crisis", "moderate_crisis", "critical_crisis"]
            assert result.score > 0.3
    
    def test_emotional_crisis_detection(self):
        """Test detection of emotional crisis situations."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        emotional_crisis = [
            "I'm having a breakdown",
            "I'm falling apart",
            "I can't cope with this",
            "I'm overwhelmed",
            "I'm drowning in problems",
            "I'm having a panic attack",
            "I need help immediately",
            "I'm desperate",
            "This is an emergency",
            "I can't handle this anymore"
        ]
        
        for text in emotional_crisis:
            result = detector.predict(text)
            print(f"Emotional crisis: '{text}' -> {result.label} (score: {result.score:.3f})")
            assert result.label in ["moderate_crisis", "severe_crisis", "critical_crisis"]
            assert result.score > 0.2
    
    def test_substance_crisis_detection(self):
        """Test detection of substance-related crisis."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        substance_crisis = [
            "I'm drunk and want to die",
            "I took too many pills",
            "I'm high and feeling suicidal",
            "I'm addicted and can't stop",
            "I'm going to overdose",
            "I need help with my addiction",
            "I'm in withdrawal and it's killing me"
        ]
        
        for text in substance_crisis:
            result = detector.predict(text)
            print(f"Substance crisis: '{text}' -> {result.label} (score: {result.score:.3f})")
            assert result.label in ["moderate_crisis", "severe_crisis", "critical_crisis"]
    
    def test_relationship_crisis_detection(self):
        """Test detection of relationship-related crisis."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        relationship_crisis = [
            "My partner left me and I want to die",
            "I'm being abused and can't take it",
            "I'm being bullied and it's too much",
            "I was betrayed and feel worthless",
            "I'm being harassed and it's destroying me",
            "I feel rejected by everyone"
        ]
        
        for text in relationship_crisis:
            result = detector.predict(text)
            print(f"Relationship crisis: '{text}' -> {result.label} (score: {result.score:.3f})")
            assert result.label in ["moderate_crisis", "severe_crisis", "critical_crisis"]
    
    def test_support_seeking_behavior(self):
        """Test detection of support-seeking behavior (protective factor)."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        support_seeking = [
            "I need help and want to talk to someone",
            "I'm reaching out for support",
            "I want to call a crisis line",
            "I need to see a therapist",
            "I'm asking for help because I'm struggling",
            "I want to talk to my doctor about this",
            "I need mental health support"
        ]
        
        for text in support_seeking:
            result = detector.predict(text)
            print(f"Support seeking: '{text}' -> {result.label} (score: {result.score:.3f})")
            # Support-seeking should reduce crisis score
            assert result.metadata['scores']['support_seeking'] > 0
    
    def test_protective_factors(self):
        """Test detection of protective factors."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        protective_factors = [
            "I have plans for tomorrow",
            "My family needs me",
            "I have faith that things will get better",
            "I'm hopeful about the future",
            "I'm getting treatment and it's helping",
            "I have children who depend on me"
        ]
        
        for text in protective_factors:
            result = detector.predict(text)
            print(f"Protective factors: '{text}' -> {result.label} (score: {result.score:.3f})")
            # Protective factors should reduce crisis score
            assert result.metadata['scores']['protective_factors'] > 0
    
    def test_multilingual_crisis_detection(self):
        """Test crisis detection in multiple languages."""
        detector = AdvancedCrisisDetector(self.configs['multilingual'])
        
        multilingual_cases = [
            # Spanish
            ("quiero morir", "severe_crisis"),  # "I want to die"
            ("me quiero suicidar", "critical_crisis"),  # "I want to commit suicide"
            ("necesito ayuda", "moderate_crisis"),  # "I need help"
            
            # French
            ("je veux mourir", "severe_crisis"),  # "I want to die"
            ("je veux me suicider", "critical_crisis"),  # "I want to commit suicide"
            ("j'ai besoin d'aide", "moderate_crisis"),  # "I need help"
            
            # German
            ("ich will sterben", "severe_crisis"),  # "I want to die"
            ("ich will mich umbringen", "critical_crisis"),  # "I want to kill myself"
            ("ich brauche hilfe", "moderate_crisis"),  # "I need help"
        ]
        
        for text, expected_label in multilingual_cases:
            result = detector.predict(text)
            print(f"Multilingual: '{text}' -> {result.label} (score: {result.score:.3f})")
            assert result.label in ["moderate_crisis", "severe_crisis", "critical_crisis"]
    
    def test_edge_cases_ambiguous_language(self):
        """Test edge cases with ambiguous language."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        ambiguous_cases = [
            # Metaphorical language
            ("I'm dying of laughter", "safe"),
            ("This is killing me", "safe"),  # Context-dependent
            ("I want to die of embarrassment", "safe"),
            
            # Song lyrics or quotes
            ("'I want to die' - that's from a song", "safe"),
            ("As the saying goes, 'I'm dying inside'", "safe"),
            
            # Medical context
            ("I'm dying from cancer", "safe"),
            ("The patient is dying", "safe"),
            
            # Gaming context
            ("I died in the game", "safe"),
            ("I want to die in this level", "safe"),
            
            # Sarcasm
            ("Oh great, I want to die now", "moderate_crisis"),  # Sarcastic but concerning
            ("This is so bad I want to die", "moderate_crisis"),
            
            # Mixed signals
            ("I want to die but I'm getting help", "moderate_crisis"),  # Crisis but seeking help
            ("I feel hopeless but I have hope", "moderate_crisis"),  # Contradictory
        ]
        
        for text, expected_label in ambiguous_cases:
            result = detector.predict(text)
            print(f"Ambiguous: '{text}' -> {result.label} (score: {result.score:.3f})")
    
    def test_intensity_indicators(self):
        """Test detection of intensity indicators."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        intensity_cases = [
            # Repeated characters
            ("I'm sooooo depressed", "moderate_crisis"),
            ("I hate my lifeeeee", "moderate_crisis"),
            
            # Multiple punctuation
            ("I want to die!!!", "severe_crisis"),
            ("Help me... please...", "moderate_crisis"),
            
            # Caps
            ("I WANT TO DIE", "severe_crisis"),
            ("I'M HAVING A BREAKDOWN", "severe_crisis"),
            
            # Temporal intensity
            ("I always want to die", "severe_crisis"),
            ("I never feel happy", "moderate_crisis"),
            ("I constantly think about suicide", "severe_crisis"),
        ]
        
        for text, expected_label in intensity_cases:
            result = detector.predict(text)
            print(f"Intensity: '{text}' -> {result.label} (score: {result.score:.3f})")
            assert result.metadata['features']['intensity_penalty'] > 0
    
    def test_negation_patterns(self):
        """Test detection of negation patterns."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        negation_cases = [
            # High negation (concerning)
            ("I have nothing, nobody, nowhere to go", "severe_crisis"),
            ("I never feel anything, nothing matters", "severe_crisis"),
            ("I don't want to live, I don't care", "severe_crisis"),
            
            # Low negation (less concerning)
            ("I don't want to die", "safe"),
            ("I don't think about suicide", "safe"),
            ("I'm not depressed", "safe"),
        ]
        
        for text, expected_label in negation_cases:
            result = detector.predict(text)
            print(f"Negation: '{text}' -> {result.label} (score: {result.score:.3f})")
    
    def test_first_person_indicators(self):
        """Test detection of first-person indicators."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        first_person_cases = [
            # High first-person (concerning)
            ("I want to die, I hate myself, I can't go on", "severe_crisis"),
            ("I feel hopeless, I'm worthless, I want to end it", "severe_crisis"),
            
            # Low first-person (less concerning)
            ("People sometimes feel sad", "safe"),
            ("Life can be difficult", "safe"),
        ]
        
        for text, expected_label in first_person_cases:
            result = detector.predict(text)
            print(f"First person: '{text}' -> {result.label} (score: {result.score:.3f})")
    
    def test_safety_levels(self):
        """Test safety level classification."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        safety_level_cases = [
            ("I want to kill myself", SafetyLevel.CRITICAL),
            ("I feel hopeless and want to die", SafetyLevel.HIGH_RISK),
            ("I'm having a breakdown", SafetyLevel.MEDIUM_RISK),
            ("I'm feeling sad", SafetyLevel.LOW_RISK),
            ("Hello, how are you?", SafetyLevel.MINIMAL),
        ]
        
        for text, expected_level in safety_level_cases:
            result = detector.predict(text)
            print(f"Safety level: '{text}' -> {result.safety_level}")
            assert result.safety_level == expected_level
    
    def test_metadata_completeness(self):
        """Test that metadata contains all expected fields."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        result = detector.predict("I want to die")
        
        # Check metadata structure
        assert 'features' in result.metadata
        assert 'scores' in result.metadata
        assert 'rule_score' in result.metadata
        assert 'ml_score' in result.metadata
        assert 'multilingual' in result.metadata
        assert 'timestamp' in result.metadata
        
        # Check features
        features = result.metadata['features']
        assert 'immediate_threat_count' in features
        assert 'severe_distress_count' in features
        assert 'emotional_crisis_count' in features
        assert 'support_count' in features
        assert 'protective_count' in features
        
        # Check scores
        scores = result.metadata['scores']
        assert 'immediate_threat' in scores
        assert 'severe_distress' in scores
        assert 'emotional_crisis' in scores
        assert 'overall' in scores
    
    def test_performance_metrics(self):
        """Test performance on various metrics."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        # Test processing time
        import time
        start_time = time.time()
        
        for _ in range(100):
            detector.predict("I'm feeling sad today")
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        print(f"Average processing time: {avg_time:.4f} seconds")
        assert avg_time < 0.1  # Should be fast for rule-based model
    
    def test_error_handling(self):
        """Test error handling."""
        detector = AdvancedCrisisDetector(self.configs['rule_based'])
        
        # Test empty input
        result = detector.predict("")
        assert result.label == "safe"
        
        # Test very long input
        long_text = "I'm sad " * 1000
        result = detector.predict(long_text)
        assert result.label in ["safe", "mild_concern", "moderate_crisis"]
        
        # Test special characters
        result = detector.predict("!@#$%^&*()")
        assert result.label == "safe"


if __name__ == "__main__":
    pytest.main([__file__])
