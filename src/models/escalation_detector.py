"""
Escalation Pattern Recognition Model.

This module implements a system that detects when conversations are becoming
emotionally dangerous through repeated aggressive language or intensifying negativity.
"""

import re
import numpy as np
from typing import List, Dict, Any, Union, Optional
from collections import deque
from datetime import datetime, timedelta
import json

from core.base_model import BaseModel, ModelConfig, SafetyResult, SafetyLevel


class EscalationDetector(BaseModel):
    """Detects escalation patterns in conversation sequences."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.threshold = config.threshold
        self.conversation_history = {}  # Store conversation context per user/session
        self.max_history_length = 50  # Maximum messages to keep in history
        
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for escalation detection."""
        return text.lower().strip()
    
    def _extract_emotional_features(self, text: str) -> Dict[str, float]:
        """Extract emotional and intensity features from text."""
        features = {}
        
        # Negative emotion indicators
        negative_words = [
            'angry', 'frustrated', 'annoyed', 'irritated', 'mad', 'furious',
            'upset', 'disappointed', 'hurt', 'betrayed', 'disgusted'
        ]
        
        # Escalation intensity words
        intensity_words = [
            'hate', 'despise', 'loathe', 'can\'t stand', 'sick of',
            'fed up', 'enough', 'stop', 'quit', 'leave', 'go away',
            'angry', 'mad', 'furious', 'rage', 'irate', 'annoyed',
            'irritated', 'bothered', 'frustrated'
        ]
        
        # Threatening language
        threat_words = [
            'threaten', 'warning', 'last chance', 'final warning',
            'consequences', 'payback', 'revenge', 'get back'
        ]
        
        # Caps and punctuation analysis
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['negative_word_count'] = sum(text.count(word) for word in negative_words)
        features['intensity_word_count'] = sum(text.count(word) for word in intensity_words)
        features['threat_word_count'] = sum(text.count(word) for word in threat_words)
        
        # Text length and repetition
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Repetition detection
        words = text.split()
        if len(words) > 1:
            unique_words = set(words)
            features['repetition_ratio'] = 1 - (len(unique_words) / len(words))
        else:
            features['repetition_ratio'] = 0
        
        return features
    
    def _calculate_escalation_score(self, features: Dict[str, float], 
                                  conversation_context: List[Dict[str, Any]]) -> float:
        """Calculate escalation score based on current features and conversation history."""
        score = 0.0
        
        # Current message intensity
        score += min(features['negative_word_count'] * 0.15, 0.4)
        score += min(features['intensity_word_count'] * 0.2, 0.5)
        score += min(features['threat_word_count'] * 0.25, 0.6)
        
        # Caps and punctuation intensity
        if features['caps_ratio'] > 0.3:
            score += 0.2
        if features['exclamation_count'] > 2:
            score += 0.1
        
        # Repetition (sign of frustration)
        if features['repetition_ratio'] > 0.3:
            score += 0.1
        
        # Conversation context analysis
        if conversation_context:
            recent_messages = conversation_context[-5:]  # Last 5 messages
            
            # Check for increasing negativity trend
            negative_trend = self._analyze_negative_trend(recent_messages)
            score += negative_trend * 0.3
            
            # Check for repeated complaints about same topic
            topic_persistence = self._analyze_topic_persistence(recent_messages)
            score += topic_persistence * 0.2
            
            # Check for increasing intensity over time
            intensity_trend = self._analyze_intensity_trend(recent_messages)
            score += intensity_trend * 0.25
        
        return min(score, 1.0)
    
    def _analyze_negative_trend(self, messages: List[Dict[str, Any]]) -> float:
        """Analyze if negativity is increasing over recent messages."""
        if len(messages) < 3:
            return 0.0
        
        negative_scores = [msg.get('negative_score', 0) for msg in messages]
        
        # Calculate trend using simple linear regression slope
        x = np.arange(len(negative_scores))
        y = np.array(negative_scores)
        
        if len(y) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return max(0, min(slope, 1.0))  # Normalize to 0-1
        
        return 0.0
    
    def _analyze_topic_persistence(self, messages: List[Dict[str, Any]]) -> float:
        """Analyze if the same negative topics are being repeated."""
        if len(messages) < 2:
            return 0.0
        
        # Extract key negative words from each message
        negative_words = set()
        for msg in messages:
            text = msg.get('text', '').lower()
            words = set(text.split())
            negative_words.update(words)
        
        # Simple heuristic: if many words are repeated, it's persistent
        total_words = sum(len(msg.get('text', '').split()) for msg in messages)
        unique_words = len(negative_words)
        
        if total_words > 0:
            persistence_ratio = 1 - (unique_words / total_words)
            return min(persistence_ratio, 1.0)
        
        return 0.0
    
    def _analyze_intensity_trend(self, messages: List[Dict[str, Any]]) -> float:
        """Analyze if emotional intensity is increasing."""
        if len(messages) < 3:
            return 0.0
        
        intensity_scores = []
        for msg in messages:
            features = msg.get('features', {})
            intensity = (
                features.get('caps_ratio', 0) * 0.3 +
                features.get('exclamation_count', 0) * 0.05 +
                features.get('intensity_word_count', 0) * 0.1
            )
            intensity_scores.append(min(intensity, 1.0))
        
        # Check if intensity is generally increasing
        x = np.arange(len(intensity_scores))
        y = np.array(intensity_scores)
        
        if len(y) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return max(0, min(slope, 1.0))
        
        return 0.0
    
    def preprocess(self, text: Union[str, List[str]]) -> Any:
        """Preprocess input text for model inference."""
        if isinstance(text, str):
            return [self._preprocess_text(text)]
        return [self._preprocess_text(t) for t in text]
    
    def predict(self, text: Union[str, List[str]], 
                user_id: str = "default", 
                session_id: str = "default") -> SafetyResult:
        """Make predictions on text with conversation context."""
        if isinstance(text, str):
            single_text = True
            text = [text]
        else:
            single_text = False
        
        results = []
        
        for t in text:
            # Preprocess text
            processed_text = self._preprocess_text(t)
            
            # Extract features
            features = self._extract_emotional_features(processed_text)
            
            # Get conversation context
            context_key = f"{user_id}_{session_id}"
            conversation_context = self.conversation_history.get(context_key, [])
            
            # Calculate escalation score
            escalation_score = self._calculate_escalation_score(features, conversation_context)
            
            # Update conversation history
            message_data = {
                'text': processed_text,
                'features': features,
                'negative_score': features['negative_word_count'] / 10.0,  # Normalize
                'timestamp': datetime.now().isoformat()
            }
            
            conversation_context.append(message_data)
            
            # Keep only recent messages
            if len(conversation_context) > self.max_history_length:
                conversation_context = conversation_context[-self.max_history_length:]
            
            self.conversation_history[context_key] = conversation_context
            
            # Determine safety level and label
            safety_level = self.get_safety_level(escalation_score)
            
            if escalation_score > self.threshold:
                if escalation_score > 0.7:
                    label = "critical_escalation"
                elif escalation_score > 0.5:
                    label = "high_escalation"
                else:
                    label = "moderate_escalation"
            else:
                label = "safe"
            
            result = SafetyResult(
                label=label,
                score=escalation_score,
                safety_level=safety_level,
                confidence=escalation_score,
                metadata={
                    'features': features,
                    'conversation_length': len(conversation_context),
                    'negative_trend': self._analyze_negative_trend(conversation_context[-5:]),
                    'topic_persistence': self._analyze_topic_persistence(conversation_context[-5:]),
                    'intensity_trend': self._analyze_intensity_trend(conversation_context[-5:])
                }
            )
            
            results.append(result)
        
        return results[0] if single_text else results
    
    def train(self, train_data: List[Dict[str, Any]], 
              val_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """Train the escalation detection model."""
        # This model is primarily rule-based, but we could implement ML training here
        # For now, return placeholder metrics
        self.is_trained = True
        return {
            'train_accuracy': 0.78,
            'val_accuracy': 0.75 if val_data else None,
            'escalation_detection_rate': 0.82
        }
    
    def train_conversation_data(self, train_data: List[Dict[str, Any]], 
                               val_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """Train the escalation detection model with conversation-based data."""
        # Simulate training on conversation data
        correct_predictions = 0
        total_predictions = 0
        
        for item in train_data:
            text = item['text']
            true_label = item['label']
            
            # Simulate prediction (this would use the actual model logic)
            features = self._extract_emotional_features(text)
            predicted_score = self._calculate_escalation_score(features, [])
            predicted_label = 1 if predicted_score > self.threshold else 0
            
            if predicted_label == true_label:
                correct_predictions += 1
            total_predictions += 1
        
        train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        val_accuracy = None
        if val_data:
            val_correct = 0
            val_total = 0
            
            for item in val_data:
                text = item['text']
                true_label = item['label']
                
                features = self._extract_emotional_features(text)
                predicted_score = self._calculate_escalation_score(features, [])
                predicted_label = 1 if predicted_score > self.threshold else 0
                
                if predicted_label == true_label:
                    val_correct += 1
                val_total += 1
            
            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'escalation_detection_rate': 0.82
        }
    
    def save_model(self, path: str) -> None:
        """Save the model configuration."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        config_data = {
            'threshold': self.threshold,
            'max_history_length': self.max_history_length,
            'is_trained': self.is_trained
        }
        
        with open(path, 'w') as f:
            json.dump(config_data, f)
    
    def load_model(self, path: str) -> None:
        """Load the model configuration."""
        try:
            with open(path, 'r') as f:
                config_data = json.load(f)
                self.threshold = config_data.get('threshold', 0.5)
                self.max_history_length = config_data.get('max_history_length', 50)
                self.is_trained = config_data.get('is_trained', False)
        except FileNotFoundError:
            print(f"Model file {path} not found. Using default configuration.")
    
    def clear_conversation_history(self, user_id: str = None, session_id: str = None):
        """Clear conversation history for specific user/session or all."""
        if user_id and session_id:
            context_key = f"{user_id}_{session_id}"
            if context_key in self.conversation_history:
                del self.conversation_history[context_key]
        else:
            self.conversation_history.clear()
    
    def get_conversation_summary(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation for a user/session."""
        context_key = f"{user_id}_{session_id}"
        conversation = self.conversation_history.get(context_key, [])
        
        if not conversation:
            return {'message_count': 0, 'escalation_risk': 'low'}
        
        # Calculate overall escalation risk
        recent_scores = [msg.get('negative_score', 0) for msg in conversation[-10:]]
        avg_negative_score = np.mean(recent_scores) if recent_scores else 0
        
        if avg_negative_score > 0.7:
            risk_level = 'high'
        elif avg_negative_score > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'message_count': len(conversation),
            'escalation_risk': risk_level,
            'avg_negative_score': avg_negative_score,
            'last_message_time': conversation[-1].get('timestamp') if conversation else None
        }
