"""
Safety Manager - Central coordinator for AI Safety Models.

This module integrates all safety models into a cohesive system that provides
comprehensive safety analysis and intervention recommendations.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json

from core.base_model import ModelConfig, SafetyResult, SafetyLevel
from models.abuse_detector import AdvancedAbuseDetector as AbuseDetector
from models.escalation_detector import EscalationDetector
from models.crisis_detector import AdvancedCrisisDetector as CrisisDetector
from models.content_filter import ContentFilter, AgeGroup


class SafetyManager:
    """Central coordinator for all AI Safety Models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Safety Manager with all safety models."""
        self.config = config or self._get_default_config()
        
        # Initialize all safety models
        self.models = {}
        self._initialize_models()
        
        # Safety thresholds
        self.thresholds = {
            'abuse': 0.5,
            'escalation': 0.4,
            'crisis': 0.3,
            'content_filter': 0.6
        }
        
        # Intervention levels
        self.intervention_levels = {
            'none': 0,
            'monitor': 1,
            'warn': 2,
            'intervene': 3,
            'emergency': 4
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for all models."""
        return {
            'abuse_detector': {
                'model_type': 'sklearn',
                'threshold': 0.5,
                'device': 'cpu'
            },
            'escalation_detector': {
                'model_type': 'rule_based',
                'threshold': 0.4,
                'device': 'cpu'
            },
            'crisis_detector': {
                'model_type': 'rule_based',
                'threshold': 0.3,
                'device': 'cpu'
            },
            'content_filter': {
                'model_type': 'rule_based',
                'threshold': 0.6,
                'device': 'cpu'
            }
        }
    
    def _initialize_models(self):
        """Initialize all safety models with their configurations."""
        try:
            # Abuse Detector
            abuse_config = ModelConfig(**self.config['abuse_detector'])
            self.models['abuse_detector'] = AbuseDetector(abuse_config)
            
            # Escalation Detector
            escalation_config = ModelConfig(**self.config['escalation_detector'])
            self.models['escalation_detector'] = EscalationDetector(escalation_config)
            
            # Crisis Detector
            crisis_config = ModelConfig(**self.config['crisis_detector'])
            self.models['crisis_detector'] = CrisisDetector(crisis_config)
            
            # Content Filter
            content_config = ModelConfig(**self.config['content_filter'])
            self.models['content_filter'] = ContentFilter(content_config)
            
        except Exception as e:
            print(f"Error initializing safety models: {e}")
            # Fallback to basic initialization
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize fallback models if main initialization fails."""
        # Create basic configs for fallback
        basic_config = ModelConfig(
            model_type='rule_based',
            threshold=0.5,
            device='cpu'
        )
        
        try:
            self.models['abuse_detector'] = AbuseDetector(basic_config)
            self.models['escalation_detector'] = EscalationDetector(basic_config)
            self.models['crisis_detector'] = CrisisDetector(basic_config)
            self.models['content_filter'] = ContentFilter(basic_config)
        except Exception as e:
            print(f"Error in fallback initialization: {e}")
    
    def analyze(self, text: Union[str, List[str]], 
                user_id: str = "default",
                session_id: str = "default",
                age_group: AgeGroup = AgeGroup.ADULT,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive safety analysis of text using all models.
        
        Args:
            text: Input text to analyze
            user_id: User identifier for conversation tracking
            session_id: Session identifier for conversation tracking
            age_group: Age group for content filtering
            context: Additional context information
            
        Returns:
            Comprehensive safety analysis results
        """
        if isinstance(text, str):
            text = [text]
        
        # Start timing
        import time
        start_time = time.time()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'session_id': session_id,
            'age_group': age_group.value,
            'context': context or {},
            'models': {},
            'overall_assessment': {},
            'intervention_recommendations': []
        }
        
        # Run all model analyses
        model_results = {}
        
        try:
            # Abuse Detection
            abuse_result = self.models['abuse_detector'].predict(text[0])
            model_results['abuse'] = {
                'result': abuse_result,
                'risk_level': self._assess_risk_level(abuse_result, 'abuse')
            }
            
            # Escalation Detection (requires user/session context)
            escalation_result = self.models['escalation_detector'].predict(
                text[0], user_id=user_id, session_id=session_id
            )
            model_results['escalation'] = {
                'result': escalation_result,
                'risk_level': self._assess_risk_level(escalation_result, 'escalation')
            }
            
            # Crisis Detection
            crisis_result = self.models['crisis_detector'].predict(text[0])
            model_results['crisis'] = {
                'result': crisis_result,
                'risk_level': self._assess_risk_level(crisis_result, 'crisis')
            }
            
            # Content Filtering
            content_result = self.models['content_filter'].predict(text[0], age_group=age_group)
            model_results['content_filter'] = {
                'result': content_result,
                'risk_level': self._assess_risk_level(content_result, 'content_filter')
            }
            
        except Exception as e:
            print(f"Error during model analysis: {e}")
            # Return basic error response
            return self._create_error_response(str(e))
        
        results['models'] = model_results
        
        # Generate overall assessment
        overall_assessment = self._generate_overall_assessment(model_results)
        results['overall_assessment'] = overall_assessment
        
        # Generate intervention recommendations
        intervention_recommendations = self._generate_intervention_recommendations(
            model_results, overall_assessment
        )
        results['intervention_recommendations'] = intervention_recommendations
        
        # Calculate processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        return results
    
    def _assess_risk_level(self, result: SafetyResult, model_type: str) -> str:
        """Assess risk level based on model result."""
        threshold = self.thresholds.get(model_type, 0.5)
        
        if result.score >= threshold:
            if result.safety_level == SafetyLevel.CRITICAL:
                return 'critical'
            elif result.safety_level == SafetyLevel.HIGH_RISK:
                return 'high'
            elif result.safety_level == SafetyLevel.MEDIUM_RISK:
                return 'medium'
            else:
                return 'low'
        else:
            return 'minimal'
    
    def _generate_overall_assessment(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall safety assessment from all model results."""
        # Calculate overall risk score
        risk_scores = []
        critical_issues = []
        high_risk_issues = []
        
        for model_name, model_result in model_results.items():
            result = model_result['result']
            risk_level = model_result['risk_level']
            
            risk_scores.append(result.score)
            
            if risk_level == 'critical':
                critical_issues.append(model_name)
            elif risk_level == 'high':
                high_risk_issues.append(model_name)
            elif risk_level == 'medium' and result.score > 0.5:
                # Treat medium risk with high scores as high risk for severe cases
                high_risk_issues.append(model_name)
        
        # Determine overall risk level
        max_score = max(risk_scores) if risk_scores else 0.0
        avg_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        
        if critical_issues:
            overall_risk = 'critical'
            intervention_level = 'emergency'
        elif high_risk_issues:
            overall_risk = 'high'
            intervention_level = 'intervene'
        elif max_score > 0.5:
            overall_risk = 'medium'
            intervention_level = 'warn'
        elif avg_score > 0.3:
            overall_risk = 'low'
            intervention_level = 'monitor'
        else:
            overall_risk = 'minimal'
            intervention_level = 'none'
        
        return {
            'overall_risk': overall_risk,
            'intervention_level': intervention_level,
            'max_score': max_score,
            'average_score': avg_score,
            'critical_issues': critical_issues,
            'high_risk_issues': high_risk_issues,
            'risk_distribution': {
                model_name: model_result['risk_level'] 
                for model_name, model_result in model_results.items()
            }
        }
    
    def _generate_intervention_recommendations(self, 
                                             model_results: Dict[str, Any],
                                             overall_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate intervention recommendations based on analysis results."""
        recommendations = []
        
        intervention_level = overall_assessment['intervention_level']
        
        # Emergency interventions
        if intervention_level == 'emergency':
            recommendations.extend([
                {
                    'type': 'immediate_action',
                    'priority': 'critical',
                    'action': 'Contact emergency services or crisis intervention team',
                    'reason': 'Critical safety issues detected',
                    'models_triggered': overall_assessment['critical_issues']
                },
                {
                    'type': 'content_blocking',
                    'priority': 'critical',
                    'action': 'Block content and prevent further interaction',
                    'reason': 'Content poses immediate safety risk',
                    'models_triggered': overall_assessment['critical_issues']
                }
            ])
        
        # High-risk interventions
        elif intervention_level == 'intervene':
            recommendations.extend([
                {
                    'type': 'human_review',
                    'priority': 'high',
                    'action': 'Flag for immediate human moderator review',
                    'reason': 'High-risk content detected',
                    'models_triggered': overall_assessment['high_risk_issues']
                },
                {
                    'type': 'content_warning',
                    'priority': 'high',
                    'action': 'Show content warning before displaying',
                    'reason': 'Content may be inappropriate or harmful',
                    'models_triggered': overall_assessment['high_risk_issues']
                }
            ])
        
        # Medium-risk interventions
        elif intervention_level == 'warn':
            recommendations.extend([
                {
                    'type': 'automated_warning',
                    'priority': 'medium',
                    'action': 'Display automated safety warning',
                    'reason': 'Content flagged by safety models',
                    'models_triggered': [k for k, v in model_results.items() 
                                       if v['risk_level'] in ['medium', 'high']]
                }
            ])
        
        # Low-risk interventions
        elif intervention_level == 'monitor':
            recommendations.append({
                'type': 'enhanced_logging',
                'priority': 'low',
                'action': 'Log interaction for potential pattern analysis',
                'reason': 'Minor safety concerns detected',
                'models_triggered': [k for k, v in model_results.items() 
                                   if v['risk_level'] == 'low']
            })
        
        return recommendations
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response when analysis fails."""
        return {
            'timestamp': datetime.now().isoformat(),
            'error': True,
            'error_message': error_message,
            'models': {},
            'overall_assessment': {
                'overall_risk': 'unknown',
                'intervention_level': 'none',
                'error': True
            },
            'intervention_recommendations': [
                {
                    'type': 'system_error',
                    'priority': 'high',
                    'action': 'Manual review required due to system error',
                    'reason': error_message,
                    'models_triggered': []
                }
            ]
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status information for all models."""
        status = {}
        
        for model_name, model in self.models.items():
            try:
                status[model_name] = {
                    'initialized': True,
                    'trained': getattr(model, 'is_trained', False),
                    'model_type': getattr(model, 'model_type', 'unknown'),
                    'threshold': getattr(model, 'threshold', 'unknown')
                }
            except Exception as e:
                status[model_name] = {
                    'initialized': False,
                    'error': str(e)
                }
        
        return status
    
    def get_models_status(self) -> Dict[str, Any]:
        """Get status information for all models (alias for get_model_status)."""
        return self.get_model_status()
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update safety thresholds for models."""
        self.thresholds.update(new_thresholds)
        
        # Update model thresholds
        for model_name, threshold in new_thresholds.items():
            if model_name in self.models:
                self.models[model_name].threshold = threshold
    
    def clear_conversation_history(self, user_id: str = None, session_id: str = None):
        """Clear conversation history for escalation detector."""
        if 'escalation_detector' in self.models:
            self.models['escalation_detector'].clear_conversation_history(user_id, session_id)
    
    def get_conversation_summary(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get conversation summary for a user/session."""
        if 'escalation_detector' in self.models:
            return self.models['escalation_detector'].get_conversation_summary(user_id, session_id)
        return {'error': 'Escalation detector not available'}
    
    def save_models(self, base_path: str) -> None:
        """Save all trained models."""
        for model_name, model in self.models.items():
            try:
                model_path = f"{base_path}/{model_name}.pkl"
                model.save_model(model_path)
            except Exception as e:
                print(f"Error saving {model_name}: {e}")
    
    def load_models(self, base_path: str) -> None:
        """Load all trained models."""
        for model_name, model in self.models.items():
            try:
                model_path = f"{base_path}/{model_name}.pkl"
                model.load_model(model_path)
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
    
    # Convenience methods for individual model access
    def detect_abuse(self, text: str) -> SafetyResult:
        """Detect abuse in text."""
        return self.models['abuse_detector'].predict(text)
    
    def detect_crisis(self, text: str) -> SafetyResult:
        """Detect crisis indicators in text."""
        return self.models['crisis_detector'].predict(text)
    
    def detect_escalation(self, text: str, user_id: str = "default", session_id: str = "default") -> SafetyResult:
        """Detect escalation patterns in text."""
        return self.models['escalation_detector'].predict(text, user_id=user_id, session_id=session_id)
    
    def filter_content(self, text: str, age_group: str = "adult") -> SafetyResult:
        """Filter content based on age group."""
        # Convert string to AgeGroup enum
        age_group_enum = AgeGroup.ADULT
        if age_group.lower() == 'child':
            age_group_enum = AgeGroup.CHILD
        elif age_group.lower() == 'teen':
            age_group_enum = AgeGroup.TEEN
        elif age_group.lower() == 'adult':
            age_group_enum = AgeGroup.ADULT
        
        return self.models['content_filter'].predict(text, age_group=age_group_enum)