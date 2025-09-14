"""
Intervention handler for AI safety models.
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from enum import Enum

# Add src to path for imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class InterventionType(Enum):
    """Types of interventions."""
    WARNING = "warning"
    CONTENT_BLOCK = "content_block"
    HUMAN_REVIEW = "human_review"
    CRISIS_INTERVENTION = "crisis_intervention"


class InterventionHandler:
    """Handler for safety interventions."""
    
    def __init__(self):
        """Initialize the intervention handler."""
        self.intervention_history = []
        self.crisis_contacts = {
            'crisis_hotline': '1-800-273-8255',
            'crisis_text': 'Text HOME to 741741'
        }
    
    def handle_intervention(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle intervention based on analysis result."""
        overall_assessment = analysis_result['overall_assessment']
        overall_risk = overall_assessment['overall_risk']
        
        intervention_result = {
            'intervention_required': False,
            'intervention_type': None,
            'actions_taken': [],
            'recommendations': [],
            'timestamp': datetime.now().isoformat()
        }
        
        if overall_risk in ['critical', 'high']:
            intervention_result['intervention_required'] = True
            intervention_result['intervention_type'] = InterventionType.CRISIS_INTERVENTION.value
            intervention_result['actions_taken'].append('crisis_intervention_triggered')
            intervention_result['recommendations'].append('Provide crisis resources')
        
        elif overall_risk == 'medium':
            intervention_result['intervention_required'] = True
            intervention_result['intervention_type'] = InterventionType.HUMAN_REVIEW.value
            intervention_result['recommendations'].append('Human review recommended')
        
        self.intervention_history.append(intervention_result)
        return intervention_result
    
    def get_intervention_history(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get intervention history."""
        if user_id:
            return [i for i in self.intervention_history if i.get('user_id') == user_id]
        return self.intervention_history