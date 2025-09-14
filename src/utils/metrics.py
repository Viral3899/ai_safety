"""
Metrics utilities for AI Safety Models.

This module provides utility functions for calculating and analyzing metrics.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricsCalculator:
    """Metrics calculator for AI Safety Models."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def calculate_basic_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
    
    def calculate_confidence_intervals(self, metrics: Dict[str, float], 
                                     n_samples: int, 
                                     confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for metrics."""
        z_score = 1.96 if confidence == 0.95 else 2.58  # 95% or 99%
        
        intervals = {}
        for metric_name, metric_value in metrics.items():
            if metric_name == 'accuracy':
                se = np.sqrt(metric_value * (1 - metric_value) / n_samples)
                margin = z_score * se
                intervals[metric_name] = (
                    max(0, metric_value - margin),
                    min(1, metric_value + margin)
                )
            else:
                # Simplified for other metrics
                se = np.sqrt(metric_value * (1 - metric_value) / n_samples)
                margin = z_score * se
                intervals[metric_name] = (
                    max(0, metric_value - margin),
                    min(1, metric_value + margin)
                )
        
        return intervals
    
    def calculate_statistical_significance(self, metrics1: Dict[str, float],
                                         metrics2: Dict[str, float],
                                         n1: int, n2: int) -> Dict[str, Dict[str, float]]:
        """Calculate statistical significance between two sets of metrics."""
        significance_results = {}
        
        for metric_name in metrics1.keys():
            if metric_name in metrics2:
                m1, m2 = metrics1[metric_name], metrics2[metric_name]
                
                # Simplified z-test
                se1 = np.sqrt(m1 * (1 - m1) / n1)
                se2 = np.sqrt(m2 * (1 - m2) / n2)
                se_diff = np.sqrt(se1**2 + se2**2)
                
                z_score = (m1 - m2) / se_diff if se_diff > 0 else 0
                p_value = 2 * (1 - self._normal_cdf(abs(z_score)))
                
                significance_results[metric_name] = {
                    'z_score': z_score,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': abs(m1 - m2)
                }
        
        return significance_results
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def calculate_model_comparison(self, model_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare performance across multiple models."""
        comparison = {
            'best_model': None,
            'rankings': {},
            'statistical_tests': {}
        }
        
        # Find best model by F1 score
        if model_results:
            best_model = max(model_results.items(), key=lambda x: x[1].get('f1_score', 0))
            comparison['best_model'] = {
                'name': best_model[0],
                'f1_score': best_model[1]['f1_score']
            }
            
            # Create rankings
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                sorted_models = sorted(
                    model_results.items(),
                    key=lambda x: x[1].get(metric, 0),
                    reverse=True
                )
                comparison['rankings'][metric] = {
                    model: metrics.get(metric, 0) for model, metrics in sorted_models
                }
        
        return comparison
    
    def generate_performance_summary(self, metrics: Dict[str, float],
                                   n_samples: int) -> Dict[str, Any]:
        """Generate comprehensive performance summary."""
        confidence_intervals = self.calculate_confidence_intervals(metrics, n_samples)
        
        return {
            'metrics': metrics,
            'confidence_intervals': confidence_intervals,
            'sample_size': n_samples,
            'performance_level': self._assess_performance_level(metrics['f1_score']),
            'recommendations': self._generate_recommendations(metrics)
        }
    
    def _assess_performance_level(self, f1_score: float) -> str:
        """Assess performance level based on F1 score."""
        if f1_score >= 0.9:
            return 'excellent'
        elif f1_score >= 0.8:
            return 'good'
        elif f1_score >= 0.7:
            return 'fair'
        elif f1_score >= 0.6:
            return 'poor'
        else:
            return 'very_poor'
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        if metrics['precision'] < 0.7:
            recommendations.append("Consider reducing false positives by adjusting threshold")
        
        if metrics['recall'] < 0.7:
            recommendations.append("Consider improving sensitivity to reduce false negatives")
        
        if metrics['f1_score'] < 0.7:
            recommendations.append("Overall performance needs improvement - consider model retraining")
        
        if not recommendations:
            recommendations.append("Performance is acceptable - continue monitoring")
        
        return recommendations