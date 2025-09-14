"""
Performance metrics for AI safety models.
"""

import numpy as np
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)


class PerformanceMetrics:
    """Comprehensive performance metrics calculator for AI Safety Models."""
    
    def __init__(self):
        """Initialize the performance metrics calculator."""
        self.metrics_history = []
    
    def calculate_basic_metrics(self, y_true: List[int], y_pred: List[int], 
                               y_scores: List[float] = None) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        # Add AUC if scores are provided
        if y_scores is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            except ValueError:
                metrics['auc_roc'] = 0.0
        
        return metrics
    
    def calculate_confusion_matrix_metrics(self, y_true: List[int], 
                                         y_pred: List[int]) -> Dict[str, Any]:
        """Calculate confusion matrix and related metrics."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Handle different matrix sizes
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        metrics = {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
        
        return metrics
    
    def calculate_threshold_metrics(self, y_true: List[int], 
                                   y_scores: List[float]) -> Dict[str, Any]:
        """Calculate metrics across different thresholds."""
        if not y_scores:
            return {}
        
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred = [1 if score >= threshold else 0 for score in y_scores]
            
            metrics = {
                'threshold': float(threshold),
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
            }
            threshold_metrics.append(metrics)
        
        # Find optimal threshold (highest F1 score)
        optimal_threshold = max(threshold_metrics, key=lambda x: x['f1_score'])
        
        return {
            'threshold_analysis': threshold_metrics,
            'optimal_threshold': optimal_threshold['threshold'],
            'optimal_f1': optimal_threshold['f1_score']
        }
    
    def calculate_category_metrics(self, y_true: List[int], y_pred: List[int],
                                  categories: List[str]) -> Dict[str, Any]:
        """Calculate metrics by category."""
        category_metrics = {}
        
        for category in set(categories):
            # Get indices for this category
            cat_indices = [i for i, cat in enumerate(categories) if cat == category]
            
            if not cat_indices:
                continue
            
            # Get predictions and labels for this category
            cat_y_true = [y_true[i] for i in cat_indices]
            cat_y_pred = [y_pred[i] for i in cat_indices]
            
            if len(set(cat_y_true)) < 2:  # Skip if only one class
                continue
            
            metrics = self.calculate_basic_metrics(cat_y_true, cat_y_pred)
            metrics['sample_count'] = len(cat_indices)
            
            category_metrics[category] = metrics
        
        return category_metrics
    
    def calculate_model_comparison(self, models_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance across multiple models."""
        comparison = {
            'models': {},
            'best_model': None,
            'rankings': {}
        }
        
        # Calculate metrics for each model
        for model_name, results in models_results.items():
            if 'y_true' in results and 'y_pred' in results:
                metrics = self.calculate_basic_metrics(
                    results['y_true'], 
                    results['y_pred'],
                    results.get('y_scores')
                )
                comparison['models'][model_name] = metrics
        
        # Find best model by F1 score
        if comparison['models']:
            best_model = max(comparison['models'].items(), key=lambda x: x[1]['f1_score'])
            comparison['best_model'] = {
                'name': best_model[0],
                'f1_score': best_model[1]['f1_score']
            }
            
            # Create rankings
            sorted_models = sorted(
                comparison['models'].items(), 
                key=lambda x: x[1]['f1_score'], 
                reverse=True
            )
            comparison['rankings'] = {
                metric: {model: metrics[metric] for model, metrics in sorted_models}
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']
            }
        
        return comparison
    
    def generate_performance_report(self, y_true: List[int], y_pred: List[int],
                                   y_scores: List[float] = None,
                                   categories: List[str] = None,
                                   model_name: str = "Model") -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(y_true)
        }
        
        # Basic metrics
        report['basic_metrics'] = self.calculate_basic_metrics(y_true, y_pred, y_scores)
        
        # Confusion matrix metrics
        report['confusion_matrix'] = self.calculate_confusion_matrix_metrics(y_true, y_pred)
        
        # Threshold analysis
        if y_scores:
            report['threshold_analysis'] = self.calculate_threshold_metrics(y_true, y_scores)
        
        # Category metrics
        if categories:
            report['category_metrics'] = self.calculate_category_metrics(y_true, y_pred, categories)
        
        # Classification report
        try:
            report['classification_report'] = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
        except Exception as e:
            report['classification_report'] = {'error': str(e)}
        
        return report
    
    def save_metrics(self, report: Dict[str, Any], output_path: str):
        """Save performance metrics to file."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def load_metrics(self, input_path: str) -> Dict[str, Any]:
        """Load performance metrics from file."""
        with open(input_path, 'r') as f:
            return json.load(f)
    
    def compare_with_baseline(self, current_metrics: Dict[str, float],
                             baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare current model performance with baseline."""
        comparison = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in current_metrics and metric in baseline_metrics:
                current_val = current_metrics[metric]
                baseline_val = baseline_metrics[metric]
                
                improvement = current_val - baseline_val
                relative_improvement = (improvement / baseline_val * 100) if baseline_val > 0 else 0
                
                comparison[metric] = {
                    'current': current_val,
                    'baseline': baseline_val,
                    'improvement': improvement,
                    'relative_improvement_percent': relative_improvement,
                    'better': improvement > 0
                }
        
        return comparison


def calculate_ensemble_metrics(models_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate ensemble performance metrics."""
    if not models_results:
        return {}
    
    # Extract predictions and scores
    all_y_true = models_results[0]['y_true']
    all_y_pred = []
    all_y_scores = []
    
    for result in models_results:
        all_y_pred.append(result['y_pred'])
        all_y_scores.append(result.get('y_scores', []))
    
    # Simple voting ensemble
    ensemble_pred = []
    for i in range(len(all_y_true)):
        votes = [pred[i] for pred in all_y_pred]
        ensemble_pred.append(1 if sum(votes) > len(votes) / 2 else 0)
    
    # Average scores
    ensemble_scores = []
    if all_y_scores and all_y_scores[0]:
        for i in range(len(all_y_true)):
            scores = [score[i] for score in all_y_scores]
            ensemble_scores.append(np.mean(scores))
    
    # Calculate ensemble metrics
    metrics_calc = PerformanceMetrics()
    ensemble_report = metrics_calc.generate_performance_report(
        all_y_true, ensemble_pred, ensemble_scores, model_name="Ensemble"
    )
    
    return ensemble_report


def benchmark_performance(model_func, test_data: List[Dict[str, Any]], 
                         iterations: int = 100) -> Dict[str, Any]:
    """Benchmark model performance across multiple iterations."""
    times = []
    accuracies = []
    
    for _ in range(iterations):
        start_time = datetime.now()
        
        # Run model inference
        predictions = []
        for item in test_data:
            result = model_func(item['text'])
            predictions.append(1 if result.score > 0.5 else 0)
        
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds()
        
        # Calculate accuracy
        true_labels = [item['label'] for item in test_data]
        accuracy = accuracy_score(true_labels, predictions)
        
        times.append(inference_time)
        accuracies.append(accuracy)
    
    return {
        'avg_inference_time': np.mean(times),
        'std_inference_time': np.std(times),
        'avg_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'iterations': iterations,
        'samples_per_iteration': len(test_data)
    }