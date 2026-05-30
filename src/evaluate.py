"""
Evaluation script untuk PII Shield.
Menghitung metrics: precision, recall, F1-score (token-level dan entity-level).
"""

import json
import csv
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class Evaluator:
    """Evaluate token classification dan NER."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    @staticmethod
    def extract_entities(labels: List[str]) -> List[Tuple[str, int, int]]:
        """
        Extract entities dari BIO labels.
        
        Returns:
            List of (entity_type, start_idx, end_idx)
        """
        entities = []
        current_entity = None
        current_start = None
        
        for idx, label in enumerate(labels):
            if label == 'O':
                if current_entity is not None:
                    entities.append((current_entity, current_start, idx))
                    current_entity = None
            else:
                entity_type = label.split('-')[1]
                label_type = label.split('-')[0]
                
                if label_type == 'B' or entity_type != current_entity:
                    if current_entity is not None:
                        entities.append((current_entity, current_start, idx))
                    current_entity = entity_type
                    current_start = idx
        
        if current_entity is not None:
            entities.append((current_entity, current_start, len(labels)))
        
        return entities
    
    @staticmethod
    def token_level_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
        """
        Calculate token-level metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dict with precision, recall, f1, per_class_f1
        """
        assert len(y_true) == len(y_pred), "Length mismatch!"
        
        # Get unique labels (exclude O)
        all_labels = set(y_true + y_pred) - {'O'}
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        per_class_metrics = {}
        
        for label in sorted(all_labels):
            tp = sum((y_true[i] == label and y_pred[i] == label) for i in range(len(y_true)))
            fp = sum((y_true[i] != label and y_pred[i] == label) for i in range(len(y_true)))
            fn = sum((y_true[i] == label and y_pred[i] != label) for i in range(len(y_true)))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[label] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Macro metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'per_class': per_class_metrics
        }
    
    @staticmethod
    def entity_level_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
        """
        Calculate entity-level metrics (exact match).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dict with entity-level precision, recall, f1
        """
        entities_true = Evaluator.extract_entities(y_true)
        entities_pred = Evaluator.extract_entities(y_pred)
        
        entities_true_set = set(entities_true)
        entities_pred_set = set(entities_pred)
        
        tp = len(entities_true_set & entities_pred_set)
        fp = len(entities_pred_set - entities_true_set)
        fn = len(entities_true_set - entities_pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'total_tp': tp,
            'total_fp': fp,
            'total_fn': fn
        }
    
    @staticmethod
    def evaluate(y_true: List[str], y_pred: List[str]) -> Dict:
        """
        Evaluate model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dict with token-level dan entity-level metrics
        """
        token_metrics = Evaluator.token_level_metrics(y_true, y_pred)
        entity_metrics = Evaluator.entity_level_metrics(y_true, y_pred)
        
        return {
            'token_level': token_metrics,
            'entity_level': entity_metrics
        }


def evaluate_from_csv(csv_path: str, model_name: str = "unknown") -> Dict:
    """
    Evaluate model dari CSV predictions (format: document_id, token, true_label, pred_label).
    
    Args:
        csv_path: Path ke CSV file
        model_name: Model name untuk output
        
    Returns:
        Dict with metrics
    """
    y_true = []
    y_pred = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(row['true_label'])
            y_pred.append(row['pred_label'])
    
    metrics = Evaluator.evaluate(y_true, y_pred)
    metrics['model'] = model_name
    metrics['total_tokens'] = len(y_true)
    
    return metrics


def save_metrics(metrics: Dict, output_path: str) -> None:
    """
    Save metrics sebagai JSON.
    
    Args:
        metrics: Metrics dict
        output_path: Output file path
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def print_metrics(metrics: Dict) -> None:
    """
    Print metrics dalam format yang readable.
    
    Args:
        metrics: Metrics dict
    """
    print(f"\n{'='*70}")
    print(f"Model: {metrics.get('model', 'Unknown')}")
    print(f"Total tokens: {metrics.get('total_tokens', 'N/A')}")
    print(f"{'='*70}")
    
    # Token level
    print(f"\n{'TOKEN-LEVEL METRICS':^70}")
    print(f"{'-'*70}")
    token = metrics['token_level']
    print(f"Precision: {token['precision']:.4f}")
    print(f"Recall:    {token['recall']:.4f}")
    print(f"F1-Score:  {token['f1']:.4f}")
    
    print(f"\nPer-class F1-scores:")
    for label, metrics_dict in token['per_class'].items():
        print(f"  {label:20s}: P={metrics_dict['precision']:.4f}, R={metrics_dict['recall']:.4f}, F1={metrics_dict['f1']:.4f}")
    
    # Entity level
    print(f"\n{'ENTITY-LEVEL METRICS':^70}")
    print(f"{'-'*70}")
    entity = metrics['entity_level']
    print(f"Precision: {entity['precision']:.4f}")
    print(f"Recall:    {entity['recall']:.4f}")
    print(f"F1-Score:  {entity['f1']:.4f}")


if __name__ == "__main__":
    # Test evaluation
    y_true = ['O', 'O', 'B-NAME_STUDENT', 'I-NAME_STUDENT', 'O', 'B-EMAIL', 'O']
    y_pred = ['O', 'O', 'B-NAME_STUDENT', 'I-NAME_STUDENT', 'O', 'B-EMAIL', 'O']
    
    print("Testing evaluation script...")
    metrics = Evaluator.evaluate(y_true, y_pred)
    print_metrics({**metrics, 'model': 'test', 'total_tokens': len(y_true)})
