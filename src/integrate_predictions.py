"""
Script untuk integrate dan collect predictions dari semua model.
Jalankan setelah semua model sudah generate predictions.
"""

import os
import csv
import json
import glob
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, '/home/ata/school/fp-ml/src')
from evaluate import Evaluator, print_metrics


class PredictionIntegrator:
    """Aggregate predictions dari semua model dan evaluasi."""
    
    def __init__(self, predictions_dir: str = "results/predictions", 
                 metrics_dir: str = "results/metrics"):
        """
        Initialize integrator.
        
        Args:
            predictions_dir: Directory untuk prediction CSV files
            metrics_dir: Directory untuk metrics JSON files
        """
        self.predictions_dir = predictions_dir
        self.metrics_dir = metrics_dir
    
    def find_prediction_files(self) -> List[str]:
        """Find all prediction CSV files."""
        pattern = f"{self.predictions_dir}/*_predictions.csv"
        files = glob.glob(pattern)
        return sorted(files)
    
    def extract_model_name(self, filepath: str) -> str:
        """Extract model name dari filepath."""
        filename = os.path.basename(filepath)
        return filename.replace('_predictions.csv', '')
    
    def load_predictions(self, filepath: str) -> tuple:
        """Load predictions dari CSV."""
        y_true = []
        y_pred = []
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                y_true.append(row['true_label'])
                y_pred.append(row['pred_label'])
        
        return y_true, y_pred
    
    def evaluate_all_models(self) -> Dict:
        """Evaluate semua model."""
        prediction_files = self.find_prediction_files()
        
        if not prediction_files:
            print("❌ Tidak ada prediction files ditemukan!")
            print(f"   Cari di: {self.predictions_dir}/*_predictions.csv")
            return {}
        
        print(f"Found {len(prediction_files)} model predictions:")
        for f in prediction_files:
            print(f"  - {os.path.basename(f)}")
        
        results = {}
        
        for filepath in prediction_files:
            model_name = self.extract_model_name(filepath)
            print(f"\n[Evaluating] {model_name}...")
            
            try:
                y_true, y_pred = self.load_predictions(filepath)
                metrics = Evaluator.evaluate(y_true, y_pred)
                metrics['model'] = model_name
                metrics['total_tokens'] = len(y_true)
                
                results[model_name] = metrics
                
                # Save metrics untuk model ini
                metrics_file = f"{self.metrics_dir}/{model_name}_metrics.json"
                os.makedirs(self.metrics_dir, exist_ok=True)
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"  ✓ Metrics saved: {metrics_file}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        return results
    
    def generate_comparison_table(self, results: Dict) -> str:
        """Generate markdown table untuk perbandingan model."""
        if not results:
            return "No results to compare"
        
        lines = []
        lines.append("# Model Comparison")
        lines.append("")
        lines.append("## Token-Level Metrics")
        lines.append("")
        
        # Token-level table
        lines.append("| Model | Precision | Recall | F1-Score |")
        lines.append("|-------|-----------|--------|----------|")
        
        for model_name in sorted(results.keys()):
            metrics = results[model_name]['token_level']
            lines.append(f"| {model_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |")
        
        lines.append("")
        lines.append("## Entity-Level Metrics")
        lines.append("")
        
        # Entity-level table
        lines.append("| Model | Precision | Recall | F1-Score |")
        lines.append("|-------|-----------|--------|----------|")
        
        for model_name in sorted(results.keys()):
            metrics = results[model_name]['entity_level']
            lines.append(f"| {model_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |")
        
        lines.append("")
        lines.append("## Per-Class F1 Scores")
        lines.append("")
        
        # Per-class table
        lines.append("| Label | " + " | ".join(sorted(results.keys())) + " |")
        lines.append("|-------|" + "|".join(["-" * (len(m) + 2) for m in sorted(results.keys())]) + "|")
        
        # Get all unique labels
        all_labels = set()
        for model_metrics in results.values():
            all_labels.update(model_metrics['token_level']['per_class'].keys())
        
        for label in sorted(all_labels):
            line = f"| {label} |"
            for model_name in sorted(results.keys()):
                f1 = results[model_name]['token_level']['per_class'].get(label, {}).get('f1', 0.0)
                line += f" {f1:.4f} |"
            lines.append(line)
        
        return "\n".join(lines)
    
    def print_summary(self, results: Dict) -> None:
        """Print summary dari semua hasil."""
        print("\n" + "="*80)
        print(f"{'EVALUATION SUMMARY':^80}")
        print("="*80)
        
        if not results:
            print("❌ Tidak ada hasil untuk di-summarize")
            return
        
        # Find best model
        best_f1_model = max(results.items(), 
                            key=lambda x: x[1]['token_level']['f1'])
        
        print(f"\n📊 Total Models Evaluated: {len(results)}")
        print(f"🏆 Best Model (Token F1): {best_f1_model[0]} ({best_f1_model[1]['token_level']['f1']:.4f})")
        
        print("\n📋 All Models (Token-Level F1):")
        for model_name in sorted(results.keys(), 
                                 key=lambda x: results[x]['token_level']['f1'], 
                                 reverse=True):
            f1 = results[model_name]['token_level']['f1']
            p = results[model_name]['token_level']['precision']
            r = results[model_name]['token_level']['recall']
            print(f"  {model_name:30s}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
        
        print("\n" + "="*80)


def main():
    print("\n" + "="*80)
    print(f"{'PII Shield - Prediction Integration & Evaluation':^80}")
    print("="*80)
    
    integrator = PredictionIntegrator()
    
    # Evaluate all models
    results = integrator.evaluate_all_models()
    
    if not results:
        print("\n❌ Tidak ada model predictions ditemukan!")
        print("Pastikan setiap model generate file:")
        print("  results/predictions/[model_name]_predictions.csv")
        return
    
    # Print summary
    integrator.print_summary(results)
    
    # Generate comparison
    print("\nGenerating comparison table...")
    comparison = integrator.generate_comparison_table(results)
    
    # Save comparison
    comparison_file = "results/final_comparison.md"
    os.makedirs(os.path.dirname(comparison_file), exist_ok=True)
    with open(comparison_file, 'w') as f:
        f.write(comparison)
    
    print(f"\n✓ Comparison table saved: {comparison_file}")
    print("\n" + "="*80)
    print("Integration complete!")
    print("="*80)


if __name__ == "__main__":
    main()
