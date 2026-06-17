import os
import json
import glob
import re
import csv
import sys

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.evaluate import evaluate_from_csv, save_metrics, print_metrics

def find_best_ml_model(metrics_dir='results/metrics'):
    """
    Find the best ML model based on token-level F1 score.
    Returns the path to the best model's metrics JSON and its base name.
    """
    best_f1 = -1
    best_model_file = None
    
    metrics_files = glob.glob(os.path.join(metrics_dir, '*.json'))
    for file in metrics_files:
        if 'summary' in file or 'template' in file or 'hybrid' in file:
            continue
            
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'token_level' in data and 'f1' in data['token_level']:
                    f1 = data['token_level']['f1']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model_file = file
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    if best_model_file:
        model_name = os.path.basename(best_model_file).replace('_metrics.json', '')
        return best_model_file, model_name
    return None, None

def regex_rule_match(token):
    """
    Apply rule-based regex matching on a single token.
    Returns the predicted BIO label if a match is found, else None.
    """
    token_str = str(token)
    # Email: simple pattern for typical emails
    if re.fullmatch(r'[\w\.-]+@[\w\.-]+\.\w+', token_str):
        return 'B-EMAIL'
    
    # URL: starts with http://, https://, or www.
    if re.fullmatch(r'(https?://|www\.).*', token_str):
        return 'B-URL_PERSONAL'
    
    # Phone number: pattern like 123-456-7890 or (123) 456-7890
    if re.fullmatch(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', token_str):
        return 'B-PHONE_NUM'
    
    # ID number: arbitrary length of digits >= 5
    if re.fullmatch(r'\d{5,}', token_str):
        return 'B-ID_NUM'
        
    return None

def run_hybrid_model():
    print("Finding the best ML model...")
    best_metrics_file, best_model_name = find_best_ml_model()
    
    if not best_model_name:
        print("No valid ML model metrics found.")
        return
        
    print(f"Best ML model identified: {best_model_name} from {best_metrics_file}")
    
    # Map the best model to its predictions CSV
    predictions_dir = 'results/predictions'
    pred_file = os.path.join(predictions_dir, f"{best_model_name}.csv")
    if not os.path.exists(pred_file):
        clean_name = best_model_name.replace('_after_realval', '_after').replace('_before', '_before')
        pred_file = os.path.join(predictions_dir, f"{clean_name}.csv")
        
    if not os.path.exists(pred_file):
        print(f"Could not find exact prediction file for {best_model_name}.")
        csv_files = glob.glob(os.path.join(predictions_dir, '*.csv'))
        for f in csv_files:
            if best_model_name.replace('_metrics', '') in f or f.endswith(f"{best_model_name}.csv"):
                pred_file = f
                break
                
    if not os.path.exists(pred_file):
        print(f"Failed to find predictions CSV for {best_model_name} in {predictions_dir}")
        return
        
    print(f"Loading predictions from: {pred_file}")
    
    hybrid_pred_path = os.path.join(predictions_dir, 'hybrid_predictions.csv')
    print("Applying Hybrid (Regex + ML) logic...")
    rule_overrides = 0
    total_rows = 0
    
    with open(pred_file, 'r', encoding='utf-8') as f_in, \
         open(hybrid_pred_path, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            token = row['token']
            ml_pred = row['pred_label']
            
            rule_pred = regex_rule_match(token)
            if rule_pred:
                if rule_pred != ml_pred:
                    rule_overrides += 1
                row['pred_label'] = rule_pred
                
            writer.writerow(row)
            total_rows += 1
            
    print(f"Processed {total_rows} predictions.")
    print(f"Hybrid model replaced/overridden {rule_overrides} ML predictions using Regex.")
    print(f"Saved hybrid predictions to: {hybrid_pred_path}")
    
    print("Evaluating hybrid model...")
    metrics = evaluate_from_csv(hybrid_pred_path, 'hybrid_regex_ml')
    
    hybrid_metrics_path = os.path.join('results/metrics', 'hybrid_metrics.json')
    print(f"Saving hybrid metrics to: {hybrid_metrics_path}")
    save_metrics(metrics, hybrid_metrics_path)
    
    print_metrics(metrics)
    print("Hybrid model pipeline complete.")

if __name__ == '__main__':
    run_hybrid_model()
