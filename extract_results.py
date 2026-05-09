import os
import pickle
import pandas as pd

# Define paths to your evaluation pickle files
PREFIX = "/home/gracenewton/nlp_final/BillSum/billsum/data/score_data/"

# Define the models and domains you want to extract
# This assumes your newly generated files follow the naming convention we set up
files_to_load = {}
domains = ["us", "ca"]
models = ["full2", "full2_fast", "fullL", "fullBS"]
for domain in domains:
    for model in models:
        files_to_load[f"{domain}_{model}"] = os.path.join(PREFIX, f"bs_{domain}_bert_scores_{model}.pkl")


def extract_mean_metrics(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found at {filepath}")
        return None
        
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    rows = []
    for bill_id, metrics in data.items():
        # Handle cases where some keys might be structured differently
        try:
            row = {
                'ROUGE-1 F1': metrics['rouge-1']['f'] * 100,
                'ROUGE-2 F1': metrics['rouge-2']['f'] * 100,
                'ROUGE-L F1': metrics['rouge-l']['f'] * 100,
            }
            # Look for the newly added BERTScore metrics
            if 'bertscore-f' in metrics:
                row['BERTScore F1'] = metrics['bertscore-f'] * 100
            elif 'bert_score_f' in metrics:  # Fallback naming check
                row['BERTScore F1'] = metrics['bert_score_f'] * 100
            else:
                row['BERTScore F1'] = None
                
            rows.append(row)
        except KeyError as e:
            # Skip corrupted/unmatched entries if they exist
            continue
            
    if not rows:
        print(f"Warning: No valid metric structures found in {filepath}")
        return None
        
    df = pd.DataFrame(rows)
    return df.mean()

# Execute extraction
results = {}
for model_name, path in files_to_load.items():
    mean_scores = extract_mean_metrics(path)
    if mean_scores is not None:
        results[model_name] = mean_scores

# Format output into a combined table
if results:
    results_df = pd.DataFrame(results).round(2)
    print("\n" + "="*50)
    print("             SUMMARY EVALUATION RESULTS")
    print("="*50)
    print(results_df)
    print("="*50)
else:
    print("No evaluation files were successfully loaded.")