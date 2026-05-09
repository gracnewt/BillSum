import os
import pickle
import pandas as pd

# Setup display options so nothing gets truncated in your terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

PREFIX = "/home/gracenewton/nlp_final/BillSum/billsum/data/score_data/"

# Define domains and evaluation structures
domains = ["us", "ca"]

# Left edge starts with Oracles, followed by our fine-tuned models
models = ["oracle_scores_2", "oracle_scores_l", "oracle_scores_BS", "full2", "full2_fast", "fullL", "fullBS"]

# Exact column renaming mapping
column_mapping = {
    "oracle_scores_2": "Oracle (R2)",
    "oracle_scores_l": "Oracle (RL)",
    "oracle_scores_BS": "Oracle (BS)",
    "full2": "Legal-BERT (R2)",
    "full2_fast": "Legal-BERT (R2 Fast)",
    "fullL": "Legal-BERT (RL)",
    "fullBS": "Legal-BERT (BS)"
}

files_to_load = {}
for domain in domains:
    for model in models:
        # Construct the matching file names like 'bs_ca_oracle_scores_l.pkl' or 'bs_ca_bert_scores_fullBS.pkl'
        if "oracle" in model:
            files_to_load[f"{domain}_{model}"] = os.path.join(PREFIX, f"bs_{domain}_{model}.pkl")
        else:
            files_to_load[f"{domain}_{model}"] = os.path.join(PREFIX, f"bs_{domain}_bert_scores_{model}.pkl")

def extract_all_metrics(filepath):
    if not os.path.exists(filepath):
        # Gracefully handle missing/not-yet-run files
        return None
        
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    rows = []
    for bill_id, metrics in data.items():
        try:
            row = {
                'R1-P': metrics['rouge-1']['p'] * 100,
                'R1-R': metrics['rouge-1']['r'] * 100,
                'R1-F': metrics['rouge-1']['f'] * 100,
                
                'R2-P': metrics['rouge-2']['p'] * 100,
                'R2-R': metrics['rouge-2']['r'] * 100,
                'R2-F': metrics['rouge-2']['f'] * 100,
                
                'RL-P': metrics['rouge-l']['p'] * 100,
                'RL-R': metrics['rouge-l']['r'] * 100,
                'RL-F': metrics['rouge-l']['f'] * 100,
            }
            # Safely extract BERTScore
            raw_val = metrics.get('bertscore-f', metrics.get('bert_score_f'))
            if raw_val is not None:
                row['BERTScore F1'] = raw_val * 100 if raw_val < 1.0 else raw_val
            else:
                row['BS-F'] = None
                
            rows.append(row)
        except (KeyError, TypeError):
            continue
            
    if not rows:
        return None
        
    df = pd.DataFrame(rows)
    return df.mean()

# Extract data
raw_results = {}
for name, path in files_to_load.items():
    mean_stats = extract_all_metrics(path)
    if mean_stats is not None:
        raw_results[name] = mean_stats

if raw_results:
    # Build complete DataFrame
    df_all = pd.DataFrame(raw_results)

    # 1. GENERATE THE READABLE F1-ONLY TABLE
    # Reversing the mapping so Pandas can find the correct index names in df_all
    f1_metrics = {
        'R1-F': 'ROUGE-1 F1',
        'R2-F': 'ROUGE-2 F1',
        'RL-F': 'ROUGE-L F1',
        'BS-F': 'BERTScore F1'
    }
    
    # Now this line will map correctly and find the rows!
    df_f1 = df_all.loc[[f1_metrics[k] for k in f1_metrics]]
    
    # Extract only the F1 rows
    df_f1 = df_all.loc[[f1_metrics[k] for k in f1_metrics]]
    df_f1.index = f1_metrics.keys()
    
    # Split columns by US and CA domain, keeping our custom left-edge ordering
    us_cols = [f"us_{m}" for m in models if f"us_{m}" in df_f1.columns]
    ca_cols = [f"ca_{m}" for m in models if f"ca_{m}" in df_f1.columns]
    
    # Function to rename headers cleanly
    def format_headers(col_name):
        base_name = col_name.split('_', 1)[1]
        return column_mapping.get(base_name, base_name)

    print("\n" + "="*125)
    print("                      TABLE 1: US DOMAIN F1 SUMMARY")
    print("="*125)
    us_table = df_f1[us_cols].rename(columns=format_headers)
    print(us_table)
    
    print("\n" + "="*125)
    print("                      TABLE 2: CA DOMAIN F1 SUMMARY")
    print("="*125)
    ca_table = df_f1[ca_cols].rename(columns=format_headers)
    print(ca_table)

    # 2. GENERATE THE COMPREHENSIVE P / R / F GRID
    print("\n" + "="*125)
    print("                 TABLE 3: COMPLETE PRECISION, RECALL & F1 GRID")
    print("="*125)
    transposed_df = df_all.T
    transposed_df.index = [f"{idx.split('_')[0].upper()} - {column_mapping.get(idx.split('_', 1)[1], idx)}" for idx in transposed_df.index]
    print(transposed_df)
    print("="*125)

else:
    print("Error: No evaluation files could be loaded. Double-check your pathing: " + PREFIX)