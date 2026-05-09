import os
import pickle
import pandas as pd

# Setup display options so nothing gets truncated in your terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

PREFIX = "/home/gracenewton/nlp_final/BillSum/billsum/data/score_data/"

# Define all system runs and domains
domains = ["us", "ca"]
models = ["full2", "full2_fast", "fullL", "fullBS", "2", "l", "BS"]

files_to_load = {}
for domain in domains:
    for model in models:
        # Construct the matching file names
        files_to_load[f"{domain}_{model}"] = os.path.join(PREFIX, f"bs_{domain}_bert_scores_{model}.pkl")

def extract_all_metrics(filepath):
    if not os.path.exists(filepath):
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
            # Safely handle BERTScore
            bs_val = metrics.get('bertscore-f', metrics.get('bert_score_f', None))
            if bs_val is not None:
                row['BS-F'] = bs_val * 100
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
    f1_metrics = {
        'ROUGE-1 F1': 'R1-F',
        'ROUGE-2 F1': 'R2-F',
        'ROUGE-L F1': 'RL-F',
        'BERTScore F1': 'BS-F'
    }
    
    # Extract only the F1 rows and clean column names for presentation
    df_f1 = df_all.loc[[f1_metrics[k] for k in f1_metrics]]
    df_f1.index = f1_metrics.keys()
    
    # Split by US and CA for cleaner layout
    us_cols = [c for c in df_f1.columns if c.startswith('us_')]
    ca_cols = [c for c in df_f1.columns if c.startswith('ca_')]
    
    print("\n" + "="*90)
    print("                     TABLE 1: US DOMAIN F1 SUMMARY")
    print("="*90)
    print(df_f1[us_cols].rename(columns=lambda x: x.replace('us_', '')))
    
    print("\n" + "="*90)
    print("                     TABLE 2: CA DOMAIN F1 SUMMARY")
    print("="*90)
    print(df_f1[ca_cols].rename(columns=lambda x: x.replace('ca_', '')))

    # 2. GENERATE THE COMPREHENSIVE P / R / F GRID
    print("\n" + "="*90)
    print("                 TABLE 3: COMPLETE PRECISION, RECALL & F1 GRID")
    print("="*90)
    # Transposing makes reading long column lists down a page much easier
    print(df_all.T)
    print("="*90)

else:
    print("Error: No evaluation files could be loaded. Double-check your patch execution!")