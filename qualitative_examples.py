import os
import pickle
import pandas as pd
from billsum.post_process import mmr_selection

# --- Configuration ---
prefix = "/home/gracenewton/nlp_final/BillSum/billsum/data/"
num_examples_to_extract = 3  # How many bills you want to inspect
model = "full2"

def extract_summaries(domain):
    print(f"--- Extracting {domain.upper()} Examples ---")
    
    # 1. Load data
    if domain == 'us':
        domain_corrected_test_results = f"test_results.tsv_{model}"
    else:
        domain_corrected_test_results = f"{domain}_test_results_{model}.tsv"
    predictions = pd.read_csv(os.path.join('billsum_bert_results', domain_corrected_test_results), sep='\t', header=None)
    pos_pred = predictions[1].values if predictions.shape[1] > 1 else predictions[0].values
    
    sent_data = pickle.load(open(os.path.join(prefix, 'sent_data', f'{domain}_test_sent_scores.pkl'), 'rb'))
    docs = pd.read_json(os.path.join(prefix, 'clean_final', f'{domain}_test_data_final.jsonl'), lines=True)
    docs.set_index('bill_id', inplace=True)
    
    doc_order = sorted(sent_data.keys())
    
    # We will save the texts to a list
    examples = []
    
    i = 0
    for bill_id in doc_order:
        sents = sent_data[bill_id]
        tot_sent = len(sents)
        
        # Extract BERT predictions for this bill
        ys = pos_pred[i : i+tot_sent]
        i += tot_sent
        
        # Extract sentences and ground-truth scores
        mysents = [s[0] for s in sents]
        
        # Generate the BERT summary using MMR
        pred_summary = ' '.join(mmr_selection(mysents, ys))
        
        # Generate the Oracle summary (using the ground-truth sentence labels)
        # s[2] is the ROUGE score dictionary from your pickle
        # We sort sentences by their ROUGE-2 or ROUGE-L F1-score to find the best ones
        oracle_ys = [s[2]['rouge-2']['f'] for s in sents]
        oracle_summary = ' '.join(mmr_selection(mysents, oracle_ys))
        
        # Get the human reference summary
        human_summary = docs.loc[bill_id].get('clean_summary', docs.loc[bill_id].get('summary'))
        
        examples.append({
            "bill_id": bill_id,
            "title": docs.loc[bill_id].get('title', 'No Title'),
            "human": human_summary,
            "oracle": oracle_summary,
            "predicted": pred_summary
        })
        
        if len(examples) >= num_examples_to_extract:
            break
            
    return examples

# Run for both domains
us_examples = extract_summaries('us')
ca_examples = extract_summaries('ca')

# Write to a clean text file for your qualitative analysis
output_file = "qualitative_analysis_examples.txt"
with open(output_file, "w") as f:
    for domain, dataset in [("US Federal", us_examples), ("California State", ca_examples)]:
        f.write("="*80 + "\n")
        f.write(f"                     DOMAIN: {domain.upper()}\n")
        f.write("="*80 + "\n\n")
        
        for idx, item in enumerate(dataset):
            f.write(f"Example {idx + 1} | Bill ID: {item['bill_id']}\n")
            f.write(f"Title: {item['title']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"[HUMAN REFERENCE SUMMARY]:\n{item['human']}\n\n")
            f.write(f"[ORACLE (BEST EXTRACTIVE LIMIT)]:\n{item['oracle']}\n\n")
            f.write(f"[BERT PREDICTED SUMMARY]:\n{item['predicted']}\n\n")
            f.write("*" * 80 + "\n\n")

print(f"Success! Qualitative examples saved to {output_file}")
