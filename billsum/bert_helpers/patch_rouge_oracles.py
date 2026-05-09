import os
import pickle
import pandas as pd
import torch
from rouge import Rouge
from bert_score import BERTScorer
from billsum.post_process import mmr_selection

# Setup paths
PREFIX = "/home/gracenewton/nlp_final/BillSum/billsum/data/"
rouge = Rouge()

# Initialize BERTScorer once globally (on GPU if available for maximum speed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading BERTScorer on: {device}...")
bert_scorer = BERTScorer(lang="en", device=device, rescale_with_baseline=True)

def generate_and_evaluate_rouge_oracles(domain):
    print(f"\nProcessing {domain.upper()} ROUGE Oracles...")
    
    # 1. Load data structures
    sent_data_path = os.path.join(PREFIX, 'sent_data', f'{domain}_test_sent_scores.pkl')
    sent_data = pickle.load(open(sent_data_path, 'rb'))
    
    docs_path = os.path.join(PREFIX, 'clean_final', f'{domain}_test_data_final_OFFICIAL.jsonl')
    docs = pd.read_json(docs_path, lines=True)
    docs.set_index('bill_id', inplace=True)
    
    doc_order = sorted(sent_data.keys())
    
    # Storage for both oracle types
    oracles = {
        '2': {'summaries': [], 'refs': [], 'ids': [], 'scores': {}},
        'L': {'summaries': [], 'refs': [], 'ids': [], 'scores': {}}
    }
    
    # 2. Build the Oracle Summaries using MMR over ROUGE-2 and ROUGE-L sentence scores
    for bill_id in doc_order:
        sents = sent_data[bill_id]
        mysents = [s[0] for s in sents]
        human_summary = docs.loc[bill_id].get('clean_summary', docs.loc[bill_id].get('summary'))
        
        if len(human_summary.strip()) == 0 or len(mysents) == 0:
            continue
            
        # Extract the pre-calculated ROUGE sentence-level scores
        # Structure of sents: (sentence_text, {rouge-1: F1, rouge-2: F1, rouge-l: F1})
        # Extract by integer index: s[1][1] is ROUGE-2, s[1][2] is ROUGE-L
        r2_scores = [s[1][1] for s in sents]
        rl_scores = [s[1][2] for s in sents]
        
        # Build summaries using MMR
        summary_r2 = ' '.join(mmr_selection(mysents, r2_scores))
        summary_rl = ' '.join(mmr_selection(mysents, rl_scores))
        
        # Save structural details
        for key, summary in [('2', summary_r2), ('L', summary_rl)]:
            oracles[key]['summaries'].append(summary)
            oracles[key]['refs'].append(human_summary)
            oracles[key]['ids'].append(bill_id)
            
            # Compute ROUGE metrics
            if domain == 'us':
                r_score = rouge.get_scores([summary], [human_summary])[0]
            else:
                r_score = rouge.get_scores([human_summary], [summary])[0]
            oracles[key]['scores'][bill_id] = r_score

    # 3. Batch run BERTScore on all generated summaries to bypass NaN issues
    for key in ['2', 'L']:
        print(f"Batch evaluating BERTScore for Oracle-{key}...")
        P, R, F1 = bert_scorer.score(oracles[key]['summaries'], oracles[key]['refs'])
        f1_scores = F1.tolist()
        
        # Merge BERTScore metrics back into our scores dictionary
        for idx, bill_id in enumerate(oracles[key]['ids']):
            oracles[key]['scores'][bill_id]['bertscore-f'] = f1_scores[idx]
            
        # 4. Save directly to the pickle files expected by your extraction script
        # Using the naming structure: bs_us_bert_scores_2.pkl, etc.
        out_name = f"bs_{domain}_oracle_scores_{key.lower()}.pkl"
        out_path = os.path.join(PREFIX, 'score_data', out_name)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            pickle.dump(oracles[key]['scores'], f)
        print(f"Saved: {out_path}")

# Run for both domains
generate_and_evaluate_rouge_oracles('us')
generate_and_evaluate_rouge_oracles('ca')
print("\nAll missing ROUGE Oracle BERTScores patched successfully!")