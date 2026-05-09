import os
import pickle
import pandas as pd
from rouge import Rouge
from billsum.post_process import mmr_selection

rouge = Rouge()
prefix = "/home/gracenewton/nlp_final/BillSum/billsum/data/"

def score_oracle_dataset(domain):
    print(f"--- Scoring ORACLE Summaries for {domain.upper()} ---")
    
    # 1. Load ground-truth sentence-level data
    sent_data_path = os.path.join(prefix, 'sent_data', f'{domain}_test_sent_scores.pkl')
    sent_data = pickle.load(open(sent_data_path, 'rb'))
    
    # 2. Load clean original documents (for the human target summary)
    docs_path = os.path.join(prefix, 'clean_final', f'{domain}_test_data_final.jsonl')
    docs = pd.read_json(docs_path, lines=True)
    docs.set_index('bill_id', inplace=True)
    
    all_oracle_scores = {}
    doc_order = sorted(sent_data.keys())
    
    for idx, bill_id in enumerate(doc_order):
        sents = sent_data[bill_id]
        
        # Extract sentence strings
        mysents = [s[0] for s in sents]
        
        # Extract the sentence-level ROUGE scores computed during labeling
        # s[2] contains the ROUGE metric dict. We use ROUGE-2 F1-score as the selector.
        oracle_ys = [s[2]['rouge-l']['f'] for s in sents]
        
        # Construct the Oracle Summary by selecting the best non-redundant sentences
        oracle_summary = ' '.join(mmr_selection(mysents, oracle_ys))
        
        # Get the human reference summary
        human_summary = docs.loc[bill_id].get('clean_summary', docs.loc[bill_id].get('summary'))
        
        # Calculate full ROUGE (1, 2, L) against the human reference
        # US and CA are structured slightly differently, we keep the original swap consistent
        if domain == 'us':
            score = rouge.get_scores([oracle_summary], [human_summary])[0]
        else:
            score = rouge.get_scores([human_summary], [oracle_summary])[0]
            
        all_oracle_scores[bill_id] = score
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(doc_order)} documents...")

    # 3. Save the Oracle scores to their own pickle files
    output_path = os.path.join(prefix, 'score_data', f'{domain}_oracle_scores_l.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pickle.dump(all_oracle_scores, open(output_path, 'wb'))
    print(f"Successfully saved Oracle scores to: {output_path}\n")

# Run scoring for both US and CA
score_oracle_dataset('us')
score_oracle_dataset('ca')