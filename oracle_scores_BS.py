import os
import pickle
import pandas as pd
import torch
from rouge import Rouge
from bert_score import BERTScorer
from billsum.post_process import mmr_selection

# Initialize standard ROUGE scorer
rouge = Rouge()

# Initialize BERTScorer once globally (on GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading BERTScorer globally on: {device}...")
bert_scorer = BERTScorer(lang="en", device=device, rescale_with_baseline=True)

prefix = "/home/gracenewton/nlp_final/BillSum/billsum/data/"

def score_bertscore_oracles(domain):
    print(f"\n=== Calculating BERTScore Oracles for {domain.upper()} ===")
    
    # 1. Load sentence-level structure data
    sent_data_path = os.path.join(prefix, 'sent_data', f'{domain}_test_sent_scores.pkl')
    sent_data = pickle.load(open(sent_data_path, 'rb'))
    
    # 2. Load clean original documents (for the human target summary)
    docs_path = os.path.join(prefix, 'clean_final', f'{domain}_test_data_final_OFFICIAL.jsonl')
    docs = pd.read_json(docs_path, lines=True)
    docs.set_index('bill_id', inplace=True)
    
    all_oracle_scores = {}
    doc_order = sorted(sent_data.keys())
    
    # Containers to batch final evaluation
    oracle_summaries = []
    reference_summaries = []
    bill_ids_ordered = []
    
    for idx, bill_id in enumerate(doc_order):
        sents = sent_data[bill_id]
        mysents = [s[0] for s in sents]
        
        # Grab the human target summary
        human_summary = docs.loc[bill_id].get('clean_summary', docs.loc[bill_id].get('summary'))
        
        if len(human_summary.strip()) == 0 or len(mysents) == 0:
            continue
            
        # 3. Calculate sentence-level BERTScores to find the local "Oracle" weights
        # We compare every sentence in this bill to the target summary
        references = [human_summary] * len(mysents)
        _, _, F1 = bert_scorer.score(mysents, references)
        sentence_bert_scores = F1.tolist()
        
        # 4. Use MMR to select the best, non-redundant sentences based on BERTScore
        oracle_summary = ' '.join(mmr_selection(mysents, sentence_bert_scores))
        
        # Calculate standard ROUGE metrics on this Oracle summary
        # Keep US and CA orientation consistent with your evaluator
        if domain == 'us':
            r_score = rouge.get_scores([oracle_summary], [human_summary])[0]
        else:
            r_score = rouge.get_scores([human_summary], [oracle_summary])[0]
            
        all_oracle_scores[bill_id] = r_score
        
        # Cache for final batched BERTScore evaluation
        oracle_summaries.append(oracle_summary)
        reference_summaries.append(human_summary)
        bill_ids_ordered.append(bill_id)
        
        if (idx + 1) % 100 == 0:
            print(f"Constructed Oracle {idx + 1}/{len(doc_order)}...")

    # 5. Batch-evaluate the final Oracle Summaries against human summaries via BERTScore
    print(f"Batch evaluating final BERTScores for {domain.upper()} Oracles...")
    P, R, F1 = bert_scorer.score(oracle_summaries, reference_summaries)
    
    p_scores = P.tolist()
    r_scores = R.tolist()
    f1_scores = F1.tolist()
    
    # Merge the semantic evaluation back into the dictionary
    for idx, bill_id in enumerate(bill_ids_ordered):
        all_oracle_scores[bill_id]['bertscore-p'] = p_scores[idx]
        all_oracle_scores[bill_id]['bertscore-r'] = r_scores[idx]
        all_oracle_scores[bill_id]['bertscore-f'] = f1_scores[idx]

    # 6. Save the results
    output_path = os.path.join(prefix, 'score_data', f'{domain}_oracle_scores_BS.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pickle.dump(all_oracle_scores, open(output_path, 'wb'))
    print(f"Successfully saved BERTScore Oracle results to: {output_path}")

# Run for both US and California
score_bertscore_oracles('us')
score_bertscore_oracles('ca')
print("\nAll BERTScore Oracle evaluations completed successfully!")