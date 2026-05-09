import os
import pickle
import pandas as pd
import torch
from rouge import Rouge
from bert_score import BERTScorer
from billsum.post_process import greedy_summarize, mmr_selection

# Initialize ROUGE scorer
rouge = Rouge()

# Initialize BERTScorer once globally (on GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading BERTScorer for evaluation on: {device}...")
# "lang='en'" dynamically chooses the optimal model (e.g., roberta-large)
bert_scorer = BERTScorer(lang="en", device=device, rescale_with_baseline=True)

prefix = os.environ['BILLSUM_PREFIX']
bert_score_dir = 'billsum_bert_results/'
output_dir = 'score_data'

def run_evaluation(domain, model):
    print(f"\n--- Starting Evaluation for {domain.upper()} ---")
    
    model_file = f"test_results_{model}.tsv"
    if domain == "ca":
        model_file = "ca_" + model_file
    
    # 1. Load predictions
    pred_path = os.path.join(bert_score_dir, model_file)
    predictions = pd.read_csv(pred_path, sep='\t', header=None)
    pos_pred = predictions[1].values

    # 2. Load sentence-level and document-level data
    sent_data = pickle.load(open(os.path.join(prefix, 'sent_data', f'{domain}_test_sent_scores.pkl'), 'rb'))
    docs = pd.read_json(os.path.join(prefix, 'clean_final', f'{domain}_test_data_final_OFFICIAL.jsonl'), lines=True)
    docs.set_index('bill_id', inplace=True)

    doc_order = sorted(sent_data.keys())
    all_scores = {}
    
    # Containers to batch BERTScore computation after looping
    generated_summaries = []
    reference_summaries = []
    bill_ids_ordered = []

    # Loop to generate MMR summaries and calculate ROUGE
    i = 0
    for bill_id in doc_order:
        sents = sent_data[bill_id]
        tot_sent = len(sents)

        # Slice predictions corresponding to this specific bill
        ys = pos_pred[i : i+tot_sent]
        i += tot_sent

        # Reconstruct texts
        mysents = [s[0] for s in sents]
        final_sum = ' '.join(mmr_selection(mysents, ys))
        
        # Pull the correct human-written summary field
        target_summary = docs.loc[bill_id].get('clean_summary', docs.loc[bill_id].get('summary'))

        # Standardize ROUGE parameters (hypothesis, reference)
        score = rouge.get_scores([final_sum], [target_summary])[0]
        
        # Save ROUGE scores temporarily
        all_scores[bill_id] = score
        
        # Cache text strings for batch BERTScore evaluation
        generated_summaries.append(final_sum)
        reference_summaries.append(target_summary)
        bill_ids_ordered.append(bill_id)

    # 3. Batch Compute BERTScore for extreme GPU acceleration
    print(f"Batch computing BERTScores for {len(generated_summaries)} {domain.upper()} bills...")
    P, R, F1 = bert_scorer.score(generated_summaries, reference_summaries)
    
    # Convert PyTorch tensors to standard Python lists
    p_scores = P.tolist()
    r_scores = R.tolist()
    f1_scores = F1.tolist()

    # 4. Merge BERTScore results with the ROUGE score dictionary
    for idx, bill_id in enumerate(bill_ids_ordered):
        all_scores[bill_id]['bertscore-p'] = p_scores[idx]
        all_scores[bill_id]['bertscore-r'] = r_scores[idx]
        all_scores[bill_id]['bertscore-f'] = f1_scores[idx]

    # 5. Output metrics pickle
    output_pkl_path = os.path.join(prefix, 'score_data', f"bs_{domain}_bert_scores_{model}.pkl")
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    pickle.dump(all_scores, open(output_pkl_path, 'wb'))
    print(f"Saved evaluation results to: {output_pkl_path}")


# Run evaluation on both datasets
models = ["full2", "full2_fast", "fullL", "fullBS"]
domains = ["us", "ca"]
for model in models:
    for domain in domains:
        run_evaluation(domain, model)

print("\nEvaluation pipeline successfully finished!")
