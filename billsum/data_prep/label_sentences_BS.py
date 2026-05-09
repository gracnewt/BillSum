'''
Methods to prepare sentences for extractive sentences training.
'''
import jsonlines
import os
import pickle
import re
from rouge import Rouge
import spacy
import torch
from bert_score import score as bert_scorer

nlp = spacy.load('en_core_web_sm')
rouge = Rouge()

section_pattern = re.compile('(SECTION)|(Sec)|(Section) [0-9]+')


def spacy_to_tuple(doc):
    text_feats = [(w.text, w.i, w.lemma_, w.ent_type_, w.ent_iob_, w.pos_, w.dep_, w.head.i)
                                    for w in doc]
    return text_feats

def prepare_summary(bill_data):
    
    final_summary_data = {}
    i = 0

    for _, bill in bill_data.iterrows():

        bill_id = bill['bill_id']

        # Keep track of progress
        i += 1
        if i % 100 == 0:
            print("Processed {} summaries".format(i))

        text_nlp = nlp(bill['summary'])

        doc_data = []

        for sent in text_nlp.sents:
            # Store key features from each sentence
            text_feats = spacy_to_tuple(sent)
            doc_data.append(text_feats)
            
        final_summary_data[bill_id] = doc_data

    return final_summary_data

def prepare_labels(bill_data, min_sent_words=5):
    '''
    Take in a list of data for bills and returns a per-sentence score for
    every sentence in each document. Uses BERTScore under the hood, but formats
    it to masquerade as ROUGE scores to prevent downstream pipeline disruption.

    bill_data: list of dicts/dataframe where each dict 
            has a summary, bill_id and text field
    
    min_sent_words: skip sentences in text with less words.

    Output: dict of bill-id -> list of sent data 
        where sent data is a three-tuple of:
        (original sentence, list of words with spacy annotations, fake_rouge_scores)
    '''

    final_scores = {}
    total_bills = len(bill_data)
    
    # Check if GPU is available to speed up BERTScore calculation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for BERTScore: {device}")

    for idx, (_, bill) in enumerate(bill_data.iterrows()):
        bill_id = bill['bill_id']

        # Keep track of progress
        if idx % 100 == 0:
            print(f"Processing bill {idx}/{total_bills} ({len(final_scores)} saved)")

        text_nlp = nlp(bill['clean_text'])
        summary_text = bill['clean_summary']

        # Skip if summary is empty
        if len(summary_text.strip()) == 0:
            continue

        valid_sents = []
        spacy_feats_list = []

        # Step 1: Filter and collect all valid sentences for this bill
        for sent in text_nlp.sents:
            if len(sent) > min_sent_words and len(sent.text.strip()) > 0:
                valid_sents.append(sent.text)
                spacy_feats_list.append(spacy_to_tuple(sent))

        # Skip if no valid sentences found in this bill
        if not valid_sents:
            continue

        # Step 2: Compute BERTScore in one batched call (massive speedup!)
        # We compare all sentences of this bill against the single human summary
        references = [summary_text] * len(valid_sents)
        
        # 'lang="en"' automatically picks the recommended model ('roberta-common' / 'roberta-large')
        # We set verbose=False to keep the logs clean
        _, _, F1 = bert_scorer(
            valid_sents, 
            references, 
            lang="en", 
            device=device, 
            verbose=False
        )
        
        # Convert the PyTorch tensor of scores back to a standard Python list
        f1_scores = F1.tolist()

        # Step 3: Package scores into a fake "ROUGE" dictionary format
        sent_data = []
        for sent_text, text_feats, f1_score in zip(valid_sents, spacy_feats_list, f1_scores):
            
            # This structural masquerade avoids breaking downstream code.
            # We map the BERTScore F1 directly to the slots the training/eval scripts look for.
            fake_rscores = {
                'rouge-1': {'f': f1_score, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': f1_score, 'p': 0.0, 'r': 0.0}, # Used in bert_2_train
                'rouge-l': {'f': f1_score, 'p': 0.0, 'r': 0.0}  # Used in evaluation
            }

            sent_data.append((sent_text, text_feats, fake_rscores))

        final_scores[bill_id] = sent_data

    return final_scores


if __name__ == '__main__':
    import pandas as pd 

    prefix = os.environ['BILLSUM_PREFIX']

    if not prefix.endswith('/'):
        prefix += '/'

    #os.mkdir(prefix + 'sent_data/')

    print("Preparing US Train")
    data = pd.read_json(prefix + 'clean_final/us_train_data_final.jsonl', lines=True)
    sent_scores = prepare_labels(data)
    pickle.dump(sent_scores, open(prefix + 'sent_data/us_train_sent_scores.pkl', 'wb'))

    sum_sents = prepare_summary(data)
    pickle.dump(sum_sents, open(prefix + 'sent_data/us_train_sum_sents.pkl', 'wb'))


    print("Preparing US Test")
    data = pd.read_json(prefix + 'clean_final/us_test_data_final.jsonl', lines=True)
    sent_scores = prepare_labels(data)
    pickle.dump(sent_scores, open(prefix + 'sent_data/us_test_sent_scores.pkl', 'wb'))

    sum_sents = prepare_summary(data)
    pickle.dump(sum_sents, open(prefix + 'sent_data/us_test_sum_sents.pkl', 'wb'))


    print("Preparing CA Test")
    data = pd.read_json(prefix + 'clean_final/ca_test_data_final.jsonl', lines=True)
    sent_scores = prepare_labels(data)
    pickle.dump(sent_scores, open(prefix + 'sent_data/ca_test_sent_scores.pkl', 'wb'))

    sum_sents = prepare_summary(data)
    pickle.dump(sum_sents, open(prefix + 'sent_data/ca_test_sum_sents.pkl', 'wb'))


