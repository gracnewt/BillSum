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
from bert_score import BERTScorer

# 1. Initialize the scorer ONCE globally
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading BERTScorer globally on: {device}...")

# This loads RoBERTa-large into GPU memory once and keeps it there
scorer = BERTScorer(lang="en", device=device)

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
    Modified to use the pre-loaded global 'scorer' instance to prevent
    re-loading RoBERTa on every single loop iteration.
    '''
    final_scores = {}
    total_bills = len(bill_data)

    for idx, (_, bill) in enumerate(bill_data.iterrows()):
        bill_id = bill['bill_id']

        if idx % 100 == 0:
            print(f"Processing bill {idx}/{total_bills} ({len(final_scores)} saved)")

        text_nlp = nlp(bill['clean_text'])
        summary_text = bill['clean_summary']

        if len(summary_text.strip()) == 0:
            continue

        valid_sents = []
        spacy_feats_list = []

        for sent in text_nlp.sents:
            if len(sent) > min_sent_words and len(sent.text.strip()) > 0:
                valid_sents.append(sent.text)
                spacy_feats_list.append(spacy_to_tuple(sent))

        if not valid_sents:
            continue

        references = [summary_text] * len(valid_sents)
        
        # --- THE FIX ---
        # Instead of calling bert_scorer(), we call scorer.score()
        # on our pre-loaded global object. No loading reports!
        _, _, F1 = scorer.score(valid_sents, references)
        
        f1_scores = F1.tolist()

        sent_data = []
        for sent_text, text_feats, f1_score in zip(valid_sents, spacy_feats_list, f1_scores):
            fake_rscores = {
                'rouge-1': {'f': f1_score, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': f1_score, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': f1_score, 'p': 0.0, 'r': 0.0}
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


