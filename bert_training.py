import pickle
import glob
import torch
import pandas as pd
import numpy as np
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# extract from pickles
def load_billsum_pickles(pattern, rouge_threshold=0.1):
    """
    Translates the BillSum legacy pickle format into a flat table.
    threshold: The ROUGE score required to mark a sentence as 'summary-worthy' (label 1).
    """
    rows = []
    for file in glob.glob(pattern):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            for bill_id, sentence_list in data.items():
                for sent_text, spacy_tuple, rouge_scores in sentence_list:
                    # The paper focuses on ROUGE-L or ROUGE-2
                    score = rouge_scores['rouge-l']['f']
                    label = 1 if score >= rouge_threshold else 0
                    rows.append({"text": sent_text, "label": label})
    
    return pd.DataFrame(rows)

# prep data
print("Loading legacy pickle data...")
train_df = load_billsum_pickles("billsum/data/sent_data/us_train.pickle")
test_df  = load_billsum_pickles("billsum/data/sent_data/us_test.pickle")

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 'bert-large-uncased' matches the paper, but 'nlpaueb/legal-bert-base-uncased' is faster/better
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("Tokenizing...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# trainer setup
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./billsum_bert_results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none" # Prevents needing a Weights & Biases account
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

# run
print("Starting training...")
trainer.train()

# predictions on test
print("Generating predictions for evaluate_bert.py...")
predictions = trainer.predict(tokenized_test)

# extract logits
# BERT outputs two numbers per sentence: [prob_of_0, prob_of_1]: only extract 1
probs = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions), dim=-1)
probabilities_of_one = probs[:, 1].numpy()

# save in legacy format (.tsv with no header)
# evaluate_bert.py expects a single column of probabilities
os.makedirs("BERT_CLASSIFIER_DIR", exist_ok=True)

pd.DataFrame(probabilities_of_one).to_csv(
    "BERT_CLASSIFIER_DIR/us_test_results.tsv", 
    sep='\t', 
    index=False, 
    header=False
)

print("Done! us_test_results.tsv is ready for evaluate_bert.py.")