import pickle
import glob
import torch
import pandas as pd
import numpy as np
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# --- FIX 1: DYNAMIC PATHS ---
prefix = os.getenv("BILLSUM_PREFIX", "/home/gracenewton/BillSum/data")
os.makedirs(os.path.join(prefix, 'score_data'), exist_ok=True)
sent_data_path = os.path.join(prefix, "sent_data")
output_path = "billsum_bert_results"

def load_billsum_pickles(pattern, bert_threshold=0.85):
    """
    pattern: path to the pickle file
    rouge_threshold: The score needed to mark a sentence as '1'
    """
    rows = []
    files = glob.glob(pattern)
    if not files:
        print(f"Warning: No files found for pattern {pattern}")
        return pd.DataFrame()

    print(f"Processing {pattern} using BertScore for Oracle labels...")
    
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            for bill_id, sentence_list in data.items():
                for sent_text, spacy_tuple, rouge_scores in sentence_list:
                    score = rouge_scores['rouge-2']['f'] # here, rouge-2 will still point to BertScore F1
                    
                    label = 1 if score >= bert_threshold else 0
                    rows.append({"text": sent_text, "label": label})
    
    return pd.DataFrame(rows)

# prep data
print(f"Loading legacy pickle data from {sent_data_path}...")
train_df = load_billsum_pickles(os.path.join(sent_data_path, "us_train_sent_scores.pkl"))
test_df  = load_billsum_pickles(os.path.join(sent_data_path, "us_test_sent_scores.pkl"))
ca_test_df = load_billsum_pickles(os.path.join(sent_data_path, "ca_test_sent_scores.pkl"))
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
ca_test_dataset = Dataset.from_pandas(ca_test_df)

model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("Tokenizing...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_ca = ca_test_dataset.map(tokenize_function, batched=True)

# trainer setup
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir=output_path,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

print("Starting training (Oracle: BertScore)...")
trainer.train()

# --- SAVE RESULTS ---
# --- SYNTHESIZED PREDICTION LOGIC ---

def save_results_aligned(dataset, filename):
    """
    Synthesizes the behavior of the original BillSum run_classifier.py:
    1. Outputs TWO columns (prob of 0, prob of 1)
    2. Uses Tab-separation
    3. No headers/index
    """
    print(f"Generating aligned predictions for {filename}...")
    
    # Ensure trainer.predict doesn't shuffle (shuffling is off by default for predict)
    preds = trainer.predict(dataset)
    
    # Apply Softmax to get probabilities for BOTH classes
    probs = torch.nn.functional.softmax(torch.from_numpy(preds.predictions), dim=-1).numpy()
    
    # Original script pathing: it puts them in a folder often called 'bert_data'
    # but we'll use your output_path for consistency
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, filename)
    
    # SAVE AS TWO COLUMNS (Class 0 \t Class 1)
    # This ensures evaluate_bert.py's predictions[1] finds the right data
    pd.DataFrame(probs).to_csv(
        out_file,
        sep='\t',
        index=False,
        header=False,
        float_format='%.8f' # High precision matching the original
    )
    print(f"Aligment complete: {out_file}")

# --- EXECUTE PREDICTIONS WITH ORIGINAL FILENAMES ---

# Note: The original script uses 'test_results.tsv' for US test
save_results_aligned(tokenized_test, "test_results_full_BSFast.tsv")

# Note: The original script uses 'ca_test_results.tsv'
save_results_aligned(tokenized_ca, "ca_test_results_full_BSFast.tsv")
