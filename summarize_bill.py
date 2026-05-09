#!/usr/bin/env python3
import os
import argparse
import nltk
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK sentence splitter is downloaded silently
nltk.download('punkt', quiet=True)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Applied Extractive Summarization Pipeline for Legislative Bills using Fine-Tuned Legal-BERT"
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the raw text file containing the legislative bill."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True, 
        help="Path where the generated summary file will be saved."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to your fine-tuned Legal-BERT checkpoint (e.g., path to full2, fullL, or fullBS)."
    )
    parser.add_argument(
        "--lambda_val", 
        type=float, 
        default=0.5, 
        help="MMR diversity trade-off parameter (0.0 = maximum diversity, 1.0 = maximum relevance). Default: 0.5"
    )
    parser.add_argument(
        "--summary_length", 
        type=int, 
        default=3, 
        help="Number of sentences to extract for the summary. Default: 3"
    )
    return parser.parse_args()

def mmr_selection(sentences, scores, lambda_val, summary_length):
    """
    Applies Maximal Marginal Relevance (MMR) selection over sentences 
    using TF-IDF cosine similarity to balance salience and diversity.
    """
    if len(sentences) <= summary_length:
        return sentences

    # Calculate TF-IDF matrix for pairwise cosine similarities
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    selected_indices = []
    candidate_indices = list(range(len(sentences)))

    # Phase 1: Select the single highest-scoring sentence unconditionally
    first_choice = int(np.argmax(scores))
    selected_indices.append(first_choice)
    candidate_indices.remove(first_choice)

    # Phase 2: Iteratively select remaining sentences maximizing MMR
    while len(selected_indices) < summary_length and candidate_indices:
        mmr_scores = []
        for cand in candidate_indices:
            relevance = scores[cand]
            
            # Find the maximum redundancy to any sentence already in the summary
            redundancy = max(similarity_matrix[cand, sel] for sel in selected_indices)
            
            # MMR formula
            mmr_score = (lambda_val * relevance) - ((1 - lambda_val) * redundancy)
            mmr_scores.append((mmr_score, cand))
        
        # Select candidate with the best balance
        best_score, best_cand = max(mmr_scores, key=lambda x: x[0])
        selected_indices.append(best_cand)
        candidate_indices.remove(best_cand)

    # Return sentences ordered by their original document appearance
    selected_indices.sort()
    return [sentences[idx] for idx in selected_indices]

def main():
    args = parse_arguments()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input bill file not found: {args.input_file}")

    print(f"Reading input bill text...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        print("Error: Input bill file is empty.")
        return

    # Pre-process: Segment the raw bill into clean sentences
    print("Segmenting document into sentences...")
    raw_sentences = nltk.sent_tokenize(raw_text)
    # Filter out empty strings or extremely short noise lines (under 5 characters)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]

    if not sentences:
        print("Error: No valid sentences extracted from input.")
        return

    print(f"Initializing fine-tuned Legal-BERT model from: {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    print("Computing sentence salience scores...")
    scores = []
    
    # Run batch inference to acquire binary classification logits (salient vs. non-salient)
    with torch.no_grad():
        for sent in sentences:
            inputs = tokenizer(
                sent, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(device)
            
            outputs = model(**inputs)
            # Apply Softmax to get the true probability distribution over classes
            probs = torch.softmax(outputs.logits, dim=-1)
            # Capture positive class probability (index 1: "salient")
            salience_prob = probs[0][1].item()
            scores.append(salience_prob)

    print("Executing MMR diversity selection...")
    summary_sentences = mmr_selection(
        sentences=sentences, 
        scores=scores, 
        lambda_val=args.lambda_val, 
        summary_length=args.summary_length
    )

    # Reconstruct final summary text
    summary_text = "\n".join(summary_sentences)

    print(f"Writing summary to: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + "="*50)
    print("                 GENERATED SUMMARY")
    print("="*50)
    print(summary_text)
    print("="*50)
    print("Process complete!")

if __name__ == "__main__":
    main()