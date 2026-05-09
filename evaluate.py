import os
import pickle
import pandas as pd
import numpy as np
from rouge import Rouge

rouge = Rouge()

prefix = os.environ["BILLSUM_PREFIX"]
sent_data_path = os.path.join(prefix, "sent_data")


def load_sent_data(file_path):
    with open(file_path, "rb") as f:
        return list(pickle.load(f).items())


def evaluate_lead3(sent_file, json_file, summary_field):

    print(f"\nEvaluating Lead-3 on {sent_file}")

    sent_data = load_sent_data(sent_file)

    docs = pd.read_json(json_file, lines=True)
    docs.set_index("bill_id", inplace=True)

    scores = {}

    for bill_id, sents in sent_data:

        if bill_id not in docs.index:
            continue

        sent_texts = [s[0] for s in sents]

        # ---- LEAD-3 BASELINE ----
        summary = " ".join(sent_texts[:3])

        reference = docs.loc[bill_id][summary_field]

        if not isinstance(reference, str) or not reference.strip():
            continue

        scores[bill_id] = rouge.get_scores([summary], [reference])[0]

    print(f"Evaluated docs: {len(scores)}")
    return scores


def summarize(scores):
    vals = list(scores.values())
    return {
        "rouge-1": np.mean([v["rouge-1"]["f"] for v in vals]),
        "rouge-2": np.mean([v["rouge-2"]["f"] for v in vals]),
        "rouge-l": np.mean([v["rouge-l"]["f"] for v in vals]),
    }


# ---- RUN ----
us_lead3 = evaluate_lead3(
    os.path.join(sent_data_path, "us_test_sent_scores.pkl"),
    os.path.join(prefix, "clean_final/us_test_data_final.jsonl"),
    "summary"
)

ca_lead3 = evaluate_lead3(
    os.path.join(sent_data_path, "ca_test_sent_scores.pkl"),
    os.path.join(prefix, "clean_final/ca_test_data_final.jsonl"),
    "clean_summary"
)

print("\nLEAD-3 RESULTS")
print("US:", summarize(us_lead3))
print("CA:", summarize(ca_lead3))