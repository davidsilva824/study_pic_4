# This code is complete. 
# BOS = True
# Normal Bow correction is working, as can be seen by the 0 surprisal atributed to spaces, with bow_correction=True.


import pandas as pd
from minicons import scorer
import json


models = [
    "bbunzeck/grapheme-llama"
]

BOS = True
output_file = "results_experiment_1/100M/results_experiment_1_grapheme_llama_with_spaces.csv"


# Obtaining the compounds from the json file.
with open("experiment_1/compounds_experiment_1.json", "r", encoding="utf-8") as f:
    compound_groups_data = json.load(f)

compound_groups = [
    (group["non_heads"], group["heads"])
    for group in compound_groups_data
]

cat_labels = {
    0: "Irregular Singular",
    1: "Irregular Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

def process_pairs(lm, pairs, data):
    for non_heads, heads in compound_groups:
        for head in heads:
            for i, non_head in enumerate(non_heads):
                category_name = cat_labels[i]
                sentence = f"{non_head} {head}"

                tok_scores = lm.token_score(
                    sentence,
                    bos_token=BOS,
                    prob=False,
                    surprisal=True,
                    bow_correction=True
                )[0]

                tokens = [tok for tok, s, *_ in tok_scores]
                surprisal_values = [s for tok, s, *_ in tok_scores]

                # --- Original Print Block ---
                print("TOK_SCORES:")
                for tok, s in tok_scores:
                    print(f"{repr(tok):<20} {s:.7f}")

                non_n = 0
                reconstructed_word = ""

                cleaned_tokens = [tok.lstrip('Ġ ') for tok in tokens]

                # --- MINIMAL FIX: set where "real tokens" start ---
                # If first token is a special BOS-like token, skip it; otherwise start at 0.
                start_idx = 1 if (len(cleaned_tokens) > 0 and cleaned_tokens[0].startswith("<")) else 0

                for k in range(start_idx, len(cleaned_tokens)):
                    reconstructed_word += cleaned_tokens[k]
                    non_n += 1
                    if reconstructed_word == non_head:
                        break

                total_real_tokens = len(tokens) - start_idx
                head_n = total_real_tokens - non_n

                surprisal_non_head = sum(surprisal_values[start_idx : start_idx + non_n])
                surprisal_head = sum(surprisal_values[start_idx + non_n : start_idx + non_n + head_n])

                data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])
                print(f"  Non-Head ({non_head}): {surprisal_non_head}")
                print(f"  Head     ({head}): {surprisal_head}")

# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

    data = []
    process_pairs(lm, None, data)

    output_file = f"results_experiment_1/100M/results_experiment_1_grapheme_llama_with_spaces.csv"

    df = pd.DataFrame(data, columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"])
    df.to_csv(output_file, index=False)

    print(f'\nresults in results_experiment_1 folder.\n')