### This code is complete. 
# BOS = True
# Normal BOW

import pandas as pd
from minicons import scorer
import json


models = [
    "aakarsh-nair/rerun-09-19-2024-experiment-distill-tree-babylm2024-58-1",
    "aakarsh-nair/rerun-09-19-2024-experiment-distill-tree-babylm2024-95-2",
    "aakarsh-nair/rerun-09-19-2024-experiment-distill-tree-babylm2024-360-2"
]

BOS = True

# Obtaining the compounds from the json file. 
with open("berent&pinker/compounds_experiment_3.json", "r", encoding="utf-8") as f:
    compound_groups_data = json.load(f)

compound_groups = [
    (group["non_heads"], group["heads"])
    for group in compound_groups_data
]

# Mapping
cat_labels = {
    0: "Sibilant Singular",
    1: "Sibilant Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

def process_pairs(lm, pairs, data):
    
    for non_heads, heads in compound_groups:
        # Loop over HEADS first
        for head in heads:
            # Then loop over NON-HEADS
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
                
                print(' '.join(f'{tok:>10}' for tok in tokens))
                print(' '.join(f'{s:>10.3f}' for s in surprisal_values))
                print(surprisal_values)
                
                non_n = 0
                reconstructed_word = ""

                cleaned_tokens = [tok.lstrip('Ġ ') for tok in tokens]

                for k in range(1, len(cleaned_tokens)):
                    reconstructed_word += cleaned_tokens[k]
                    non_n += 1
                    if reconstructed_word == non_head:
                        break
                
                total_real_tokens = len(tokens) - 1
                head_n = total_real_tokens - non_n

                surprisal_non_head = sum(surprisal_values[1 : 1 + non_n])
                surprisal_head = sum(surprisal_values[1 + non_n : 1 + non_n + head_n])

                data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])

                print(f"{sentence}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")


# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")
    
    data = []
    
    process_pairs(lm, None, data)
    
    # Determine filename based on model to match your style
    if "58" in model_name:
        output_file = "results_berent&pinker/10M/results_experiment_3_distill_tree__58_10M.csv"
    
    elif "95" in model_name:
        output_file = "results_berent&pinker/10M/results_experiment_3_distill_tree__95_10M.csv"

    else:
        output_file = "results_berent&pinker/10M/results_experiment_3_distill_tree__360_10M.csv.csv"
    
    df = pd.DataFrame(data, columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"])
    df.to_csv(output_file, index=False)
    
    print(f'\nresults in results_berent&pinker folder.\n')