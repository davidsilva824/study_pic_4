### This code seems to be working, but it wouldnt hurt a final reverification. 
### Change to vertical output. 

# BOS = False
# Forced BOW settings block, for separator token "W"
# Splitting head and non-head based on the token W.
# IMPORTANT note: The BOW correction is working. This can be observe in the file 'suprisal_by_token_babble_txt_char_with_spaces.py'.
# With the correction the suprisal of the token 'W' becomes zero. 

import pandas as pd
from minicons import scorer
from collections import defaultdict
import json


models = [
    "phonemetransformers/GPT2-85M-CHAR-TXT"
]

BOS = False
output_file = "results_berent&pinker/100M/results_experiment_3_babble_txt_char_with_spaces.csv"

with open("berent&pinker/compounds_experiment_3.json", "r", encoding="utf-8") as f:
    compound_groups_data = json.load(f)

compound_groups = [
    (group["non_heads"], group["heads"])
    for group in compound_groups_data
]

cat_labels = {
    0: "Sibilant Singular",
    1: "Sibilant Plural",
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
                print(' '.join(f'{tok:>10}' for tok in tokens))
                print(' '.join(f'{s:>10.3f}' for s in surprisal_values))
                print(surprisal_values)

                cleaned_tokens = [tok.lstrip('Ġ ') for tok in tokens]

                # --- MINIMAL FIX: set where "real tokens" start ---
                start_idx = 1 if (len(cleaned_tokens) > 0 and cleaned_tokens[0].startswith("<")) else 0

                # =========================
                # FIX: split using boundary token "W"
                # =========================
                boundary_tok = "W"

                # Find the first boundary token after start_idx (this is the space between words)
                boundary_pos = None
                for k in range(start_idx, len(cleaned_tokens)):
                    if cleaned_tokens[k] == boundary_tok:
                        boundary_pos = k
                        break

                # If no boundary token is found, fallback to old behaviour (treat everything as non-head)
                if boundary_pos is None:
                    non_n = len(cleaned_tokens) - start_idx
                    head_n = 0
                    surprisal_non_head = sum(surprisal_values[start_idx : start_idx + non_n])
                    surprisal_head = 0
                else:
                    # Non-head tokens are from start_idx up to boundary_pos (excluding boundary token),
                    # but we should ignore any boundary tokens inside (just in case).
                    non_head_indices = [k for k in range(start_idx, boundary_pos) if cleaned_tokens[k] != boundary_tok]

                    # Head tokens are after boundary_pos, ignoring boundary tokens.
                    head_indices = [k for k in range(boundary_pos + 1, len(cleaned_tokens)) if cleaned_tokens[k] != boundary_tok]

                    surprisal_non_head = sum(surprisal_values[k] for k in non_head_indices)
                    surprisal_head = sum(surprisal_values[k] for k in head_indices)
                # =========================

                data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])
                print(f"{sentence}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")

# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

    # =========================
    # ADDED: forced BOW settings, for "W" boundary token
    # =========================
    bow_symbol = "W"
    lm.is_bow_tokenizer = True
    lm.bow_symbol = bow_symbol

    bow_subwords = defaultdict(bool)

    vocab = lm.tokenizer.get_vocab()
    bow_id = vocab.get(bow_symbol, None)

    for _, idx in vocab.items():
        bow_subwords[idx] = False

    for idx in lm.tokenizer.get_added_vocab().values():
        bow_subwords[idx] = False

    if bow_id is not None:
        bow_subwords[bow_id] = True

    lm.bow_subwords = bow_subwords
    lm.bow_subword_idx = [int(bow_id)] if bow_id is not None else []

    print("bow_symbol =", bow_symbol)
    print("bow_id =", bow_id)
    print("len(bow_subword_idx) =", len(lm.bow_subword_idx))
    print("Forced BOW settings applied successfully.")
    print("-" * 30)
    # =========================

    data = []
    process_pairs(lm, None, data)


    df = pd.DataFrame(data, columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"])
    df.to_csv(output_file, index=False)

    print(f'\nresults in results_berent&pinker folder.\n')