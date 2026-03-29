### This code is complete.
# BOS = True
# IMPORTANT note: the tokenizer of the model does not remove the spaces automatically.
# Since this model does not use word separation at all, the BOW correction should be kept False. 
# With BOW correction true it also massively increases the surprisal of the last character. So avoid it. 


import pandas as pd
from minicons import scorer
import json


models = [
    "bbunzeck/grapheme-llama-no-whitespace"
]

BOS = True
output_file = "results_berent&pinker/100M/results_experiment_2_grapheme_llama_no_spaces.csv"

with open("berent&pinker/compounds_experiment_2.json", "r", encoding="utf-8") as f:
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

def split_surprisal_by_offsets(lm, sentence, tok_scores, boundary):
    enc = lm.tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    toks_only = tok_scores[1:]  # drop UTT_BOUNDARY
    if len(offsets) != len(toks_only):
        raise ValueError(f"Offsets/token mismatch: offsets={len(offsets)} vs toks={len(toks_only)}")

    non_head_sum = 0.0
    head_sum = 0.0

    for (tok, s, *_), (start, end) in zip(toks_only, offsets):
        if end <= boundary:
            non_head_sum += s
        else:
            head_sum += s

    return non_head_sum, head_sum

def process_pairs(lm, pairs, data):
    
    for non_heads, heads in compound_groups:
        # Loop over HEADS first
        for head in heads:
  
            for i, non_head in enumerate(non_heads):
                category_name = cat_labels[i]
                
                sentence = f"{non_head}{head}"
                boundary = len(non_head)
                
                tok_scores = lm.token_score(
                    sentence,
                    bos_token=BOS,
                    prob=False,
                    surprisal=True,
                    bow_correction=False
                )[0]
                
                # --- Original Print Block ---
                print("TOK_SCORES:")
                for tok, s in tok_scores:
                    print(f"{repr(tok):<20} {s:.7f}")
                
                surprisal_non_head, surprisal_head = split_surprisal_by_offsets(
                    lm, sentence, tok_scores, boundary
                )

                data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])
                # --- Original Sentence Print ---
                print(f"  Non-Head ({non_head}): {surprisal_non_head}")
                print(f"  Head     ({head}): {surprisal_head}")


# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")
    
    data = []
    
    process_pairs(lm, None, data)
    

    df = pd.DataFrame(data, columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"])
    df.to_csv(output_file, index=False)

    print(f'\nresults in results_berent&pinker folder.\n')