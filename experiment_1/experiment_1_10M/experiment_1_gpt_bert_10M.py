### This code is complete (minimal adaptation: tuple-output wrapper + trust_remote_code).
# BOS = False
# Has a special wrap around the 'scorer.IncrementalLMScorer' method because the model does not save the logits in the place minicons expects.
# must have trust_remote_code = True. 
# Check the methods better in  'surprisal_by_token_gpt_bert.py'

import pandas as pd
from types import SimpleNamespace
from minicons import scorer
import json


models = [
    "BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus",
    "BabyLM-community/babylm-baseline-10m-gpt-bert-mixed",
    "BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus"
]

BOS = False

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

# --- MINIMAL FIX: wrap tuple outputs so minicons can read .logits ---
class _WrapOutputsWithLogits:
    def __init__(self, model):
        self._m = model

    def __call__(self, *args, **kwargs):
        out = self._m(*args, **kwargs)
        if hasattr(out, "logits"):
            return out
        if isinstance(out, tuple):
            return SimpleNamespace(logits=out[0])
        return out

    def __getattr__(self, name):
        return getattr(self._m, name)
# --- end fix ---

def process_pairs(lm, pairs, data):
    
    for non_heads, heads in compound_groups:
        # Loop over HEADS first
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
                # --- Original Sentence Print ---
                print(f"{sentence}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")


# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda", trust_remote_code=True)

    # apply tuple-output wrapper
    lm.model = _WrapOutputsWithLogits(lm.model)
    
    data = []
    
    process_pairs(lm, None, data)
    
    # Determine filename based on model to match your style
    if "causal" in model_name:
        output_file = "results_experiment_1/10M/results_experiment_1_gpt_bert_10M_causal.csv"
    
    elif "mixed" in model_name:
        output_file = "results_experiment_1/10M/results_experiment_1_gpt_bert_10M_mixed.csv"

    else:
        output_file = "results_experiment_1/10M/results_experiment_1_gpt_bert_10M_masked.csv"
    
    df = pd.DataFrame(data, columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"])
    df.to_csv(output_file, index=False)
    
    print(f'\nresults in results_experiment_1 folder.\n')