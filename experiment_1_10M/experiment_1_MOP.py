### This code seems to be working, but it wouldnt hurt a final reverification. 
# Uses Morphologically-aware tokenization via MorPiece: https://huggingface.co/NeTS-lab/babylm-mop-10m-gpt2 
# The tokenization here is particular. 
# Subword continuation is marked '++' instead of marking the word separation.
# This means that the information about the new word is already in the right place. Making the BOW correction unecessary.  

import pandas as pd
from minicons import scorer
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


models = [
    "NeTS-lab/babylm-mop-10m-gpt2"
]

BOS = True
output_file = "results_experiment_1_10M/results_experiment_1_MOP.csv"

# Obtaining the compounds from the json file. 
with open("compounds_experiment_1.json", "r", encoding="utf-8") as f:
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
                    bow_correction=False
                )[0]
                
                tokens = [tok for tok, s, *_ in tok_scores]
                surprisal_values = [s for tok, s, *_ in tok_scores]
                
                print(' '.join(f'{tok:>10}' for tok in tokens))
                print(' '.join(f'{s:>10.3f}' for s in surprisal_values))
                print(surprisal_values)
                
                non_n = 0
                reconstructed_word = ""

                cleaned_tokens = [tok.lstrip('Ġ ').replace('++', '') for tok in tokens]

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


for model_name in models:
    print(f"\nLoading model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        return_dict=True
    )

    lm = scorer.IncrementalLMScorer(
        model,
        tokenizer=tokenizer,
        device="cpu"
    )
    
    data = []
    
    process_pairs(lm, None, data)
   
    
    df = pd.DataFrame(data, columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"])
    df.to_csv(output_file, index=False)
    
    print(f'\nresults in results_experiment_1_10M folder.\n')