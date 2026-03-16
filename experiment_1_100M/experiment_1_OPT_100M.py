### This code is complete. 
# BOS = False
# Normal BOW
# Não existe OPT de 2024. Penso que foi porque não teve resultados suficientemente bons. 

import pandas as pd
from minicons import scorer

models = [
    "babylm/opt-125m-strict-2023"
]

BOS = False

output_file = f"results_experiment_1_100M/results_experiment_1_OPT_100M.csv"


compound_groups = [
    (['goose', 'geese', 'swan', 'swans'],
     ['protector', 'trader', 'tracker', 'expert']),

    (['ox', 'oxen', 'cow', 'cows'],
     ['register', 'trader', 'tracker', 'finder']),

    (['louse', 'lice', 'flea', 'fleas'],
     ['issue', 'trader', 'tracker', 'expert']),

    (['mouse', 'mice', 'rat', 'rats'],
     ['issue', 'trader', 'tracker', 'inspector']),

    (['foot', 'feet', 'leg', 'legs'],
     ['issue', 'examination', 'expert', 'inspector']),

    (['tooth', 'teeth', 'bone', 'bones'],
     ['issue', 'examination', 'expert', 'protector']),

    (['child', 'children', 'adult', 'adults'],
     ['patrol', 'register', 'institute', 'crew']),

    (['woman', 'women', 'girl', 'girls'],
     ['protector', 'register', 'hangout', 'crew']),

    (['man', 'men', 'boy', 'boys'],
     ['institute', 'register', 'finder', 'hangout']),

    (['salesman', 'salesmen', 'retailer', 'retailers'],
     ['institute', 'inspector', 'protector', 'employer']),

    (['nobleman', 'noblemen', 'aristocrat', 'aristocrats'],
     ['patrol', 'hangout', 'institute', 'crew']),

    (['boatman', 'boatmen', 'shipmate', 'shipmates'],
     ['patrol', 'finder', 'inspector', 'employer']),

    (['craftsman', 'craftsmen', 'labourer', 'labourers'],
     ['employer', 'examination', 'hangout', 'finder']),
    
    (['fireman', 'firemen', 'lifeguard', 'lifeguards'],
     ['examination', 'employer', 'crew', 'patrol'])
]

cat_labels = {
    0: "Irregular Singular",
    1: "Irregular Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

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
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")
    
    data = []
    
    process_pairs(lm, None, data)
    

    df = pd.DataFrame(data, columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"])
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_experiment_1_100M folder.\n")