### This code is complete.

import pandas as pd
from minicons import scorer
from collections import defaultdict
# -----------------------------------------------

models = [
    "babylm/opt-125m-strict-small-2023"
]

BOS = False

stimuli_file = "berent&pinker/stimuli_compounds_experiment_2.csv"
output_file = "results_berent&pinker/results_experiment_2_OPT_10M_compounds.csv"

stimuli_df = pd.read_csv(stimuli_file)
# ---------------------------------------------------------------

cat_labels = {
    "a": "singular_1",
    "b": "plural_1",
    "c": "singular_2",
    "d": "plural_2",
}

# --- Function to force BOW settings for this model ---
def force_bow_settings(lm, bow_symbol="Ġ"):
    lm.is_bow_tokenizer = True
    lm.bow_symbol = bow_symbol

    bow_subwords = defaultdict(bool)

    for word, idx in lm.tokenizer.get_vocab().items():
        bow_subwords[idx] = (len(word) > 0 and word[0] == bow_symbol)

    for idx in lm.tokenizer.get_added_vocab().values():
        bow_subwords[idx] = False

    lm.bow_subwords = bow_subwords
    lm.bow_subword_idx = [k for k, v in lm.bow_subwords.items() if v]
# ----------------------------------------------------------


def process_pairs(lm, pairs, data):
    
    # --- CHANGED: loop over compounds from CSV ---
    for _, row in stimuli_df.iterrows():
        suffix = str(row["story_id"]).strip().split("_")[-1]
        category_name = cat_labels[suffix]

        sentence = str(row["compound"]).strip()
        non_head, head = sentence.split(" ", 1)
        # ------------------------------------------
                
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

    print(f"\nresults in results_berent&pinker folder.\n")