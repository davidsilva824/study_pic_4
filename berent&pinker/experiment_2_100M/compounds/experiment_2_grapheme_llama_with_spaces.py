### This code is complete. (MINIMAL change: don’t assume a BOS token is present in `tokens`)

import pandas as pd
from minicons import scorer

models = [
    "bbunzeck/grapheme-llama"
]

BOS = True

# --- CHANGED: read compounds from CSV instead of hardcoded list ---
stimuli_file = "berent&pinker/stimuli_compounds_experiment_2.csv"  # adjust path if needed
stimuli_df = pd.read_csv(stimuli_file)
# ---------------------------------------------------------------

cat_labels = {
    "a": "singular_1",
    "b": "plural_1",
    "c": "singular_2",
    "d": "plural_2",
}

def process_pairs(lm, pairs, data):
    # --- CHANGED: loop over compounds from CSV ---
    for _, row in stimuli_df.iterrows():
        suffix = str(row["story_id"]).strip().split("_")[-1]
        category_name = cat_labels[suffix]

        sentence = str(row["compound"]).strip()  # if your column is story_text, change only this name
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
        print(f"{sentence}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")

# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

    data = []
    process_pairs(lm, None, data)

    output_file = f"results_berent&pinker/results_experiment_2_grapheme_llama_compounds.csv"

    df = pd.DataFrame(data, columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"])
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")