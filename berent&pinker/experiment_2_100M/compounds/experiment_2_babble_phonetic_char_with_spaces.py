### This code is complete.
### MINIMAL FIX (file-based version):
### - keep file workflow
### - convert each compound to IPA with WORD_BOUNDARY
### - split Non-Head vs Head using token "WORD_BOUNDARY" (not orthographic offsets)

import pandas as pd
from g2p_plus import transcribe_utterances
from minicons import scorer

models = [
    "phonemetransformers/GPT2-85M-CHAR-PHON"
]

BOS = False

# ---- INPUT / OUTPUT ----
input_file = "berent&pinker/stimuli_compounds_experiment_2.csv"
output_file = "results_berent&pinker/results_experiment_2_babble_phonetic_char_compounds.csv"

# ---- CHANGE ONLY IF YOUR COLUMN NAME IS DIFFERENT ----
# Your error showed 'compounds' was wrong. Use the real column name here.
# Example possibilities: "compound", "Compound", "stimulus", "item"
compound_column = "compound"

def to_ipa_with_boundaries(text):
    ipa_list = transcribe_utterances(
        [text],
        backend="phonemizer",
        language="en-us",
        keep_word_boundaries=True
    )
    return ipa_list[0]

def split_surprisal_by_word_boundary(tok_scores):
    tokens = [tok for tok, s, *_ in tok_scores]
    surprisal_values = [s for tok, s, *_ in tok_scores]

    # skip UTT boundary if present
    start_idx = 1 if (len(tokens) > 0 and str(tokens[0]).startswith("UTT_")) else 0

    boundary_tok = "WORD_BOUNDARY"

    # first boundary = between non-head and head
    first_boundary = None
    for k in range(start_idx, len(tokens)):
        if tokens[k] == boundary_tok:
            first_boundary = k
            break

    if first_boundary is None:
        raise ValueError("WORD_BOUNDARY not found in token sequence.")

    # second boundary = end of head (usually present)
    second_boundary = None
    for k in range(first_boundary + 1, len(tokens)):
        if tokens[k] == boundary_tok:
            second_boundary = k
            break

    non_head_indices = [k for k in range(start_idx, first_boundary) if tokens[k] != boundary_tok]
    head_end = second_boundary if second_boundary is not None else len(tokens)
    head_indices = [k for k in range(first_boundary + 1, head_end) if tokens[k] != boundary_tok]

    surprisal_non_head = sum(surprisal_values[k] for k in non_head_indices)
    surprisal_head = sum(surprisal_values[k] for k in head_indices)

    return surprisal_non_head, surprisal_head

def process_file_rows(lm, df, data):
    for _, row in df.iterrows():
        sentence = str(row[compound_column]).strip()

        ipa_text = to_ipa_with_boundaries(sentence)

        tok_scores = lm.token_score(
            ipa_text,
            bos_token=BOS,
            prob=False,
            surprisal=True,
            bow_correction=False
        )[0]

        tokens = [tok for tok, s, *_ in tok_scores]
        surprisal_values = [s for tok, s, *_ in tok_scores]

        # --- Original Print Block ---
        print(' '.join(f'{tok:>10}' for tok in tokens))
        print(' '.join(f'{s:>10.3f}' for s in surprisal_values))
        print(surprisal_values)

        # --- FIXED split (phonetic + WORD_BOUNDARY) ---
        surprisal_non_head, surprisal_head = split_surprisal_by_word_boundary(tok_scores)

        data.append([sentence, ipa_text, surprisal_non_head, surprisal_head])
        print(f"{sentence} | IPA: {ipa_text} | Non-Head: {surprisal_non_head}, Head: {surprisal_head}")

# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

    df_in = pd.read_csv(input_file)

    data = []
    process_file_rows(lm, df_in, data)

    df_out = pd.DataFrame(data, columns=["Compound", "IPA", "Surprisal Non-head", "Surprisal head"])
    df_out.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")