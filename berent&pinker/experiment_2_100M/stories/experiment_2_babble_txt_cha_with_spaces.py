### This code is not working!!! the stories mught be larger than the limit of input. 


# BOS = False
# Forced BOW settings block, for separator token "W"
# Splitting head and non-head based on the token W.
# IMPORTANT note: The BOW correction is working. This can be observe in the file 'suprisal_by_token_babble_txt_char_with_spaces.py'.
# With the correction the suprisal of the token 'W' becomes zero. 

import json
import pandas as pd
from minicons import scorer
from collections import defaultdict

models = [
    "phonemetransformers/GPT2-85M-CHAR-TXT"
]

BOS = False
json_file = "berent&pinker/compounds_with_stories_experiment_2.json"
output_file = "results_berent&pinker/100M/results_experiment_2_babble_txt_char_with_spaces_stories.csv"

with open(json_file, "r", encoding="utf-8") as f:
    stimuli_data = json.load(f)

cat_labels = {
    0: "Sibilant Singular",
    1: "Sibilant Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

def _find_compound_char_span(story_text, compound_text):
    matches = []
    start = 0
    while True:
        pos = story_text.find(compound_text, start)
        if pos == -1:
            break
        matches.append(pos)
        start = pos + 1

    if len(matches) == 0:
        raise ValueError(f"Could not find compound '{compound_text}' in story.")
    if len(matches) > 1:
        raise ValueError(f"Compound '{compound_text}' appears multiple times in story.")

    start_char = matches[0]
    end_char = start_char + len(compound_text)
    return start_char, end_char

def _find_compound_token_span(lm, story_text, tok_scores, compound_text):
    start_char, end_char = _find_compound_char_span(story_text, compound_text)

    enc = lm.tokenizer(
        story_text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    offsets = enc["offset_mapping"]

    first_tok = str(tok_scores[0][0]) if len(tok_scores) > 0 else ""
    start_idx = 1 if (first_tok.startswith("<") or first_tok == "UTT_BOUNDARY") else 0
    toks_only = tok_scores[start_idx:]

    if len(offsets) != len(toks_only):
        raise ValueError(f"Offsets/token mismatch: offsets={len(offsets)} vs toks={len(toks_only)}")

    span_indices = []
    for i, ((tok, s, *_), (tok_start, tok_end)) in enumerate(zip(toks_only, offsets), start=start_idx):
        if tok_end <= start_char:
            continue
        if tok_start >= end_char:
            break
        span_indices.append(i)

    if not span_indices:
        raise ValueError(f"Found compound text but could not map it to tokens: '{compound_text}'")

    return span_indices[0], span_indices[-1] + 1

def process_pairs(lm, data):
    for group in stimuli_data:
        non_heads = group["non_heads"]
        heads = group["heads"]
        stories = group["stories"]

        if len(heads) != 1:
            raise ValueError(f"Expected exactly one head in experiment 2 item, got {len(heads)}")

        head = str(heads[0]).strip()

        for i, (non_head, story_text) in enumerate(zip(non_heads, stories)):
            category_name = cat_labels[i]

            non_head = str(non_head).strip()
            story_text = str(story_text).strip()
            compound = f"{non_head} {head}"

            tok_scores = lm.token_score(
                story_text,
                bos_token=BOS,
                prob=False,
                surprisal=True,
                bow_correction=True
            )[0]

            tokens = [tok for tok, s, *_ in tok_scores]
            surprisal_values = [s for tok, s, *_ in tok_scores]

            print("\nTOK_SCORES:")
            for tok, s in tok_scores:
                print(f"{repr(tok):<20} {s:.7f}")

            compound_start_idx, compound_end_idx = _find_compound_token_span(
                lm, story_text, tok_scores, compound
            )

            compound_tokens = tokens[compound_start_idx:compound_end_idx]
            compound_surprisals = surprisal_values[compound_start_idx:compound_end_idx]

            cleaned_tokens = [str(tok).lstrip("Ġ ") for tok in compound_tokens]

            boundary_tok = "W"
            boundary_pos = None
            for k in range(len(cleaned_tokens)):
                if cleaned_tokens[k] == boundary_tok:
                    boundary_pos = k
                    break

            if boundary_pos is None:
                non_head_indices = [k for k in range(len(cleaned_tokens)) if cleaned_tokens[k] != boundary_tok]
                head_indices = []
                surprisal_non_head = sum(compound_surprisals[k] for k in non_head_indices)
                surprisal_head = 0
            else:
                non_head_indices = [k for k in range(boundary_pos) if cleaned_tokens[k] != boundary_tok]
                head_indices = [k for k in range(boundary_pos + 1, len(cleaned_tokens)) if cleaned_tokens[k] != boundary_tok]

                surprisal_non_head = sum(compound_surprisals[k] for k in non_head_indices)
                surprisal_head = sum(compound_surprisals[k] for k in head_indices)

            data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])

            print(f"COMPOUND: {compound}")
            print(f"Compound token span: [{compound_start_idx}, {compound_end_idx})")
            print(f"  Non-Head ({non_head}): {surprisal_non_head}")
            print(f"  Head     ({head}): {surprisal_head}")

# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

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

    data = []
    process_pairs(lm, data)

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")