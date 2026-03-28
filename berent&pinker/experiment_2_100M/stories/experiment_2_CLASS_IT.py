### This code is complete. (MINIMAL change: score full stories, then extract target compound surprisal in the same way)
### + Keeps forced BOW settings for Ġ
### + Reads compounds + stories CSVs and matches by story_id
### + MINIMAL FIX: don’t assume a BOS token is present in tokens

import pandas as pd
from minicons import scorer
from collections import defaultdict

models = [
    "colinglab/CLASS_IT-140M"
]

BOS = True

# --- Read BOTH CSVs ---
compounds_file = "berent&pinker/stimuli_compounds_experiment_2.csv"   # story_id, compound
stories_file   = "berent&pinker/stimuli_stories_experiment_2.csv"     # story_id, story_text
output_file    = "results_berent&pinker/results_experiment_2_CLASS_IT_stories.csv"

compounds_df = pd.read_csv(compounds_file)
stories_df = pd.read_csv(stories_file)

# Merge by story_id so each row has target compound + full story
stimuli_df = compounds_df.merge(stories_df, on="story_id", how="inner")
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


def _is_special_token(tok):
    tok = str(tok)
    return tok.startswith("<") and tok.endswith(">")


def _token_to_surface_piece_bpe(tok):
    """
    Reconstruct approximate text from BPE tokens.
    GPT2-style BPE often uses Ġ to indicate a leading space.
    """
    tok = str(tok)

    if _is_special_token(tok):
        return ""

    if tok.startswith("Ġ"):
        return " " + tok[1:]

    return tok


def _clean_token_for_word_reconstruction(tok):
    # same spirit as your original code
    return str(tok).lstrip("Ġ ")


def _find_compound_token_span(tokens, start_idx, compound_text):
    """
    Find [start, end) token span of the target compound inside FULL story tokens
    by reconstructing approximate surface text and mapping chars->tokens.
    """
    reconstructed_text = ""
    token_char_spans = []  # (token_index, start_char, end_char)

    for i in range(start_idx, len(tokens)):
        piece = _token_to_surface_piece_bpe(tokens[i])
        s = len(reconstructed_text)
        reconstructed_text += piece
        e = len(reconstructed_text)
        token_char_spans.append((i, s, e))

    pos = reconstructed_text.find(compound_text)
    if pos == -1:
        raise ValueError(f"Could not find compound '{compound_text}' in token-reconstructed story text.")

    target_start = pos
    target_end = pos + len(compound_text)

    span_indices = []
    for tok_i, s, e in token_char_spans:
        if e <= target_start:
            continue
        if s >= target_end:
            break
        if e > s:  # ignore zero-length (special token)
            span_indices.append(tok_i)

    if not span_indices:
        raise ValueError(f"Found compound text but could not map to tokens: '{compound_text}'")

    return span_indices[0], span_indices[-1] + 1


def process_pairs(lm, pairs, data):
    for _, row in stimuli_df.iterrows():
        suffix = str(row["story_id"]).strip().split("_")[-1]
        category_name = cat_labels[suffix]

        compound = str(row["compound"]).strip()
        story_text = str(row["story_text"]).strip()

        non_head, head = compound.split(" ", 1)

        # --- CHANGED: score FULL STORY instead of only compound ---
        tok_scores = lm.token_score(
            story_text,
            bos_token=BOS,
            prob=False,
            surprisal=True,
            bow_correction=False
        )[0]

        tokens = [tok for tok, s, *_ in tok_scores]
        surprisal_values = [s for tok, s, *_ in tok_scores]

        # --- Original Print Block (full story tokens) ---
        print(' '.join(f'{str(tok):>10}' for tok in tokens))
        print(' '.join(f'{s:>10.3f}' for s in surprisal_values))
        print(surprisal_values)

        cleaned_tokens = [_clean_token_for_word_reconstruction(tok) for tok in tokens]

        # --- MINIMAL FIX: don’t assume BOS is present ---
        start_idx = 1 if (len(cleaned_tokens) > 0 and str(cleaned_tokens[0]).startswith("<")) else 0

        # --- NEW: find target compound inside FULL story tokens ---
        compound_start_idx, compound_end_idx = _find_compound_token_span(tokens, start_idx, compound)

        # --- SAME SPLIT LOGIC AS BEFORE, but only inside matched compound span ---
        non_n = 0
        reconstructed_word = ""

        compound_cleaned_tokens = [
            _clean_token_for_word_reconstruction(tok)
            for tok in tokens[compound_start_idx:compound_end_idx]
        ]

        for k in range(len(compound_cleaned_tokens)):
            reconstructed_word += compound_cleaned_tokens[k]
            non_n += 1
            if reconstructed_word == non_head:
                break

        total_compound_tokens = len(compound_cleaned_tokens)
        head_n = total_compound_tokens - non_n

        surprisal_non_head = sum(surprisal_values[compound_start_idx : compound_start_idx + non_n])
        surprisal_head = sum(surprisal_values[compound_start_idx + non_n : compound_start_idx + non_n + head_n])

        data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])

        print(f"STORY ID: {row['story_id']}")
        print(f"COMPOUND: {compound}")
        print(f"Compound token span: [{compound_start_idx}, {compound_end_idx})")
        print(f"{compound}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")


# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

    # --- ADDED: apply forced BOW method for this model ---
    force_bow_settings(lm, bow_symbol="Ġ")
    # ----------------------------------------------------

    data = []

    process_pairs(lm, None, data)

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")