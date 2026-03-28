### This code is complete. (MINIMAL change: score full stories, then extract surprisal for the target compound inside the story)

import pandas as pd
from minicons import scorer

models = [
    "bbunzeck/grapheme-llama"
]

BOS = True

# --- Read BOTH CSVs ---
compounds_file = "berent&pinker/stimuli_compounds_experiment_2.csv"   # has: story_id, compound
stories_file   = "berent&pinker/stimuli_stories_experiment_2.csv"     # has: story_id, story_text

compounds_df = pd.read_csv(compounds_file)
stories_df = pd.read_csv(stories_file)

# Merge by story_id so each row has the target compound + full story context
stimuli_df = compounds_df.merge(stories_df, on="story_id", how="inner")
# ------------------------------------------

cat_labels = {
    "a": "singular_1",
    "b": "plural_1",
    "c": "singular_2",
    "d": "plural_2",
}


def _is_special_token(tok):
    tok = str(tok)
    return tok.startswith("<") and tok.endswith(">")


def _token_to_surface_piece(tok):
    """
    Convert token string to an approximate surface-text piece for alignment.
    Keeps spaces when encoded as token-prefix markers.
    """
    tok = str(tok)

    # Common word-boundary markers (GPT-style and SentencePiece-style)
    if tok.startswith("Ġ"):
        return " " + tok[1:]
    if tok.startswith("▁"):
        return " " + tok[1:]

    return tok


def _clean_token_for_word_reconstruction(tok):
    """
    Remove word-boundary markers/spaces for word-level reconstruction (same idea as before).
    """
    tok = str(tok)
    return tok.lstrip("Ġ ▁ ")


def _find_compound_token_span(tokens, start_idx, compound_text):
    """
    Find token span [compound_start, compound_end) in the FULL story token list
    that corresponds to the target compound string (e.g., 'goose protector').

    Uses approximate surface reconstruction from tokens and char-span mapping.
    """
    # Build reconstructed text + char spans per token
    reconstructed_text = ""
    token_char_spans = []  # list of (token_index, start_char, end_char)

    for i in range(start_idx, len(tokens)):
        tok = str(tokens[i])

        # skip explicit special tokens
        if _is_special_token(tok):
            token_char_spans.append((i, len(reconstructed_text), len(reconstructed_text)))
            continue

        piece = _token_to_surface_piece(tok)
        s = len(reconstructed_text)
        reconstructed_text += piece
        e = len(reconstructed_text)
        token_char_spans.append((i, s, e))

    # Try exact search first
    pos = reconstructed_text.find(compound_text)

    # Fallback: ignore double-spacing differences
    if pos == -1:
        compact_text = " ".join(reconstructed_text.split())
        compact_target = " ".join(compound_text.split())
        # If normalization changes text length, mapping becomes unreliable; keep strict behavior
        # so we fail clearly instead of silently extracting wrong span.
        if compact_text == reconstructed_text and compact_target == compound_text:
            pos = reconstructed_text.find(compound_text)

    if pos == -1:
        raise ValueError(f"Could not find compound '{compound_text}' in token-reconstructed story text.")

    target_start = pos
    target_end = pos + len(compound_text)

    # Map char span to token span
    compound_token_indices = []
    for tok_i, s, e in token_char_spans:
        if e <= target_start:
            continue
        if s >= target_end:
            break
        if e > s:  # ignore zero-length spans (special tokens)
            compound_token_indices.append(tok_i)

    if not compound_token_indices:
        raise ValueError(f"Found compound text but could not map to tokens: '{compound_text}'")

    return compound_token_indices[0], compound_token_indices[-1] + 1  # [start, end)


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
            bow_correction=True
        )[0]

        tokens = [tok for tok, s, *_ in tok_scores]
        surprisal_values = [s for tok, s, *_ in tok_scores]

        # --- Original Print Block (full story tokens) ---
        print(' '.join(f'{str(tok):>10}' for tok in tokens))
        print(' '.join(f'{s:>10.3f}' for s in surprisal_values))
        print(surprisal_values)

        # --- MINIMAL FIX: set where "real tokens" start ---
        cleaned_tokens = [_clean_token_for_word_reconstruction(tok) for tok in tokens]
        start_idx = 1 if (len(cleaned_tokens) > 0 and str(cleaned_tokens[0]).startswith("<")) else 0

        # --- NEW: find where the target compound occurs in the FULL story tokens ---
        compound_start_idx, compound_end_idx = _find_compound_token_span(tokens, start_idx, compound)

        # Reconstruct only inside the compound span to split non-head/head exactly like before
        non_n = 0
        reconstructed_word = ""

        compound_cleaned_tokens = [_clean_token_for_word_reconstruction(tok) for tok in tokens[compound_start_idx:compound_end_idx]]

        for k in range(len(compound_cleaned_tokens)):
            reconstructed_word += compound_cleaned_tokens[k]
            non_n += 1
            if reconstructed_word == non_head:
                break

        total_compound_tokens = len(compound_cleaned_tokens)
        head_n = total_compound_tokens - non_n

        # Sum surprisal in the matched compound span
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

    data = []
    process_pairs(lm, None, data)

    output_file = "results_berent&pinker/results_experiment_2_grapheme_llama_stories.csv"

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")