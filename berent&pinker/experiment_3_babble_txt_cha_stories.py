# Not working!


### This code is complete. (MINIMAL change: score full stories, then extract target compound surprisal with same "W" boundary logic)
### + Keeps split Non-Head vs Head using boundary token "W"
### + Reads compounds + stories CSVs and matches by story_id
### + Recursive split fallback on story_text (split on spaces)
### + Manual boundary correction: add boundary token "W" surprisal to previous word
### + REMOVED forced BOW settings hack

import pandas as pd
from minicons import scorer

import torch

models = [
    "phonemetransformers/GPT2-85M-CHAR-TXT"
]

BOS = False

# --- Read BOTH CSVs ---
compounds_file = "berent&pinker/stimuli_compounds_experiment_3.csv"   # has: story_id, compound
stories_file   = "berent&pinker/stimuli_stories_experiment_3.csv"     # has: story_id, story_text
output_file = "results_berent&pinker/results_experiment_3_babble_txt_cha_stories.csv"

compounds_df = pd.read_csv(compounds_file)
stories_df = pd.read_csv(stories_file)

# Merge by story_id so each row has target compound + full story context
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


def _token_to_surface_piece_char_txt(tok):
    """
    Approximate surface reconstruction for GPT2-85M-CHAR-TXT tokens.
    This model uses 'W' as a word boundary marker (space). We map it to a real space
    so we can find the target compound in reconstructed full-story text.
    """
    tok = str(tok)

    # skip explicit special tokens in reconstruction
    if _is_special_token(tok):
        return ""

    # Model-specific separator token for word boundary / space
    if tok == "W":
        return " "

    return tok


def _find_compound_token_span(tokens, start_idx, compound_text):
    """
    Find token span [start, end) of the target compound inside FULL story tokens
    by reconstructing approximate surface text from tokens.
    """
    reconstructed_text = ""
    token_char_spans = []  # (token_index, start_char, end_char)

    for i in range(start_idx, len(tokens)):
        tok = str(tokens[i])

        piece = _token_to_surface_piece_char_txt(tok)
        s = len(reconstructed_text)
        reconstructed_text += piece
        e = len(reconstructed_text)

        token_char_spans.append((i, s, e))

    # Exact match in reconstructed text
    pos = reconstructed_text.find(compound_text)
    if pos == -1:
        raise ValueError(f"Could not find compound '{compound_text}' in token-reconstructed story text.")

    target_start = pos
    target_end = pos + len(compound_text)

    compound_token_indices = []
    for tok_i, s, e in token_char_spans:
        if e <= target_start:
            continue
        if s >= target_end:
            break

        if e > s:
            compound_token_indices.append(tok_i)

    if not compound_token_indices:
        raise ValueError(f"Found compound text but could not map it to tokens: '{compound_text}'")

    return compound_token_indices[0], compound_token_indices[-1] + 1  # [start, end)


def _safe_token_score_with_recursive_split(lm, text, bos_token, prob, surprisal, bow_correction):
    """
    Try scoring full story. If it fails (or returns empty), recursively split on the closest
    space to the middle until chunks fit. Then concatenate token scores.
    """
    def _score_recursive(s):
        s = str(s).strip()
        if s == "":
            return []

        try:
            out = lm.token_score(
                s,
                bos_token=bos_token,
                prob=prob,
                surprisal=surprisal,
                bow_correction=bow_correction
            )[0]

            # guard for weird empty outputs
            if out is None or len(out) == 0:
                raise ValueError("empty token_score output")

            return out

        except Exception:
            # split on nearest space to middle
            mid = len(s) // 2
            spaces = [i for i, ch in enumerate(s) if ch == " "]
            if not spaces:
                raise  # cannot split further

            split_idx = min(spaces, key=lambda i: abs(i - mid))

            left = s[:split_idx].strip()
            right = s[split_idx + 1:].strip()

            left_scores = _score_recursive(left) if left else []
            right_scores = _score_recursive(right) if right else []

            # concatenate chunks (keeps same simple logic as before)
            return left_scores + right_scores

    return _score_recursive(text)


def process_pairs(lm, pairs, data):
    for _, row in stimuli_df.iterrows():
        suffix = str(row["story_id"]).strip().split("_")[-1]
        category_name = cat_labels[suffix]

        compound = str(row["compound"]).strip()
        story_text = str(row["story_text"]).strip()
        non_head, head = compound.split(" ", 1)

        # --- CHANGED: score FULL STORY with recursive split fallback ---
        try:
            tok_scores = _safe_token_score_with_recursive_split(
                lm,
                story_text,
                bos_token=BOS,
                prob=False,
                surprisal=True,
                bow_correction=False
            )
        except Exception as e:
            print("[SKIP: token_score failed even after recursive split]")
            print(f"STORY ID: {row['story_id']}")
            print(f"COMPOUND: {compound}")
            print(f"STORY TXT: {story_text}")
            print(f"ERROR: {e}")
            print()
            continue

        tokens = [tok for tok, s, *_ in tok_scores]
        surprisal_values = [s for tok, s, *_ in tok_scores]

        # --- Original Print Block (full story tokens) ---
        print(' '.join(f'{str(tok):>10}' for tok in tokens))
        print(' '.join(f'{s:>10.3f}' for s in surprisal_values))
        print(surprisal_values)

        cleaned_tokens = [str(tok).lstrip('Ġ ') for tok in tokens]

        # --- MINIMAL FIX: set where "real tokens" start ---
        start_idx = 1 if (len(cleaned_tokens) > 0 and cleaned_tokens[0].startswith("<")) else 0

        # --- NEW: find target compound span inside FULL story tokens ---
        try:
            compound_start_idx, compound_end_idx = _find_compound_token_span(tokens, start_idx, compound)
        except Exception as e:
            print("[SKIP: compound not found in reconstructed full-story tokens]")
            print(f"STORY ID: {row['story_id']}")
            print(f"COMPOUND: {compound}")
            print(f"STORY TXT: {story_text}")
            print(f"ERROR: {e}")
            print()
            continue

        # =========================
        # SAME LOGIC AS BEFORE: split using boundary token "W"
        # but only inside the matched compound span
        # + MANUAL BOUNDARY CORRECTION: add W surprisal to previous word
        # =========================
        boundary_tok = "W"
        compound_cleaned_tokens = cleaned_tokens[compound_start_idx:compound_end_idx]

        # Find first boundary token inside the compound span
        boundary_pos_local = None
        for k in range(len(compound_cleaned_tokens)):
            if compound_cleaned_tokens[k] == boundary_tok:
                boundary_pos_local = k
                break

        if boundary_pos_local is None:
            # fallback (same spirit as before)
            non_head_indices_abs = [k for k in range(compound_start_idx, compound_end_idx) if cleaned_tokens[k] != boundary_tok]
            head_indices_abs = []
            surprisal_non_head = sum(surprisal_values[k] for k in non_head_indices_abs)
            surprisal_head = 0
        else:
            boundary_pos_abs = compound_start_idx + boundary_pos_local

            # Non-head: from start of matched span up to boundary token (excluding boundary)
            non_head_indices_abs = [
                k for k in range(compound_start_idx, boundary_pos_abs)
                if cleaned_tokens[k] != boundary_tok
            ]

            # Head: after boundary token to end of matched span (excluding boundary)
            head_indices_abs = [
                k for k in range(boundary_pos_abs + 1, compound_end_idx)
                if cleaned_tokens[k] != boundary_tok
            ]

            surprisal_non_head = sum(surprisal_values[k] for k in non_head_indices_abs)
            surprisal_head = sum(surprisal_values[k] for k in head_indices_abs)

            # MANUAL BOW-LIKE CORRECTION:
            # add the compound boundary token "W" surprisal to the previous word (Non-Head)
            surprisal_non_head += surprisal_values[boundary_pos_abs]
        # =========================

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

    output_file = "results_berent&pinker/results_experiment_2_babble_txt_cha_stories.csv"

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")