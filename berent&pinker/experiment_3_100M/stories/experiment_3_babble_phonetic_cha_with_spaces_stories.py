### This code is complete.
### Adapted for STORIES (CHAR-PHON) with minimal changes:
### - NO forced BOW correction (bow_correction=False)
### - full-story phonemization first
### - fallback: split near middle at closest space if full-story phonemization is empty

import pandas as pd
from g2p_plus import transcribe_utterances
from minicons import scorer

models = [
    "phonemetransformers/GPT2-85M-CHAR-PHON"
]

BOS = False

# --- Read compounds + stories ---
compounds_file = "berent&pinker/stimuli_compounds_experiment_3.csv"
stories_file = "berent&pinker/stimuli_stories_experiment_3.csv"
output_file = "results_berent&pinker/results_experiment_3_babble_phonetic_cha_stories.csv"

compounds_df = pd.read_csv(compounds_file)
stories_df = pd.read_csv(stories_file)
stimuli_df = pd.merge(compounds_df, stories_df, on="story_id", how="inner")

cat_labels = {
    "a": "singular_1",
    "b": "plural_1",
    "c": "singular_2",
    "d": "plural_2",
}

def to_ipa_with_boundaries(text):
    ipa_list = transcribe_utterances(
        [text],
        backend="phonemizer",
        language="en-us",
        keep_word_boundaries=True
    )
    return ipa_list[0]

def phonemize_story_with_split_fallback(text):
    def _phonemize_recursive(s):
        s = str(s).strip()
        if s == "":
            return ""

        # try full chunk first
        ipa = to_ipa_with_boundaries(s)
        if str(ipa).strip() != "":
            return str(ipa).strip()

        # if it fails, split at closest blank space to the middle
        mid = len(s) // 2
        spaces = [i for i, ch in enumerate(s) if ch == " "]
        if not spaces:
            return ""

        split_idx = min(spaces, key=lambda i: abs(i - mid))

        left = s[:split_idx].strip()
        right = s[split_idx + 1:].strip()

        ipa_left = _phonemize_recursive(left) if left else ""
        ipa_right = _phonemize_recursive(right) if right else ""

        return " ".join([x for x in [ipa_left, ipa_right] if x]).strip()

    return _phonemize_recursive(text)

def compact(s):
    return "".join(str(s).replace("WORD_BOUNDARY", " ").split())

def find_compound_span(tokens, ipa_compound):
    start_idx = 1 if (len(tokens) > 0 and str(tokens[0]).startswith("UTT_")) else 0

    target_compact = compact(ipa_compound)

    rebuilt = ""
    spans = []  # (token_index, start_char, end_char)

    for i in range(start_idx, len(tokens)):
        tok = tokens[i]
        if tok == "WORD_BOUNDARY" or str(tok).startswith("UTT_"):
            piece = "" if str(tok).startswith("UTT_") else ""
        else:
            piece = compact(tok)

        s0 = len(rebuilt)
        rebuilt += piece
        s1 = len(rebuilt)
        spans.append((i, s0, s1))

    pos = rebuilt.find(target_compact)
    if pos == -1:
        return None, None

    end_pos = pos + len(target_compact)
    hit = []
    for tok_i, s0, s1 in spans:
        if s1 <= pos:
            continue
        if s0 >= end_pos:
            break
        if s1 > s0:
            hit.append(tok_i)

    if not hit:
        return None, None

    return hit[0], hit[-1] + 1

def split_surprisal_inside_span_by_word_boundary(tok_scores, start_idx, end_idx):
    tokens = [tok for tok, s, *_ in tok_scores]
    surprisal_values = [s for tok, s, *_ in tok_scores]

    boundary_positions = [k for k in range(start_idx, end_idx) if tokens[k] == "WORD_BOUNDARY"]

    if len(boundary_positions) < 1:
        raise ValueError("No WORD_BOUNDARY inside compound span.")

    first_boundary = boundary_positions[0]

    non_head_indices = [k for k in range(start_idx, first_boundary) if tokens[k] != "WORD_BOUNDARY"]
    head_indices = [k for k in range(first_boundary + 1, end_idx) if tokens[k] != "WORD_BOUNDARY"]

    surprisal_non_head = sum(surprisal_values[k] for k in non_head_indices) + surprisal_values[first_boundary]
    surprisal_head = sum(surprisal_values[k] for k in head_indices)

    
    
    return surprisal_non_head, surprisal_head

def process_pairs(lm, data):
    for _, row in stimuli_df.iterrows():
        suffix = str(row["story_id"]).strip().split("_")[-1]
        category_name = cat_labels[suffix]

        sentence = str(row["compound"]).strip()
        story_text = str(row["story_text"]).strip()
        non_head, head = sentence.split(" ", 1)

        ipa_story = phonemize_story_with_split_fallback(story_text)
        if ipa_story == "":
            print(f"SKIP (empty IPA story) | STORY ID: {row['story_id']} | COMPOUND: {sentence}")
            continue

        ipa_compound = to_ipa_with_boundaries(sentence)

        tok_scores = lm.token_score(
            ipa_story,
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

        compound_start_idx, compound_end_idx = find_compound_span(tokens, ipa_compound)

        if compound_start_idx is None:
            print(f"SKIP (compound not found) | STORY ID: {row['story_id']} | COMPOUND: {sentence}")
            continue

        surprisal_non_head, surprisal_head = split_surprisal_inside_span_by_word_boundary(
            tok_scores, compound_start_idx, compound_end_idx
        )

        data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])

        print(f"STORY ID: {row['story_id']}")
        print(f"COMPOUND: {sentence}")
        print(f"Compound token span: [{compound_start_idx}, {compound_end_idx})")
        print(f"{sentence}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")

# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

    data = []
    process_pairs(lm, data)

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")