### This code is complete.
# BOS = True
# IMPORTANT note: the tokenizer of the model does not remove the spaces automatically.
# Since this model does not use word separation at all, the BOW correction should be kept False. 
# With BOW correction true it also massively increases the surprisal of the last character. So avoid it. 


import json
import pandas as pd
from minicons import scorer

models = [
    "bbunzeck/grapheme-llama-no-whitespace"
]

BOS = True
json_file = "berent&pinker/compounds_with_stories_experiment_3.json"
output_file = "results_berent&pinker/100M/results_experiment_3_grapheme_llama_no_spaces_stories.csv"

with open(json_file, "r", encoding="utf-8") as f:
    stimuli_data = json.load(f)

cat_labels = {
    0: "Sibilant Singular",
    1: "Sibilant Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

def _remove_spaces(text):
    return "".join(str(text).split())

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

    start_idx = 1 if (len(tok_scores) > 0 and str(tok_scores[0][0]).startswith("<")) else 0
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

def _split_compound_surprisal_by_offsets(lm, compound_text, compound_tok_scores, boundary):
    enc = lm.tokenizer(
        compound_text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    offsets = enc["offset_mapping"]

    start_idx = 1 if (len(compound_tok_scores) > 0 and str(compound_tok_scores[0][0]).startswith("<")) else 0
    toks_only = compound_tok_scores[start_idx:]

    if len(offsets) != len(toks_only):
        raise ValueError(f"Offsets/token mismatch inside compound: offsets={len(offsets)} vs toks={len(toks_only)}")

    non_head_sum = 0.0
    head_sum = 0.0

    for (tok, s, *_), (start, end) in zip(toks_only, offsets):
        if end <= boundary:
            non_head_sum += s
        else:
            head_sum += s

    return non_head_sum, head_sum

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

            compound_no_spaces = f"{non_head}{head}"
            story_no_spaces = _remove_spaces(story_text)

            tok_scores = lm.token_score(
                story_no_spaces,
                bos_token=BOS,
                prob=False,
                surprisal=True,
                bow_correction=False
            )[0]

            tokens = [tok for tok, s, *_ in tok_scores]

            print("\nTOK_SCORES:")
            for tok, s in tok_scores:
                print(f"{repr(tok):<20} {s:.7f}")

            compound_start_idx, compound_end_idx = _find_compound_token_span(
                lm, story_no_spaces, tok_scores, compound_no_spaces
            )

            compound_tok_scores = tok_scores[compound_start_idx:compound_end_idx]
            boundary = len(non_head)

            surprisal_non_head, surprisal_head = _split_compound_surprisal_by_offsets(
                lm, compound_no_spaces, compound_tok_scores, boundary
            )

            data.append([
                category_name,
                non_head,
                head,
                surprisal_non_head,
                surprisal_head
            ])

            print(f"COMPOUND: {non_head} {head}")
            print(f"Compound token span: [{compound_start_idx}, {compound_end_idx})")
            print(f"  Non-Head ({non_head}): {surprisal_non_head}")
            print(f"  Head     ({head}): {surprisal_head}")

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