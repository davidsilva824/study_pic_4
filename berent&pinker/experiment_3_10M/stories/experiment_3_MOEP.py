### This code is complete. 
# BOS = False
# Normal BOW
# During loading it mus have trust_remote_code=True

models = [
    "Jtapsa/moep_swiglu"
]


import json
import pandas as pd
from minicons import scorer

models = [
    "Jtapsa/moep_swiglu"
]

BOS = False

json_file = "berent&pinker/compounds_with_stories_experiment_3.json"
output_file = "results_berent&pinker/10M/results_experiment_3_MOEP_stories.csv"

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

    toks_only = tok_scores[1:]  # drop BOS-like token
    if len(offsets) != len(toks_only):
        raise ValueError(f"Offsets/token mismatch: offsets={len(offsets)} vs toks={len(toks_only)}")

    span_indices = []
    for i, ((tok, s, *_), (tok_start, tok_end)) in enumerate(zip(toks_only, offsets), start=1):
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

            non_n = 0
            reconstructed_word = ""

            for k in range(len(cleaned_tokens)):
                reconstructed_word += cleaned_tokens[k]
                non_n += 1
                if reconstructed_word == non_head:
                    break

            if reconstructed_word != non_head:
                raise ValueError(
                    f"Could not reconstruct non-head '{non_head}' from compound tokens {compound_tokens}"
                )

            total_compound_tokens = len(compound_tokens)
            head_n = total_compound_tokens - non_n

            surprisal_non_head = sum(compound_surprisals[:non_n])
            surprisal_head = sum(compound_surprisals[non_n:non_n + head_n])

            data.append([
                category_name,
                non_head,
                head,
                surprisal_non_head,
                surprisal_head
            ])

            print(f"COMPOUND: {compound}")
            print(f"Compound token span: [{compound_start_idx}, {compound_end_idx})")
            print(f"  Non-Head ({non_head}): {surprisal_non_head}")
            print(f"  Head     ({head}): {surprisal_head}")

# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda", trust_remote_code=True)

    data = []
    process_pairs(lm, data)

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")