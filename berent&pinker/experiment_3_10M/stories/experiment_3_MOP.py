### This code seems to be working, but it wouldnt hurt a final reverification. 
# Uses Morphologically-aware tokenization via MorPiece: https://huggingface.co/NeTS-lab/babylm-mop-10m-gpt2 
# The tokenization here is particular. 
# Subword continuation is marked '++' instead of marking the word separation.
# This means that the information about the new word is already in the right place. Making the BOW correction unecessary. 
# Needs aspecial method to load, as you can see below.   
# Does not handle major letters, so the full text mus be converted to minor letters before being inputed to the model. 





import json
import pandas as pd
from minicons import scorer
from transformers import AutoTokenizer, AutoModelForCausalLM

models = [
    "NeTS-lab/babylm-mop-10m-gpt2"
]

BOS = True
json_file = "berent&pinker/compounds_with_stories_experiment_3.json"
output_file = "results_berent&pinker/10M/results_experiment_3_MOP_stories.csv"

with open(json_file, "r", encoding="utf-8") as f:
    stimuli_data = json.load(f)

cat_labels = {
    0: "Sibilant Singular",
    1: "Sibilant Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

def _find_compound_token_span(tokens, compound_tokens, start_idx=1):
    matches = []

    for i in range(start_idx, len(tokens) - len(compound_tokens) + 1):
        if tokens[i:i + len(compound_tokens)] == compound_tokens:
            matches.append((i, i + len(compound_tokens)))

    if len(matches) == 0:
        raise ValueError(f"Could not find compound token sequence {compound_tokens} in story tokens.")
    if len(matches) > 1:
        raise ValueError(f"Compound token sequence appears multiple times in story tokens: {compound_tokens}")

    return matches[0]

def process_pairs(lm, data):
    for group in stimuli_data:
        non_heads = group["non_heads"]
        heads = group["heads"]
        stories = group["stories"]

        if len(heads) != 1:
            raise ValueError(f"Expected exactly one head in experiment 2 item, got {len(heads)}")

        head = str(heads[0]).strip().lower()

        for i, (non_head, story_text) in enumerate(zip(non_heads, stories)):
            category_name = cat_labels[i]

            non_head = str(non_head).strip().lower()
            story_text = str(story_text).strip().lower()
            compound = f"{non_head} {head}"

            tok_scores = lm.token_score(
                story_text,
                bos_token=BOS,
                prob=False,
                surprisal=True,
                bow_correction=False
            )[0]

            tokens = [tok for tok, s, *_ in tok_scores]
            surprisal_values = [s for tok, s, *_ in tok_scores]

            print("\nTOK_SCORES:")
            for tok, s in tok_scores:
                print(f"{repr(tok):<20} {s:.7f}")

            compound_tokens = lm.tokenizer.tokenize(compound)
            compound_start_idx, compound_end_idx = _find_compound_token_span(
                tokens, compound_tokens, start_idx=1
            )

            compound_tokens_story = tokens[compound_start_idx:compound_end_idx]
            compound_surprisals = surprisal_values[compound_start_idx:compound_end_idx]

            cleaned_tokens = [str(tok).lstrip("Ġ ").replace("++", "") for tok in compound_tokens_story]

            non_n = 0
            reconstructed_word = ""

            for k in range(len(cleaned_tokens)):
                reconstructed_word += cleaned_tokens[k]
                non_n += 1
                if reconstructed_word == non_head:
                    break

            if reconstructed_word != non_head:
                raise ValueError(
                    f"Could not reconstruct non-head '{non_head}' from compound tokens {compound_tokens_story}"
                )

            total_compound_tokens = len(compound_tokens_story)
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

for model_name in models:
    print(f"\nLoading model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        return_dict=True
    )

    lm = scorer.IncrementalLMScorer(
        model,
        tokenizer=tokenizer,
        device="cpu"
    )

    data = []

    process_pairs(lm, data)

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")