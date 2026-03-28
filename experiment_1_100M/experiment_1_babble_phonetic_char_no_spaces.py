### This code is complete.
# BOS = False
# IMPORTANT note: the tokenizer of the model removes the spaces automatically. So it is indiferent if you put keep_word_boundaries=False or true.
# Since this model does not use word separation at all, the BOW correction should be kept False. 


import json
import pandas as pd
from g2p_plus import transcribe_utterances
from minicons import scorer

models = [
    "phonemetransformers/GPT2-85M-CHAR-PHON-SPACELESS"
]

BOS = False
output_file = "results_experiment_1_100M/results_experiment_1_babble_phonetic_char_no_spaces.csv"

with open("compounds_experiment_1.json", "r", encoding="utf-8") as f:
    compound_groups_data = json.load(f)

compound_groups = [
    (group["non_heads"], group["heads"])
    for group in compound_groups_data
]

cat_labels = {
    0: "Irregular Singular",
    1: "Irregular Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

def word_to_ipa(word):
    return transcribe_utterances(
        [word],
        backend="phonemizer",
        language="en-us",
        keep_word_boundaries=False
    )[0]

def split_surprisal_by_offsets(lm, ipa_sentence, tok_scores, boundary):
    enc = lm.tokenizer(ipa_sentence, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    toks_only = tok_scores[1:]  # drop UTT_BOUNDARY
    if len(offsets) != len(toks_only):
        raise ValueError(f"Offsets/token mismatch: offsets={len(offsets)} vs toks={len(toks_only)}")

    non_head_sum = 0.0
    head_sum = 0.0

    for (tok, s, *_), (start, end) in zip(toks_only, offsets):
        if end <= boundary:
            non_head_sum += s
        else:
            head_sum += s

    return non_head_sum, head_sum

def process_pairs(lm, data):
    for non_heads, heads in compound_groups:
        for head in heads:
            for i, non_head in enumerate(non_heads):
                category_name = cat_labels[i]

                ipa_non_head = word_to_ipa(non_head)
                ipa_head = word_to_ipa(head)

                ipa_sentence = f"{ipa_non_head} {ipa_head}"
                boundary = len(ipa_non_head)

                tok_scores = lm.token_score(
                    ipa_sentence,
                    bos_token=BOS,
                    prob=False,
                    surprisal=True,
                    bow_correction=False
                )[0]

                print("\nORTHO:", f"{non_head} {head}")
                print("IPA (folded):", ipa_sentence)

                print("TOK_SCORES:")
                for tok, s in tok_scores:
                    print(f"{repr(tok):<20} {s:.7f}")

                surprisal_non_head, surprisal_head = split_surprisal_by_offsets(
                    lm, ipa_sentence, tok_scores, boundary
                )

                data.append([
                    category_name,
                    non_head,
                    head,
                    surprisal_non_head,
                    surprisal_head
                ])

                print(f"  Non-Head ({non_head}): {surprisal_non_head}")
                print(f"  Head     ({head}): {surprisal_head}")

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

    print(f"\nResults saved in results_experiment_1_100M\n")