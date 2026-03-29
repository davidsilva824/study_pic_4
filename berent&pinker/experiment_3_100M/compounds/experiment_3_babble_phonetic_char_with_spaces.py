### This code seems to be working, but it wouldnt hurt a final reverification. 
# It uses athe conversion from text to phonemes g2plus. 
# IMPORTANT note: The Forced BOW correction is working. This can be observe in the file 'surprisal_by_token_babble_phonetic_char_with_spaces_forced_BOW.py'.
# With the correction the suprisal of the token 'WORD_BOUNDARY' becomes zero.


import pandas as pd
from collections import defaultdict
from g2p_plus import transcribe_utterances
from minicons import scorer
import json

output_file = "results_berent&pinker/100M/results_experiment_3_babble_phonetic_char_with_spaces.csv"

model_name = "phonemetransformers/GPT2-85M-CHAR-PHON"

# Obtaining the compounds from the json file.
with open("berent&pinker/compounds_experiment_3.json", "r", encoding="utf-8") as f:
    compound_groups_data = json.load(f)

compound_groups = [
    (group["non_heads"], group["heads"])
    for group in compound_groups_data
]

cat_labels = {
    0: "Sibilant Singular",
    1: "Sibilant Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

def ipa_word_surprisals(lm, ipa_text):
    """
    ipa_text: string like 'ð ɪ s WORD_BOUNDARY m ɒ n ...'
    Returns a list of word surprisals by summing phoneme surprisals
    between WORD_BOUNDARY tokens (ignoring UTT_BOUNDARY).
    """
    tok_scores = lm.token_score(
        ipa_text,
        bos_token=False,
        prob=False,
        surprisal=True,
        bow_correction=True
    )[0]

    print("TOK_SCORES:")
    for tok, s in tok_scores:
        print(f"{tok:<20} {s:.7f}")

    word_surps = []
    current = 0.0

    for tok, s in tok_scores:
        if tok == "UTT_BOUNDARY":
            continue
        if tok == "WORD_BOUNDARY":
            word_surps.append(current)
            current = 0.0
        else:
            current += s

    if current != 0.0 or not word_surps:
        word_surps.append(current)

    return word_surps


# --- Main processing for the phoneme model ---
print(f"\nLoading phoneme model: {model_name}...")
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

# --- Forced BOW setup for WORD_BOUNDARY ---
bow_symbol = "WORD_BOUNDARY"
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
print("-" * 30)

data = []

for non_heads, heads in compound_groups:
    for head in heads:
        for i, non_head in enumerate(non_heads):
            category_name = cat_labels[i]

            sentence = f"{non_head} {head}"
            lines = [sentence]

            ipa_list = transcribe_utterances(
                lines,
                backend="phonemizer",
                language="en-us",
                keep_word_boundaries=True,
                allow_possibly_faulty_word_boundaries=True
            )

            ipa_text = ipa_list[0]
            print("\nORTHO:", sentence)
            print("IPA (folded):", ipa_text)

            word_surps = ipa_word_surprisals(lm, ipa_text)

            if len(word_surps) != 2:
                print("WARNING: expected 2 words, got", len(word_surps), "for", sentence)

            s_non_head = word_surps[0]
            s_head = word_surps[1] if len(word_surps) > 1 else float('nan')

            print(f"  Non-Head ({non_head}): {s_non_head}")
            print(f"  Head     ({head}): {s_head}")

            data.append([
                category_name,
                non_head,
                head,
                s_non_head,
                s_head
            ])



df = pd.DataFrame(
    data,
    columns=[
        "Category",
        "Non-Head",
        "Head",
        "Surprisal Non-head",
        "Surprisal head"
    ]
)
df.to_csv(output_file, index=False)

print(f'\nresults in results_berent&pinker folder.\n')