import json
import pandas as pd
from g2p import make_g2p
from minicons import scorer

output_file = "results_experiment_1/100M/results_experiment_1_phoneme_llama_no_spaces.csv"
model_name = "bbunzeck/phoneme-llama"

with open("experiment_1/compounds_experiment_1.json", "r", encoding="utf-8") as f:
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

g2p = make_g2p("eng", "eng-ipa")

def word_to_ipa_no_spaces(word):
    if word.lower() == "noblemen":
        ipa_word = "noʊbʌlmɛn"
    elif word.lower() == "firemen":
        ipa_word = "faɪrmɛn"
    elif word.lower() == "boatmen":
        ipa_word = "boʊtmɛn"
    elif word.lower() == "labourer":
        ipa_word = "leɪbɜ˞ɜ˞"
    elif word.lower() == "labourers":
        ipa_word = "leɪbɜ˞ɜ˞z"
    else:
        out = g2p(word)
        ipa_word = str(out)
        ipa_word = "".join(ipa_word.split())

    return ipa_word

def split_surprisal_by_offsets(lm, sentence, tok_scores, boundary):
    enc = lm.tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    toks_only = tok_scores[1:]  # drop BOS-like token
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

                ipa_non_head = word_to_ipa_no_spaces(non_head)
                ipa_head = word_to_ipa_no_spaces(head)

                ipa_text = f"{ipa_non_head}{ipa_head}"
                boundary = len(ipa_non_head)

                tok_scores = lm.token_score(
                    ipa_text,
                    bos_token=True,
                    prob=False,
                    surprisal=True,
                    bow_correction=False
                )[0]

                print("\nIPA (folded):", ipa_text)
                print("TOK_SCORES:")
                for tok, s in tok_scores:
                    print(f"{repr(tok):<20} {s:.7f}")

                surprisal_non_head, surprisal_head = split_surprisal_by_offsets(
                    lm, ipa_text, tok_scores, boundary
                )

                print(f"  Non-Head ({non_head}): {surprisal_non_head}")
                print(f"  Head     ({head}): {surprisal_head}")

                data.append([
                    category_name,
                    non_head,
                    head,
                    surprisal_non_head,
                    surprisal_head
                ])

print(f"\nLoading phoneme model: {model_name}...")
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

data = []
process_pairs(lm, data)

df = pd.DataFrame(
    data,
    columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
)
df.to_csv(output_file, index=False)

print(f'\nresults in results_experiment_1 folder.\n')