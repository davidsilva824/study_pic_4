### This code is working.
# Uses g2p to convert the text to phonemes
# Bow correction is working normally, as can be seen by the zero value atributed to spaces.

import json
import pandas as pd
from g2p import make_g2p
from minicons import scorer

output_file = "results_berent&pinker/100M/results_experiment_3_phoneme_llama_no_spaces.csv"
translations_file = "text_to_phonemes/compounds_experiment_3_phoneme_llama_no_spaces_ipa.txt"
model_name = "bbunzeck/phoneme-llama-no-whitespace"

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

g2p = make_g2p("eng", "eng-ipa")

manual_ipa = {
    "noblemen": "noʊbʌlmɛn",
    "firemen": "faɪrmɛn",
    "boatmen": "boʊtmɛn",
    "labourer": "leɪbɜ˞ɜ˞",
    "labourers": "leɪbɜ˞ɜ˞z",
    "classifier": "klæsʌfaɪɜ˞",
    "evaluators": "ɪvæljueɪtɜ˞z",
    "attractor": "ʌtɹæktɜ˞",
    "evaluator": "ɪvæljueɪtɜ˞",
    "avoiders": "ʌvɔɪdɜ˞z",
    "10,000": "tɛn θaʊzʌnd"
}

def word_to_ipa_no_spaces(word):
    word_l = word.lower()

    if word_l in manual_ipa:
        ipa_word = manual_ipa[word_l]
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

def process_pairs(lm, data, translation_records):
    for group_idx, (non_heads, heads) in enumerate(compound_groups, start=1):
        group_records = []

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

                group_records.append({
                    "item_idx": i + 1,
                    "compound_ortho": f"{non_head} {head}",
                    "compound_ipa": f"{ipa_non_head} {ipa_head}",
                })

        translation_records.append({
            "group_idx": group_idx,
            "items": group_records
        })

print(f"\nLoading phoneme model: {model_name}...")
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

data = []
translation_records = []

process_pairs(lm, data, translation_records)

df = pd.DataFrame(
    data,
    columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
)
df.to_csv(output_file, index=False)

with open(translations_file, "w", encoding="utf-8") as f:
    for group in translation_records:
        f.write(f"GROUP {group['group_idx']}\n")
        f.write("------------------------------------------------------------\n")

        for item in group["items"]:
            f.write(f"ITEM {item['item_idx']}\n")
            f.write(f"COMPOUND ORTHO: {item['compound_ortho']}\n")
            f.write(f"COMPOUND IPA:   {item['compound_ipa']}\n")
            f.write("\n")

        f.write("\n")

print(f"\nresults in results_berent&pinker folder.\n")
print(f"translations file saved in: {translations_file}\n")