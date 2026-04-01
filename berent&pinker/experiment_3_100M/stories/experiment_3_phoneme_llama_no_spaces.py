### This code is complete.
# Some words had to be manually converted, namely firemen, boatmen, labourer, labourers, classifier and evaluators.
# Each word was checked individually for the best conversion.

import json
import pandas as pd
from g2p import make_g2p
from minicons import scorer

output_file = "results_berent&pinker/100M/results_experiment_3_phoneme_llama_with_spaces.csv"
translations_file = "text_to_phonemes/compounds_experiment_3_ipa.txt"
model_name = "bbunzeck/phoneme-llama"

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

def text_to_ipa(text):
    words = text.split()
    ipa_words = []

    for word in words:
        word_l = word.lower()

        if word_l in manual_ipa:
            ipa_word = manual_ipa[word_l]
        else:
            out = g2p(word)
            ipa_word = str(out)
            ipa_word = " ".join(ipa_word.split())

        ipa_words.append(ipa_word)

    return " ".join(ipa_words)

def ipa_word_surprisals(lm, ipa_text):
    """
    ipa_text: string like 'ð ɪ s m ɒ n s t ə ɹ ...'
    Returns a list with [non_head_surprisal, head_surprisal]
    by reconstructing the first IPA word from token strings.
    """

    tok_scores = lm.token_score(
        ipa_text,
        bos_token=True,
        prob=False,
        surprisal=True,
        bow_correction=True
    )[0]

    print("TOK_SCORES:")
    for tok, s in tok_scores:
        print(f"{repr(tok):<20} {s:.7f}")

    ipa_words = ipa_text.split(" ")
    if len(ipa_words) != 2:
        raise ValueError(f"Expected exactly 2 IPA words, got {len(ipa_words)} in: {ipa_text}")

    first_ipa_word = ipa_words[0]

    tokens = [tok for tok, s, *_ in tok_scores]
    surprisal_values = [s for tok, s, *_ in tok_scores]

    start_idx = 1 if len(tokens) > 0 and tokens[0].startswith("<") else 0

    reconstructed = ""
    non_head_indices = []
    head_indices = []

    k = start_idx

    # Reconstruct first IPA word from token strings, ignoring pure spaces
    while k < len(tokens) and reconstructed != first_ipa_word:
        tok = tokens[k]

        if tok.strip() != "":
            reconstructed += tok
            non_head_indices.append(k)

        k += 1

    if reconstructed != first_ipa_word:
        raise ValueError(
            f"Could not reconstruct first IPA word.\n"
            f"Target: {first_ipa_word}\n"
            f"Got:    {reconstructed}\n"
            f"Tokens: {tokens}"
        )

    # Remaining non-space tokens belong to head
    for j in range(k, len(tokens)):
        if tokens[j].strip() != "":
            head_indices.append(j)

    surprisal_non_head = sum(surprisal_values[j] for j in non_head_indices)
    surprisal_head = sum(surprisal_values[j] for j in head_indices)

    return [surprisal_non_head, surprisal_head]

# --- Main processing for the phoneme model ---
print(f"\nLoading phoneme model: {model_name}...")
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

data = []
translation_records = []

for group_idx, (non_heads, heads) in enumerate(compound_groups, start=1):
    group_records = []

    for head in heads:
        for i, non_head in enumerate(non_heads):
            category_name = cat_labels[i]

            sentence = f"{non_head} {head}"
            ipa_text = text_to_ipa(sentence)

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

            group_records.append({
                "item_idx": i + 1,
                "compound_ortho": sentence,
                "compound_ipa": ipa_text
            })

    translation_records.append({
        "group_idx": group_idx,
        "items": group_records
    })

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