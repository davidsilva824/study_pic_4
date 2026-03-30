### This code is complete. 
# Some words had to be manually converted, namely firemen, boatmen, labourer and labourers. The IPA system did not work for them.
# Each word was checked individually for the best conversion. 


import json
import pandas as pd
from g2p import make_g2p
from minicons import scorer

output_file = "results_berent&pinker/100M/results_experiment_2_phoneme_llama_with_spaces.csv"
model_name = "bbunzeck/phoneme-llama"

# Obtaining the compounds from the json file.
with open("berent&pinker/compounds_experiment_2.json", "r", encoding="utf-8") as f:
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

def text_to_ipa(text):
    words = text.split()
    ipa_words = []

    for word in words:
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

for non_heads, heads in compound_groups:
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