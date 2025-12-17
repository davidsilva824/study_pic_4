import pandas as pd
from g2p_plus import transcribe_utterances
from minicons import scorer

model_name = "phonemetransformers/GPT2-85M-CHAR-PHON"

compound_groups = [
    (['goose', 'geese', 'swan', 'swans'],
     ['protector', 'trader', 'tracker', 'expert']),

    (['ox', 'oxen', 'cow', 'cows'],
     ['register', 'trader', 'tracker', 'finder']),

    (['louse', 'lice', 'flea', 'fleas'],
     ['issue', 'trader', 'tracker', 'expert']),

    (['mouse', 'mice', 'rat', 'rats'],
     ['issue', 'trader', 'tracker', 'inspector']),

    (['foot', 'feet', 'leg', 'legs'],
     ['issue', 'examination', 'expert', 'inspector']),

    (['tooth', 'teeth', 'bone', 'bones'],
     ['issue', 'examination', 'expert', 'protector']),

    (['child', 'children', 'adult', 'adults'],
     ['patrol', 'register', 'institute', 'crew']),

    (['woman', 'women', 'girl', 'girls'],
     ['protector', 'register', 'hangout', 'crew']),

    (['man', 'men', 'boy', 'boys'],
     ['institute', 'register', 'finder', 'hangout']),

    (['salesman', 'salesmen', 'retailer', 'retailers'],
     ['institute', 'inspector', 'protector', 'employer']),

    (['nobleman', 'noblemen', 'aristocrat', 'aristocrats'],
     ['patrol', 'hangout', 'institute', 'crew']),

    (['boatman', 'boatmen', 'shipmate', 'shipmates'],
     ['patrol', 'finder', 'inspector', 'employer']),

    (['craftsman', 'craftsmen', 'labourer', 'labourers'],
     ['employer', 'examination', 'hangout', 'finder']),
    
    (['fireman', 'firemen', 'lifeguard', 'lifeguards'],
     ['examination', 'employer', 'crew', 'patrol'])
]

cat_labels = {
    0: "Irregular Singular",
    1: "Irregular Plural",
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
        bow_correction=False
    )[0]

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

    # In case final word is not followed by WORD_BOUNDARY
    if current != 0.0 or not word_surps:
        word_surps.append(current)

    return word_surps


# --- Main processing for the phoneme model ---
print(f"\nLoading phoneme model: {model_name}...")
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

data = []

for non_heads, heads in compound_groups:
    # same head-first order as in your GPT/MLM code
    for head in heads:
        for i, non_head in enumerate(non_heads):
            category_name = cat_labels[i]

            sentence = f"{non_head} {head}"
            lines = [sentence]

            # 1) Convert to IPA with the SAME config
            ipa_list = transcribe_utterances(
                lines,
                backend="phonemizer",
                language="en-us",
                keep_word_boundaries=True,
                allow_possibly_faulty_word_boundaries=True
                # uncorrected=False by default → folding ON
            )

            ipa_text = ipa_list[0]
            print("\nORTHO:", sentence)
            print("IPA (folded):", ipa_text)

            # 2) Get word-level surprisal by summing phoneme surprisals
            word_surps = ipa_word_surprisals(lm, ipa_text)

            # We expect exactly 2 words: non-head and head
            if len(word_surps) != 2:
                print("WARNING: expected 2 words, got", len(word_surps), "for", sentence)

            s_non_head = word_surps[0]
            s_head = word_surps[1] if len(word_surps) > 1 else float('nan')

            # Debug print similar in spirit to other experiments
            print(f"  Non-Head ({non_head}): {s_non_head}")
            print(f"  Head     ({head}): {s_head}")

            data.append([
                category_name,
                non_head,
                head,
                s_non_head,
                s_head
            ])

# Save CSV in same format
base_name = model_name.split("/")[-1]
clean_name = base_name.replace("-", "_")
output_file = f"results_experiment_3_babble.csv"

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

print(f"\nResults saved in {output_file}\n")
