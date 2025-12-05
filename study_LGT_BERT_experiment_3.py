import pandas as pd
from minicons import scorer
from nltk.tokenize import TweetTokenizer

models = [
    "babylm/ltgbert-10m-2024",
    "babylm/ltgbert-100m-2024",
]

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

# Mapping
cat_labels = {
    0: "Irregular Singular",
    1: "Irregular Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

tweet_tok = TweetTokenizer()

def get_masked_word_surprisal(model, text):
    """
    EXACTLY your logic:
    - PLL_metric='within_word_l2r'
    - TweetTokenizer defines target words
    - Sum subtoken surprisals until reconstructed word matches
    """
    token_scores = model.token_score(
        text,
        surprisal=True,
        base_two=True,
        PLL_metric="within_word_l2r",
    )[0]

    target_words = tweet_tok.tokenize(text)

    final_results = []
    token_idx = 0

    for word in target_words:
        current_surprisal = 0.0
        reconstructed = ""

        while token_idx < len(token_scores):
            tok_text, tok_score = token_scores[token_idx]
            clean_tok = tok_text.replace("##", "").strip()
            reconstructed += clean_tok
            current_surprisal += tok_score
            token_idx += 1

            if reconstructed == word:
                break

        final_results.append((word, current_surprisal))

    return final_results


def process_pairs_mlm(lm, data):
    # Same head-first order as in your GPT code
    for non_heads, heads in compound_groups:
        for head in heads:
            for i, non_head in enumerate(non_heads):
                category_name = cat_labels[i]
                sentence = f"{non_head} {head}"

                # Word-level surprisals via within_word_l2r
                word_scores = get_masked_word_surprisal(lm, sentence)
                (w1, s_non_head), (w2, s_head) = word_scores  # sentence is always 2 words

                # Debug print
                print(f"{sentence}")
                print(f"  Non-Head ({w1}): {s_non_head}")
                print(f"  Head     ({w2}): {s_head}")

                data.append([
                    category_name,
                    non_head,
                    head,
                    s_non_head,
                    s_head
                ])


# MAIN EXECUTION
for model_name in models:
    print(f"\nLoading MLM model: {model_name}...")
    lm = scorer.MaskedLMScorer(model_name, device="cuda", trust_remote_code=True)

    data = []

    process_pairs_mlm(lm, data)

    # Fixed filenames
    if model_name == "babylm/ltgbert-10m-2024":
        output_file = "results_LGT_BERT_10M_experiment_3.csv"
    elif model_name == "babylm/ltgbert-100m-2024":
        output_file = "results_LGT_BERT_100M_experiment_3.csv"


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
