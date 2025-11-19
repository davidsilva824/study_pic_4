import pandas as pd
from minicons import scorer

model_name = "bbunzeck/gpt-wee-large"
lm = scorer.IncrementalLMScorer(model_name, device="cpu")

BOS = True

# New compact format: ( [irr_sg, irr_pl, reg_sg, reg_pl], [head1, head2, head3, head4] )
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
     ['patrol', 'register', 'committee', 'brigade']),
    (['woman', 'women', 'girl', 'girls'],
     ['protector', 'register', 'hangout', 'brigade']),
    (['man', 'men', 'boy', 'boys'],
     ['committee', 'register', 'finder', 'hangout']),
    (['salesman', 'salesmen', 'retailer', 'retailers'],
     ['committee', 'inspector', 'protector', 'employer']),
    (['nobleman', 'noblemen', 'aristocrat', 'aristocrats'],
     ['patrol', 'hangout', 'committee', 'brigade']),
    (['boatman', 'boatmen', 'shipmate', 'shipmates'],
     ['brigade', 'finder', 'inspector', 'employer']),
    (['craftsman', 'craftsmen', 'labourer', 'labourers'],
     ['employer', 'examination', 'hangout', 'finder']),
    (['fireman', 'firemen', 'lifeguard', 'lifeguards'],
     ['examination', 'employer', 'brigade', 'patrol']),
]

# Map noun position → category label
cat_labels = {
    0: "Irregular Singular",
    1: "Irregular Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

# Build the full list of (Category, Non-head, Head) pairs as before
pairs = []
for non_heads, heads in compound_groups:
    for i, non_head in enumerate(non_heads):
        category_name = cat_labels[i]
        for head in heads:
            pairs.append((category_name, non_head, head))

data = []

def process_pairs(pairs):
    for category_name, non_head, head in pairs:
        sentence = f"{non_head} {head}"
        
        tok_scores = lm.token_score(
            sentence,
            bos_token=BOS,
            prob=False,
            surprisal=True,
            bow_correction=True
        )[0]
        
        tokens = [tok for tok, s, *_ in tok_scores]
        surprisal_values = [s for tok, s, *_ in tok_scores]
        
        print(' '.join(f'{tok:>10}' for tok in tokens))
        print(' '.join(f'{s:>10.3f}' for s in surprisal_values))
        print(surprisal_values)
        
        non_n = 0
        reconstructed_word = ""

        cleaned_tokens = [tok.lstrip('Ġ ') for tok in tokens]

        for i in range(1, len(cleaned_tokens)):
            reconstructed_word += cleaned_tokens[i]
            non_n += 1
            if reconstructed_word == non_head:
                break
        
        total_real_tokens = len(tokens) - 1
        head_n = total_real_tokens - non_n

        surprisal_non_head = sum(surprisal_values[1 : 1 + non_n])
        surprisal_head = sum(surprisal_values[1 + non_n : 1 + non_n + head_n])

        data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])
        print(f"{sentence}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")

process_pairs(pairs)

output_file = "study_gpt_wee_large_experiment_3.csv"
df = pd.DataFrame(data, columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"])
df.to_csv(output_file, index=False)

print(f'\n results in {output_file} \n')
