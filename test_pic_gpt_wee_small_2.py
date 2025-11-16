import pandas as pd
from minicons import scorer

model_name = "bbunzeck/gpt-wee-small"
lm = scorer.IncrementalLMScorer(model_name, device="cpu")

# Define the BOS token strategy
BOS = True

# The list of pairs remains exactly the same
pairs = [
    # goose/geese — swan/swans  | feeder / catcher
    ("Irregular Singular", "goose", "feeder"),
    ("Regular Singular",   "swan",  "catcher"),
    ("Irregular Plural",   "geese", "feeder"),
    ("Regular Plural",     "swans", "catcher"),

    # louse/lice — flea/fleas  | eater / catcher
    ("Irregular Singular", "louse", "eater"),
    ("Regular Singular",   "flea",  "catcher"),
    ("Irregular Plural",   "lice",  "eater"),
    ("Regular Plural",     "fleas", "catcher"),

    # child/children — adult/adults  | rescuer / hugger
    ("Irregular Singular", "child",    "rescuer"),
    ("Regular Singular",   "adult",    "hugger"),
    ("Irregular Plural",   "children","rescuer"),
    ("Regular Plural",     "adults",  "hugger"),

    # mouse/mice — rat/rats  | eater / feeder
    ("Irregular Singular", "mouse", "eater"),
    ("Regular Singular",   "rat",   "feeder"),
    ("Irregular Plural",   "mice",  "eater"),
    ("Regular Plural",     "rats",  "feeder"),

    # woman/women — girl/girls  | healer / hugger
    ("Irregular Singular", "woman", "healer"),
    ("Regular Singular",   "girl",  "hugger"),
    ("Irregular Plural",   "women", "healer"),
    ("Regular Plural",     "girls", "hugger"),

    # salesman/salesmen — retailer/retailers  | advisor / supplier
    ("Irregular Singular", "salesman",  "advisor"),
    ("Regular Singular",   "retailer",  "supervisor"),
    ("Irregular Plural",   "salesmen",  "advisor"),
    ("Regular Plural",     "retailers", "supervisor"),

    # tooth/teeth — bone/bones  | warmer / protector
    ("Irregular Singular", "tooth", "fixer"),
    ("Regular Singular",   "bone",  "washer"),
    ("Irregular Plural",   "teeth", "fixer"),
    ("Regular Plural",     "bones", "washer"),

    # foot/feet — leg/legs  | fixer / washer
    ("Irregular Singular", "foot", "fixer"),
    ("Regular Singular",   "leg",  "washer"),
    ("Irregular Plural",   "feet", "fixer"),
    ("Regular Plural",     "legs", "washer"),

    # nobleman/noblemen — aristocrat/aristocrats  | helper / regulator
    ("Irregular Singular", "nobleman",    "helper"),
    ("Regular Singular",   "aristocrat",  "regulator"),
    ("Irregular Plural",   "noblemen",    "helper"),
    ("Regular Plural",     "aristocrats", "regulator"),

    # boatman/boatmen — shipmate/shipmates  | helper / supplier
    ("Irregular Singular", "boatman",   "helper"),
    ("Regular Singular",   "shipmate",  "supervisor"),
    ("Irregular Plural",   "boatmen",   "helper"),
    ("Regular Plural",     "shipmates", "supervisor"),

    # craftsman/craftsmen — labourer/labourers  | regulator / advisor
    ("Irregular Singular", "craftsman",  "regulator"),
    ("Regular Singular",   "labourer",   "advisor"),
    ("Irregular Plural",   "craftsmen",  "regulator"),
    ("Regular Plural",     "labourers",  "advisor"),

    # ox/oxen — cow/cows  | healer / rescuer
    ("Irregular Singular", "ox",  "healer"),
    ("Regular Singular",   "cow", "rescuer"),
    ("Irregular Plural",   "oxen","healer"),
    ("Regular Plural",     "cows","rescuer"),
]

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
        
        # Counting from index 1 to skip the BOS token.
        # The number of tokens in the first word (non_head).
        non_n = 0
        reconstructed_word = ""

        # The tokenizer might add a space or special character (like 'Ġ') at the start of a word.
        # We clean the tokens before comparing them.
        cleaned_tokens = [tok.lstrip('Ġ ') for tok in tokens]

        # Iterate from the first real token (index 1)
        for i in range(1, len(cleaned_tokens)):
            reconstructed_word += cleaned_tokens[i]
            non_n += 1
            if reconstructed_word == non_head:
                break
        
        # The total number of real tokens (excluding BOS)
        total_real_tokens = len(tokens) - 1
        head_n = total_real_tokens - non_n

        # Sum the surprisal values, skipping the BOS token's surprisal at index 0
        surprisal_non_head = sum(surprisal_values[1 : 1 + non_n])
        surprisal_head = sum(surprisal_values[1 + non_n : 1 + non_n + head_n])

        data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])
        print(f"{sentence}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")

# Run and save
process_pairs(pairs)

output_file = "results_gpt_wee_small_2.csv"
df = pd.DataFrame(data, columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"])
df.to_csv(output_file, index=False)

print(f'\n results in {output_file} \n')