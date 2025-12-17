import torch
from collections import defaultdict
import pandas as pd
from minicons import scorer

# ---------- COMPOUND LISTS ----------

# LIST 1
compound_groups_list1 = [
    (['blaze', 'blazes', 'spark', 'sparks'], ['protector']),
    (['breeze', 'breezes', 'storm', 'storms'], ['protector']),
    (['vase', 'vases', 'pot', 'pots'], ['maker']),
    (['hoax', 'hoaxes', 'joke', 'jokes'], ['victims']),
    (['phase', 'phases', 'step', 'steps'], ['classifier']),
    (['hose', 'hoses', 'pipe', 'pipes'], ['installer']),
    (['fox', 'foxes', 'wolf', 'wolves'], ['chaser']),
    (['mix', 'mixes', 'blend', 'blends'], ['winner']),
    (['nose', 'noses', 'thigh', 'thighs'], ['rashes']),
    (['clause', 'clauses', 'article', 'articles'], ['finder']),
    (['maze', 'mazes', 'web', 'webs'], ['decoder']),
    (['bruise', 'bruises', 'sore', 'sores'], ['healer']),
    (['rise', 'rises', 'drop', 'drops'], ['addict']),
    (['praise', 'praises', 'compliment', 'compliments'], ['getter']),
    (['fax', 'faxes', 'copy', 'copies'], ['man']),
    (['box', 'boxes', 'pack', 'packs'], ['lifters']),
    (['tax', 'taxes', 'toll', 'tolls'], ['receivers']),
    (['sex', 'sexes', 'gender', 'genders'], ['disparities']),
    (['quiz', 'quizzes', 'puzzle', 'puzzles'], ['expert']),
    (['size', 'sizes', 'shape', 'shapes'], ['device']),
    (['prize', 'prizes', 'award', 'awards'], ['seeker']),
    (['rose', 'roses', 'flower', 'flowers'], ['gardeners']),
]

# LIST 2
compound_groups_list2 = [
    (['hose', 'hoses', 'hoe', 'hoes'], ['collector']),
    (['rose', 'roses', 'row', 'rows'], ['organizer']),
    (['rise', 'rises', 'lie', 'lies'], ['addict']),
    (['clause', 'clauses', 'claw', 'claws'], ['hider']),
    (['gaze', 'gazes', 'guy', 'guys'], ['attractor']),
    (['box', 'boxes', 'book', 'books'], ['horde']),
    (['size', 'sizes', 'sigh', 'sighs'], ['machine']),
    (['praise', 'praises', 'tray', 'trays'], ['getter']),
    (['bruise', 'bruises', 'brew', 'brews'], ['expert']),
    (['raise', 'raises', 'ray', 'rays'], ['seeker']),
    (['blaze', 'blazes', 'play', 'plays'], ['lover']),
    (['fox', 'foxes', 'shock', 'shocks'], ['avoiders']),
    (['maze', 'mazes', 'bay', 'bays'], ['expert']),
    (['breeze', 'breezes', 'tree', 'trees'], ['protector']),
    (['cause', 'causes', 'paw', 'paws'], ['evaluator']),
    (['phase', 'phases', 'fee', 'fees'], ['fanatic']),
    (['fax', 'faxes', 'shack', 'shacks'], ['man']),
    (['vase', 'vases', 'bee', 'bees'], ['owner']),
    (['sex', 'sexes', 'sack', 'sacks'], ['distinctions']),
    (['size', 'sizes', 'sigh', 'sighs'], ['device']),
    (['tax', 'taxes', 'tack', 'tacks'], ['fee']),
    (['blaze', 'blazes', 'play', 'plays'], ['admirer']),
]

# First two = Pair 1, last two = Pair 2
cat_labels = {
    0: "Pair 1",
    1: "Pair 1",
    2: "Pair 2",
    3: "Pair 2",
}

# ---------- MODEL + BOW SETUP (UNCHANGED LOGIC) ----------

model_name = "phonemetransformers/GPT2-85M-BPE-TXT"
BOS = False

print(f"\nLoading model: {model_name}...")
lm = scorer.IncrementalLMScorer(model_name, device="cuda")  # or "cpu"

bow_symbol = "Ġ"
lm.is_bow_tokenizer = True
lm.bow_symbol = bow_symbol

from collections import defaultdict
bow_subwords = defaultdict(bool)

# Mark standard vocabulary
for word, idx in lm.tokenizer.get_vocab().items():
    if len(word) > 0 and word[0] == bow_symbol:
        bow_subwords[idx] = True
    else:
        bow_subwords[idx] = False

# Mark special/added tokens (like BOS/EOS) as False to be safe
for idx in lm.tokenizer.get_added_vocab().values():
    bow_subwords[idx] = False

lm.bow_subwords = bow_subwords
lm.bow_subword_idx = [k for k, v in lm.bow_subwords.items() if v]

print("Forced BOW settings applied successfully.")
print("-" * 30)

# ---------- WORD-LEVEL SURPRISAL (TEXT) ----------

def get_word_surprisals(lm, sentence: str, non_head: str, head: str):
    tok_scores = lm.token_score(
        sentence,
        bos_token=BOS,
        prob=False,
        surprisal=True,
        bow_correction=True
    )[0]

    tokens = [tok for tok, s, *_ in tok_scores]
    surps = [s for tok, s, *_ in tok_scores]

    # ---- skip BOS / first token, like in your GPT-J code ----
    tokens_eff = tokens[1:]
    surps_eff = surps[1:]

    cleaned_tokens = [tok.lstrip("Ġ ") for tok in tokens_eff]

    # Find how many subword tokens form the non-head
    reconstructed = ""
    non_n = 0
    for tok in cleaned_tokens:
        reconstructed += tok
        non_n += 1
        if reconstructed == non_head:
            break

    total_real_tokens = len(tokens_eff)
    head_n = total_real_tokens - non_n

    # Sum surprisals for non-head and head (over the effective tokens)
    surprisal_non_head = sum(surps_eff[:non_n])
    surprisal_head = sum(surps_eff[non_n:non_n + head_n])

    return surprisal_non_head, surprisal_head


def run_experiment(compound_groups, output_file):
    data = []

    for non_heads, heads in compound_groups:
        for head in heads:
            for i, non_head in enumerate(non_heads):
                category_name = cat_labels[i]
                sentence = f"{non_head} {head}"

                s_non_head, s_head = get_word_surprisals(lm, sentence, non_head, head)

                print(f"{sentence}: Non-Head: {s_non_head}, Head: {s_head}")

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
    print(f"\nResults saved in {output_file}\n")


# Run for both lists
run_experiment(compound_groups_list1, "results_berent&pinker_text_with_spaces_experiment_2.csv")
run_experiment(compound_groups_list2, "results_berent&pinker_text_with_spaces_experiment_3.csv")
