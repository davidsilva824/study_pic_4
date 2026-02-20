import pandas as pd
from g2p_plus import transcribe_utterances
from minicons import scorer

model_name = "phonemetransformers/GPT2-85M-CHAR-PHON"

# ---------- LIST 1 ----------
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

# ---------- LIST 2 ----------
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


print(f"\nLoading phoneme model: {model_name}...")
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

def run_experiment(compound_groups, output_file):
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
    print(f"\nResults saved in {output_file}\n")


# Run for both lists
run_experiment(compound_groups_list1, "results_berent&pinker_phonemes_with_spaces_experiment_2.csv")
run_experiment(compound_groups_list2, "results_berent&pinker_phonemes_with_spaces_experiment_3.csv")
