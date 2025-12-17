import pandas as pd
from g2p_plus import transcribe_utterances
from minicons import scorer

model_name = "phonemetransformers/GPT2-85M-CHAR-PHON-SPACELESS"

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

def ipa_word_surprisals(lm, ipa_compound: str, ipa_non_head: str, ipa_head: str):
    """
    - ipa_compound: phonemes for 'non_head head' (no boundaries).
    - ipa_non_head: phonemes for non_head alone.
    - ipa_head: phonemes for head alone.
    Sum token surprisals over the segment matching each word.
    """
    compound_ph = ipa_compound.split()
    nh_ph = ipa_non_head.split()
    h_ph = ipa_head.split()

    # Raw token scores from the model
    tok_scores_raw = lm.token_score(
        ipa_compound,
        bos_token=False,
        prob=False,
        surprisal=True,
        bow_correction=False
    )[0]

    # Keep only tokens that correspond to actual phoneme symbols
    phoneme_set = set(compound_ph)
    tok_scores = [(tok, s) for (tok, s) in tok_scores_raw if tok in phoneme_set]

    if len(tok_scores) != len(compound_ph):
        print("WARNING after filtering: still mismatch:",
              len(tok_scores), "tokens vs", len(compound_ph), "phonemes")

    # Assume order matches: first len(nh_ph) tokens = non-head, next len(h_ph) = head
    len_nh = len(nh_ph)
    len_h = len(h_ph)

    s_non_head = sum(s for (_, s) in tok_scores[:len_nh])
    s_head = sum(s for (_, s) in tok_scores[len_nh:len_nh + len_h])

    return [s_non_head, s_head]

print(f"\nLoading phoneme model: {model_name}...")
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

def run_experiment(compound_groups, output_file):
    data = []

    for non_heads, heads in compound_groups:
        for head in heads:
            for i, non_head in enumerate(non_heads):
                category_name = cat_labels[i]

                sentence = f"{non_head} {head}"

                # IPA for non-head alone
                ipa_non_head = transcribe_utterances(
                    [non_head],
                    backend="phonemizer",
                    language="en-us",
                    keep_word_boundaries=False,
                )[0]

                # IPA for head alone
                ipa_head = transcribe_utterances(
                    [head],
                    backend="phonemizer",
                    language="en-us",
                    keep_word_boundaries=False,
                )[0]

                # IPA for full compound (what we feed to the model)
                ipa_compound = transcribe_utterances(
                    [sentence],
                    backend="phonemizer",
                    language="en-us",
                    keep_word_boundaries=False,
                )[0]

                print("\nORTHO:", sentence)
                print("IPA non-head: ", ipa_non_head)
                print("IPA head:     ", ipa_head)
                print("IPA compound: ", ipa_compound)

                word_surps = ipa_word_surprisals(
                    lm,
                    ipa_compound,
                    ipa_non_head,
                    ipa_head
                )

                if len(word_surps) != 2:
                    print("WARNING: expected 2 words, got", len(word_surps), "for", sentence)

                s_non_head = word_surps[0] if len(word_surps) > 0 else float('nan')
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
run_experiment(compound_groups_list1, "results_berent&pinker_phonemes_no_spaces_experiment_2.csv")
run_experiment(compound_groups_list2, "results_berent&pinker_phonemes_no_spaces_experiment_3.csv")
