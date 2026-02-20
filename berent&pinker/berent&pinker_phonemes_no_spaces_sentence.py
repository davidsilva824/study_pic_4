import pandas as pd
from g2p_plus import transcribe_utterances
from minicons import scorer

model_name = "phonemetransformers/GPT2-85M-CHAR-PHON-SPACELESS"

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

story_prefixes_list1 = [
    "Radioactivity is the newest weapon of choice for terrorists. A lab in Texas has developed a new type of clothing that protects against blazes of radioactivity. It is called the",
    "The cold north breezes devastated my rose garden. Last week, I read about a solution to my problem. To shield the flowers, I will install a",
    "Because of the deep religious significance of pottery, the Navajo Indians spend many years learning how to make pots. It is a great honor to be chosen to become the",
    "On the Candid Camera show, a policeman approached tourists on the street, asking them to submit their money so that he could register the bills' numbers. The policeman took the bills and disappeared, to the dismay of the poor",
    "When Mary was writing her dissertation, she was constantly trying to classify the various phases of child development. She wouldn't stop talking about the subject. Her annoyed husband called her the",
    "John works for General Electric. His job is to install hoses on washing machines. His wife jokingly calls him the",
    "My dog scares the wolves away from my farmhouse. He is the perfect",
    "I like drinking coffee at Starbucks because they sell several mixes of beans. Their coffee is the",
    "The center for disease control has issued a warning about a dangerous epidemic. In early stages, patients show a red rash. Many clinics are now overwhelmed with scared patients complaining about",
    "My landlord is very tricky. Last year, I brought my lease contracts to my lawyer, because he is so good at spotting hidden articles. I call him the",
    "While at the science museum, a group of students were trying to decode words from mazes of letters in a puzzle. Mike was the only one to decode the words. His friends called him the",
    "My friend Aaron loves to play football, but always comes home full of sores. I sent him down the street to the all-natural store, which sells a wonderful cream that works wonders on face and body sores. I call it the",
    "My favorite part about roller-coaster rides are the drops--I cannot stop screaming while we are going down. At the end of the day, the operator thought I was a",
    "Four-year-old Susie will do anything to get praises from her father. He calls her the",
    "John's work description is rather simple: He sits near the fax machine and distributes the incoming faxes to his co-workers mailboxes. John is known as the",
    "FedEx workers must be careful not to lift heavy boxes, as this can result in damage to their back. To minimize the risk of injury, the company now provides workers with special",
    "One of the most tedious jobs one could have is collecting income taxes. I always feel sorry for",
    "Taylor's dissertation compared brain functioning in men and women. Her study was among the first to demonstrate",
    "Dan has an uncanny ability to answer any question on word puzzles. His friends call him the",
    "The post office has purchased a state of the art device that can sort every piece of mail by various sizes and weights. Conveniently enough, the device is called the",
    "Mary's son won prizes in the national competitions of Math and Physics. The proud mother cannot help show everyone the photo of the happy",
    "My Aunt Mary has wonderful roses in her garden. She is among the most famous",
]

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

story_prefixes_list2 = [
    "Chris has five garden hoses. I call him the",
    "My grandmother is obsessed with putting things in order. She especially loves to sort things in rows. I call her the",
    "My friend Eileen will only live in high rises. I call her a",
    "My landlord has hidden in my lease contract many nasty clauses. I call him the",
    "Detectives can now determine if a person is lying by tracking their gazes during questioning. This new technique is called the",
    "The shelf in my office is full of boxes. I am known as the",
    "Scientists have invented a machine that can detect depression by recording the number of sighs a person emits in an hour. Conveniently enough, the machine is called the",
    "Four-year-old Danny will do anything to get praises from his mother. She calls him the",
    "Aaron owns a beer refinery that makes many brews. I call him the",
    "Amanda has already had three raises this year. I call her the",
    "My friend Greg loves theater--he has seen hundreds of plays throughout his life. I call him the",
    "My lab rats are afraid of electrical shocks and will avoid them at all costs. I call them the",
    "When my Dad and I used to go sailing, he could name each of the little bays on the shore-line. I called him the",
    "The cold winter winds have devastated my trees. Last week, I found a solution to my problem. I went to Home Depot and bought a",
    "My cat has been limping for about a year, but none of the veterinarians I visited could find out the reason. Yesterday, my husband discovered she had two sprained paws. He is now called the",
    "When Mary was working for a bank, she wouldn't stop telling her poor husband about the various types of banking fees and investments. He called her the",
    "John has been homeless for over ten years and moves in and out of shacks frequently. John is known as the",
    "My friend Emily is an antique vase dealer. She asked me to help her keep some items for her. She calls me the",
    "Farmers have long recognized that stored goods differ in quality depending on the sacks in which they are kept. Much research has attempted to explain the source of",
    "The post office has purchased a state of the art device that can sort items by various sizes and weights. Conveniently enough, the device is called the",
    "When I got my order from Office Depot, I was shocked to discover they had accidentally sent me 10,000 boxes of tacks. I had to pay an extra",
    "My friend Greg loves to sit near the fireplace--he can watch the blazes for hours. I call him the",
]

cat_labels = {0: "Pair 1", 1: "Pair 1", 2: "Pair 2", 3: "Pair 2"}

def find_sublist(haystack, needle):
    if not needle or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return (i, i + len(needle))
    return None

def ipa_word_surprisals_in_context(lm, ipa_context: str, ipa_compound: str, ipa_non_head: str, ipa_head: str):
    compound_ph = ipa_compound.split()
    nh_ph = ipa_non_head.split()
    h_ph = ipa_head.split()

    # Score full context
    tok_scores_raw = lm.token_score(
        ipa_context,
        bos_token=False,
        prob=False,
        surprisal=True,
        bow_correction=False
    )[0]

    # Keep only phoneme-like tokens that appear in the context IPA
    context_ph = ipa_context.split()
    phoneme_set = set(context_ph)
    tok_scores = [(tok, s) for (tok, s) in tok_scores_raw if tok in phoneme_set]
    toks_only = [tok for (tok, _) in tok_scores]

    # Locate the compound phoneme sequence inside the full context token stream
    span = find_sublist(toks_only, compound_ph)
    if span is None:
        raise ValueError("Could not locate compound phonemes inside full context.\n"
                         f"IPA context: {ipa_context}\nIPA compound: {ipa_compound}")

    start, end = span
    comp_scores = tok_scores[start:end]

    len_nh = len(nh_ph)
    len_h = len(h_ph)

    s_non_head = sum(s for (_, s) in comp_scores[:len_nh])
    s_head = sum(s for (_, s) in comp_scores[len_nh:len_nh + len_h])

    return [s_non_head, s_head]

print(f"\nLoading phoneme model: {model_name}...")
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

def run_experiment(compound_groups, story_prefixes, output_file):
    data = []

    for group_idx, (non_heads, heads) in enumerate(compound_groups):
        prefix = story_prefixes[group_idx].strip()

        for head in heads:
            for i, non_head in enumerate(non_heads):
                category_name = cat_labels[i]

                compound_sentence = f"{non_head} {head}"
                full_sentence = f"{prefix} {compound_sentence}"

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

                # IPA for compound alone (used to find it inside the context)
                ipa_compound = transcribe_utterances(
                    [compound_sentence],
                    backend="phonemizer",
                    language="en-us",
                    keep_word_boundaries=False,
                )[0]

                # IPA for full story context (this is what we score)
                ipa_context = transcribe_utterances(
                    [full_sentence],
                    backend="phonemizer",
                    language="en-us",
                    keep_word_boundaries=False,
                )[0]

                print("\nORTHO:", full_sentence)
                print("IPA context:  ", ipa_context)
                print("IPA compound: ", ipa_compound)

                word_surps = ipa_word_surprisals_in_context(
                    lm,
                    ipa_context,
                    ipa_compound,
                    ipa_non_head,
                    ipa_head
                )

                s_non_head = word_surps[0] if len(word_surps) > 0 else float('nan')
                s_head = word_surps[1] if len(word_surps) > 1 else float('nan')

                print(f"  Non-Head ({non_head}): {s_non_head}")
                print(f"  Head     ({head}): {s_head}")
                print(f"  Compound ({non_head} {head}): {s_non_head + s_head}")

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
run_experiment(compound_groups_list1, story_prefixes_list1, "results_berent&pinker_phonemes_no_spaces_experiment_2_sentence.csv")
run_experiment(compound_groups_list2, story_prefixes_list2, "results_berent&pinker_phonemes_no_spaces_experiment_3_sentence.csv")
