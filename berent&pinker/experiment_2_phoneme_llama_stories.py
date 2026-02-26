### This code is complete.
### Adapted for bbunzeck/phoneme-llama using g2p text->IPA conversion
### + FULL STORY context (stories file)
### + Finds target compound span inside story tokens
### + FIXED split of non-head/head inside matched span (skips separator tokens correctly)
### + Keeps print output style like your other scripts
### + ADDED ONLY: recursive split fallback for story phonemization (split on spaces)

import pandas as pd
from minicons import scorer
from g2p import make_g2p

models = [
    "bbunzeck/phoneme-llama"
]

BOS = True

# --- Read compounds + stories and merge by story_id ---
compounds_file = "berent&pinker/stimuli_compounds_experiment_2.csv"
stories_file = "berent&pinker/stimuli_stories_experiment_2.csv"
output_file = "results_berent&pinker/results_experiment_2_phoneme_llama_stories.csv"

compounds_df = pd.read_csv(compounds_file)
stories_df = pd.read_csv(stories_file)
stimuli_df = pd.merge(compounds_df, stories_df, on="story_id", how="inner")

cat_labels = {
    "a": "singular_1",
    "b": "plural_1",
    "c": "singular_2",
    "d": "plural_2",
}

# --- g2p converters (rule-based + neural fallback) ---
g2p_rule = make_g2p("eng", "eng-ipa")
g2p_neural = make_g2p("eng", "eng-ipa", neural=True)


def _g2p_output_string(transducer, text):
    out = transducer(text)
    ipa = getattr(out, "output_string", str(out))
    return " ".join(str(ipa).split()).strip()


def phonemize_word(word):
    ipa_rule = _g2p_output_string(g2p_rule, word)
    if ipa_rule == "":
        ipa_neural = _g2p_output_string(g2p_neural, word)
        return ipa_neural
    return ipa_rule


def phonemize_sentence_with_fallback(sentence):
    words = sentence.split()
    ipa_words = [phonemize_word(w) for w in words]
    return " ".join([w for w in ipa_words if w != ""])


# --- ADDED: recursive split fallback for long/problematic stories ---
def phonemize_story_with_split_fallback(text):
    def _phonemize_recursive(s):
        s = str(s).strip()
        if s == "":
            return ""

        # try full chunk first
        ipa = phonemize_sentence_with_fallback(s)
        if str(ipa).strip() != "":
            return str(ipa).strip()

        # if it fails, split at nearest space to the middle
        mid = len(s) // 2
        spaces = [i for i, ch in enumerate(s) if ch == " "]
        if not spaces:
            return ""

        split_idx = min(spaces, key=lambda i: abs(i - mid))

        left = s[:split_idx].strip()
        right = s[split_idx + 1:].strip()

        ipa_left = _phonemize_recursive(left) if left else ""
        ipa_right = _phonemize_recursive(right) if right else ""

        return " ".join([x for x in [ipa_left, ipa_right] if x]).strip()

    return _phonemize_recursive(text)


def _is_special_token(tok):
    t = str(tok)
    return t.startswith("<") and t.endswith(">")


def _compact(s):
    return "".join(str(s).split())


def _find_compound_token_span(tokens, start_idx, ipa_compound):
    target = _compact(ipa_compound)

    rebuilt = ""
    spans = []  # (token_idx, start_char, end_char)

    for i in range(start_idx, len(tokens)):
        tok = str(tokens[i])

        if _is_special_token(tok):
            spans.append((i, len(rebuilt), len(rebuilt)))
            continue

        piece = _compact(tok)
        s0 = len(rebuilt)
        rebuilt += piece
        s1 = len(rebuilt)
        spans.append((i, s0, s1))

    pos = rebuilt.find(target)
    if pos == -1:
        return None

    end_pos = pos + len(target)
    hit = []

    for tok_i, s0, s1 in spans:
        if s1 <= pos:
            continue
        if s0 >= end_pos:
            break
        if s1 > s0:
            hit.append(tok_i)

    if not hit:
        return None

    return hit[0], hit[-1] + 1


def process_pairs(lm, pairs, data):
    for _, row in stimuli_df.iterrows():
        suffix = str(row["story_id"]).strip().split("_")[-1]
        category_name = cat_labels[suffix]

        sentence = str(row["compound"]).strip()   # target compound
        story_text = str(row["story_text"]).strip()
        non_head, head = sentence.split(" ", 1)

        # --- Full story IPA (context) with recursive split fallback ---
        ipa_story = phonemize_story_with_split_fallback(story_text)
        if ipa_story == "":
            print(f"SKIP (empty IPA story) | STORY ID: {row['story_id']} | COMPOUND: {sentence}")
            continue

        # --- Target compound IPA (for locating span) ---
        ipa_compound = phonemize_sentence_with_fallback(sentence)

        tok_scores = lm.token_score(
            ipa_story,
            bos_token=BOS,
            prob=False,
            surprisal=True,
            bow_correction=False
        )[0]

        tokens = [tok for tok, s, *_ in tok_scores]
        surprisal_values = [s for tok, s, *_ in tok_scores]

        # --- Original Print Block ---
        print(' '.join(f'{str(tok):>10}' for tok in tokens))
        print(' '.join(f'{s:>10.3f}' for s in surprisal_values))
        print(surprisal_values)

        start_idx = 1 if (len(tokens) > 0 and _is_special_token(tokens[0])) else 0

        span = _find_compound_token_span(tokens, start_idx, ipa_compound)
        if span is None:
            print(f"SKIP (compound not found) | STORY ID: {row['story_id']} | COMPOUND: {sentence}")
            continue

        compound_start_idx, compound_end_idx = span
        compound_tokens = tokens[compound_start_idx:compound_end_idx]

        # --- FIXED split: reconstruct non-head, then skip separators before head ---
        non_head_ipa = phonemize_word(non_head)
        head_ipa = phonemize_word(head)

        non_n = 0
        reconstructed_non = ""

        for k, tok in enumerate(compound_tokens):
            t = str(tok)

            if _is_special_token(t):
                continue

            # stop non-head reconstruction if we've already reached it
            if reconstructed_non == non_head_ipa:
                break

            reconstructed_non += t
            non_n = k + 1

            if reconstructed_non == non_head_ipa:
                break

        # If exact non-head match failed, skip (prevents wrong sums)
        if reconstructed_non != non_head_ipa:
            print(f"SKIP (non-head mismatch) | STORY ID: {row['story_id']} | COMPOUND: {sentence}")
            print(f"IPA STORY: {ipa_story}")
            print(f"IPA COMP : {ipa_compound}")
            print(f"IPA split target -> non-head: {non_head_ipa} | head: {head_ipa}")
            print(f"Reconstructed non-head: {reconstructed_non}")
            continue

        # Skip separator tokens (spaces/empties) before head starts
        head_start_in_span = non_n
        while head_start_in_span < len(compound_tokens):
            t = str(compound_tokens[head_start_in_span])
            if _is_special_token(t):
                head_start_in_span += 1
                continue
            if _compact(t) == "":
                head_start_in_span += 1
                continue
            break

        # Optional verification of head reconstruction
        reconstructed_head = ""
        for t in compound_tokens[head_start_in_span:]:
            t = str(t)
            if _is_special_token(t):
                continue
            reconstructed_head += t

        if reconstructed_head != head_ipa:
            print(f"SKIP (head mismatch) | STORY ID: {row['story_id']} | COMPOUND: {sentence}")
            print(f"IPA STORY: {ipa_story}")
            print(f"IPA COMP : {ipa_compound}")
            print(f"IPA split target -> non-head: {non_head_ipa} | head: {head_ipa}")
            print(f"Reconstructed head: {reconstructed_head}")
            continue

        surprisal_non_head = sum(surprisal_values[compound_start_idx : compound_start_idx + non_n])
        surprisal_head = sum(surprisal_values[compound_start_idx + head_start_in_span : compound_end_idx])

        data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])

        print(f"STORY ID: {row['story_id']}")
        print(f"COMPOUND: {sentence}")
        print(f"Compound token span: [{compound_start_idx}, {compound_end_idx})")
        print(f"{sentence}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")


# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

    data = []
    process_pairs(lm, None, data)

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")