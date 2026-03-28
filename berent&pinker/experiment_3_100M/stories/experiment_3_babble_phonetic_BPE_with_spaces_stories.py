### This code is complete.
### Adapted in the simplest way: score FULL STORY, then extract surprisal for the target compound span.
### + KEEP forced BOW correction
### + CHANGE only the full-story phonemization: try full sentence first, fallback split at closest space to the middle

import pandas as pd
from g2p_plus import transcribe_utterances
from minicons import scorer
from transformers import AutoTokenizer
from collections import defaultdict

# -----------------------------------------------

models = [
    "phonemetransformers/GPT2-85M-BPE-PHON"
]

BOS = False

# --- CHANGED: use compounds + stories ---
compounds_file = "berent&pinker/stimuli_compounds_experiment_3.csv"
stories_file = "berent&pinker/stimuli_stories_experiment_3.csv"
output_file = "results_berent&pinker/results_experiment_3_babble_phonetic_BPE_stories.csv"

compounds_df = pd.read_csv(compounds_file)
stories_df = pd.read_csv(stories_file)
stimuli_df = pd.merge(compounds_df, stories_df, on="story_id", how="inner")
# ---------------------------------------------------------------

cat_labels = {
    "a": "singular_1",
    "b": "plural_1",
    "c": "singular_2",
    "d": "plural_2",
}

# --- Function to force BOW settings for this model ---
def force_bow_settings(lm, bow_symbol="Ġ"):
    lm.is_bow_tokenizer = True
    lm.bow_symbol = bow_symbol

    bow_subwords = defaultdict(bool)

    for word, idx in lm.tokenizer.get_vocab().items():
        bow_subwords[idx] = (len(word) > 0 and word[0] == bow_symbol)

    for idx in lm.tokenizer.get_added_vocab().values():
        bow_subwords[idx] = False

    lm.bow_subwords = bow_subwords
    lm.bow_subword_idx = [k for k, v in lm.bow_subwords.items() if v]
# ----------------------------------------------------------


def token_to_ipa(token_str, byte_decoder):
    if token_str == "UTT_BOUNDARY":
        return "UTT_BOUNDARY"

    has_word_start = token_str.startswith("Ġ")
    core = token_str[1:] if has_word_start else token_str

    try:
        b = bytes([byte_decoder[c] for c in core])
        decoded = b.decode("utf-8")
    except Exception:
        decoded = core

    if has_word_start:
        return "Ġ" + decoded
    return decoded


def word_to_ipa_no_boundaries(word):
    ipa = transcribe_utterances(
        [word],
        backend="phonemizer",
        language="en-us",
        keep_word_boundaries=False
    )[0]
    return ipa.replace(" ", "").strip()


# --- CHANGED: full-story phonemization with fallback split at closest space to middle ---
def phonemize_story_with_mid_space_split(text):
    text = str(text).strip()
    if not text:
        return ""

    out = transcribe_utterances(
        [text],
        backend="phonemizer",
        language="en-us",
        keep_word_boundaries=True
    )[0]

    out = "" if out is None else str(out).strip()
    if out:
        return out

    mid = len(text) // 2

    left_space = text.rfind(" ", 0, mid + 1)
    right_space = text.find(" ", mid)

    if left_space == -1 and right_space == -1:
        return ""

    if left_space == -1:
        split_idx = right_space
    elif right_space == -1:
        split_idx = left_space
    else:
        split_idx = left_space if (mid - left_space) <= (right_space - mid) else right_space

    left_text = text[:split_idx].strip()
    right_text = text[split_idx + 1:].strip()

    left_ipa = phonemize_story_with_mid_space_split(left_text) if left_text else ""
    right_ipa = phonemize_story_with_mid_space_split(right_text) if right_text else ""

    return " ".join(x for x in [left_ipa, right_ipa] if x).strip()


def process_pairs(lm, tok_decoder, pairs, data):

    for _, row in stimuli_df.iterrows():
        suffix = str(row["story_id"]).strip().split("_")[-1]
        category_name = cat_labels[suffix]

        sentence = str(row["compound"]).strip()
        story_text = str(row["story_text"]).strip()
        non_head, head = sentence.split(" ", 1)

        # CHANGED: IPA of FULL STORY (context) -> full sentence, fallback split at nearest space
        ipa_text = phonemize_story_with_mid_space_split(story_text)

        # ADDED: IPA of target compound (for locating span in story)
        ipa_compound = transcribe_utterances(
            [sentence],
            backend="phonemizer",
            language="en-us",
            keep_word_boundaries=True
        )[0]

        tok_scores = lm.token_score(
            ipa_text,
            bos_token=BOS,
            prob=False,
            surprisal=True,
            bow_correction=True
        )[0]

        tokens_raw = [tok for tok, s, *_ in tok_scores]
        surprisal_values = [s for tok, s, *_ in tok_scores]

        # Decode tokenizer tokens back to IPA-like strings
        tokens_ipa = [token_to_ipa(tok, tok_decoder.byte_decoder) for tok in tokens_raw]

        # --- Original Print Block (adapted to decoded IPA tokens) ---
        print(' '.join(f'{tok:>10}' for tok in tokens_ipa))
        print(' '.join(f'{s:>10.3f}' for s in surprisal_values))
        print(surprisal_values)

        # --- find compound span in full story tokens (simple compact match) ---
        start_idx = 1 if (len(tokens_ipa) > 0 and tokens_ipa[0] == "UTT_BOUNDARY") else 0

        def compact(s):
            return "".join(str(s).replace("WORD_BOUNDARY", " ").replace("Ġ", " ").split())

        target_compact = compact(ipa_compound)

        rebuilt = ""
        spans = []  # (token_index, start_char, end_char)

        for i in range(start_idx, len(tokens_ipa)):
            tok = tokens_ipa[i]
            if tok == "UTT_BOUNDARY":
                spans.append((i, len(rebuilt), len(rebuilt)))
                continue
            piece = compact(tok)
            s0 = len(rebuilt)
            rebuilt += piece
            s1 = len(rebuilt)
            spans.append((i, s0, s1))

        pos = rebuilt.find(target_compact)
        if pos == -1:
            print(f"SKIP (compound not found) | STORY ID: {row['story_id']} | COMPOUND: {sentence}")
            print(f"IPA STORY: {ipa_text}")
            print(f"IPA COMP: {ipa_compound}")
            continue

        end_pos = pos + len(target_compact)
        hit = []
        for tok_i, s0, s1 in spans:
            if s1 <= pos:
                continue
            if s0 >= end_pos:
                break
            if s1 > s0:
                hit.append(tok_i)

        if not hit:
            print(f"SKIP (no token span) | STORY ID: {row['story_id']} | COMPOUND: {sentence}")
            print(f"IPA STORY: {ipa_text}")
            print(f"IPA COMP: {ipa_compound}")
            continue

        compound_start_idx = hit[0]
        compound_end_idx = hit[-1] + 1
        # ------------------------------------------------------------

        # Target non-head IPA (no spaces / no WORD_BOUNDARY)
        target_non_head_ipa = word_to_ipa_no_boundaries(non_head)

        # Reconstruct non-head from decoded tokens (ONLY inside compound span)
        non_n = 0
        reconstructed_word = ""

        compound_tokens_ipa = tokens_ipa[compound_start_idx:compound_end_idx]
        cleaned_tokens = [tok.lstrip('Ġ ') for tok in compound_tokens_ipa]

        for k in range(len(cleaned_tokens)):
            piece = cleaned_tokens[k]

            # stop if token is a boundary token (defensive)
            if piece == "WORD_BOUNDARY":
                break

            reconstructed_word += piece
            non_n += 1

            if reconstructed_word == target_non_head_ipa:
                break

        total_real_tokens = len(compound_tokens_ipa)
        head_n = total_real_tokens - non_n

        surprisal_non_head = sum(surprisal_values[compound_start_idx : compound_start_idx + non_n])
        surprisal_head = sum(surprisal_values[compound_start_idx + non_n : compound_start_idx + non_n + head_n])

        data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])

        print(f"STORY ID: {row['story_id']}")
        print(f"COMPOUND: {sentence}")
        print(f"Compound token span: [{compound_start_idx}, {compound_end_idx})")
        print(f"{sentence}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")


# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

    # --- apply forced BOW method for this model ---
    force_bow_settings(lm, bow_symbol="Ġ")

    # matching tokenizer (slow) for byte_decoder
    tok_decoder = AutoTokenizer.from_pretrained(
        "phonemetransformers/babble-tokenizers",
        subfolder="BABYLM-TOKENIZER-BPE-PHON",
        use_fast=False
    )

    data = []

    process_pairs(lm, tok_decoder, None, data)

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")