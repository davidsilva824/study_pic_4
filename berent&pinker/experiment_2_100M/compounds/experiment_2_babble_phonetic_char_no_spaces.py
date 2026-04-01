### This code is complete.
# BOS = False
# IMPORTANT note: the tokenizer of the model removes the spaces automatically. So it is indiferent if you put keep_word_boundaries=False or true.
# Since this model does not use word separation at all, the BOW correction should be kept False.
# The IPA conversion now tries the whole text first, and only splits recursively if necessary.

import json
import re
import pandas as pd
from g2p_plus import transcribe_utterances
from minicons import scorer

models = [
    "phonemetransformers/GPT2-85M-CHAR-PHON-SPACELESS"
]

BOS = False
output_file = "results_berent&pinker/100M/results_experiment_2_babble_phonetic_char_no_spaces.csv"

with open("berent&pinker/compounds_experiment_2.json", "r", encoding="utf-8") as f:
    compound_groups_data = json.load(f)

compound_groups = [
    (group["non_heads"], group["heads"])
    for group in compound_groups_data
]

cat_labels = {
    0: "Sibilant Singular",
    1: "Sibilant Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

def normalize_ws(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def safe_transcribe_once(text, keep_word_boundaries=True):
    text = normalize_ws(text)
    if text == "":
        return ""
    try:
        out = transcribe_utterances(
            [text],
            backend="phonemizer",
            language="en-us",
            keep_word_boundaries=keep_word_boundaries
        )[0]
        return normalize_ws(out)
    except Exception:
        return ""

def strip_edge_word_boundaries(ipa_text):
    ipa_text = normalize_ws(ipa_text)
    ipa_text = re.sub(r"^(WORD_BOUNDARY\s+)+", "", ipa_text)
    ipa_text = re.sub(r"(\s+WORD_BOUNDARY)+$", "", ipa_text)
    return normalize_ws(ipa_text)

def join_ipa_chunks(left_ipa, right_ipa, keep_word_boundaries=True):
    left_ipa = normalize_ws(left_ipa)
    right_ipa = normalize_ws(right_ipa)

    if left_ipa == "":
        return right_ipa
    if right_ipa == "":
        return left_ipa

    if keep_word_boundaries:
        left_ipa = strip_edge_word_boundaries(left_ipa)
        right_ipa = strip_edge_word_boundaries(right_ipa)
        return normalize_ws(f"{left_ipa} WORD_BOUNDARY {right_ipa}")

    return normalize_ws(f"{left_ipa} {right_ipa}")

def text_to_ipa_divide(text, keep_word_boundaries=True):
    text = normalize_ws(text)
    if text == "":
        return ""

    ipa = safe_transcribe_once(text, keep_word_boundaries=keep_word_boundaries)
    if ipa != "":
        return ipa

    words = text.split()
    if len(words) <= 1:
        simpler = re.sub(r"[^\w'-]", "", text).strip()
        if simpler != "" and simpler != text:
            ipa = safe_transcribe_once(simpler, keep_word_boundaries=keep_word_boundaries)
            if ipa != "":
                return ipa
        return ""

    mid = len(words) // 2
    left_text = " ".join(words[:mid])
    right_text = " ".join(words[mid:])

    left_ipa = text_to_ipa_divide(left_text, keep_word_boundaries=keep_word_boundaries)
    right_ipa = text_to_ipa_divide(right_text, keep_word_boundaries=keep_word_boundaries)

    return join_ipa_chunks(left_ipa, right_ipa, keep_word_boundaries=keep_word_boundaries)

def split_surprisal_by_offsets(lm, ipa_sentence, tok_scores, boundary):
    enc = lm.tokenizer(ipa_sentence, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    toks_only = tok_scores[1:]  # drop UTT_BOUNDARY
    if len(offsets) != len(toks_only):
        raise ValueError(f"Offsets/token mismatch: offsets={len(offsets)} vs toks={len(toks_only)}")

    non_head_sum = 0.0
    head_sum = 0.0

    for (tok, s, *_), (start, end) in zip(toks_only, offsets):
        if end <= boundary:
            non_head_sum += s
        else:
            head_sum += s

    return non_head_sum, head_sum

def process_pairs(lm, data):
    for non_heads, heads in compound_groups:
        for head in heads:
            for i, non_head in enumerate(non_heads):
                category_name = cat_labels[i]

                sentence = f"{non_head} {head}"

                ipa_with_boundary = text_to_ipa_divide(sentence, keep_word_boundaries=True)
                if ipa_with_boundary == "":
                    raise ValueError(f"Empty IPA after recursive conversion for: {sentence}")

                boundary_marker = " WORD_BOUNDARY "
                boundary_pos = ipa_with_boundary.find(boundary_marker)
                if boundary_pos == -1:
                    raise ValueError(f"Could not find WORD_BOUNDARY in IPA sentence: {ipa_with_boundary}")

                ipa_sentence = normalize_ws(ipa_with_boundary.replace("WORD_BOUNDARY", ""))
                boundary = boundary_pos

                tok_scores = lm.token_score(
                    ipa_sentence,
                    bos_token=BOS,
                    prob=False,
                    surprisal=True,
                    bow_correction=False
                )[0]

                print("\nORTHO:", sentence)
                print("IPA (folded):", ipa_sentence)

                print("TOK_SCORES:")
                for tok, s in tok_scores:
                    print(f"{repr(tok):<20} {s:.7f}")

                surprisal_non_head, surprisal_head = split_surprisal_by_offsets(
                    lm, ipa_sentence, tok_scores, boundary
                )

                data.append([
                    category_name,
                    non_head,
                    head,
                    surprisal_non_head,
                    surprisal_head
                ])

                print(f"  Non-Head ({non_head}): {surprisal_non_head}")
                print(f"  Head     ({head}): {surprisal_head}")

for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

    data = []
    process_pairs(lm, data)

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f'\nresults in results_berent&pinker folder.\n')