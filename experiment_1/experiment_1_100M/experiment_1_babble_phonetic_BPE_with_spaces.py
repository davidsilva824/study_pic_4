import json
from collections import defaultdict

import pandas as pd
from g2p_plus import transcribe_utterances
from minicons import scorer
from transformers import AutoTokenizer

models = [
    "phonemetransformers/GPT2-85M-BPE-PHON"
]

BOS = False
output_file = "results_experiment_1/100M/results_experiment_1_babble_phonetic_BPE_with_spaces.csv"

# Obtaining the compounds from the json file.
with open("experiment_1/compounds_experiment_1.json", "r", encoding="utf-8") as f:
    compound_groups_data = json.load(f)

compound_groups = [
    (group["non_heads"], group["heads"])
    for group in compound_groups_data
]

cat_labels = {
    0: "Irregular Singular",
    1: "Irregular Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}


def force_bow_g_marker(lm):
    bow_symbol = "Ġ"
    lm.is_bow_tokenizer = True
    lm.bow_symbol = bow_symbol

    bow_subwords = defaultdict(bool)

    for word, idx in lm.tokenizer.get_vocab().items():
        bow_subwords[idx] = len(word) > 0 and word[0] == bow_symbol

    for idx in lm.tokenizer.get_added_vocab().values():
        bow_subwords[idx] = False

    lm.bow_subwords = bow_subwords
    lm.bow_subword_idx = [k for k, v in bow_subwords.items() if v]

    print("len(bow_subword_idx) =", len(lm.bow_subword_idx))
    print("Forced BOW settings applied successfully.")
    print("-" * 30)


def token_to_ipa(token_str, byte_decoder):
    """
    Converts GPT-2 byte-level token strings back to readable IPA-ish text.
    Keeps Ġ as word-start marker in the output.
    """
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


def compact_ipa(ipa_segment):
    return "".join(ipa_segment.split())


def split_surprisal_by_ipa_words(decoded_tokens, surprisal_values, ipa_text):
    """
    Split surprisal into first IPA word vs second IPA word.
    Uses the WORD_BOUNDARY structure in the IPA text,
    then reconstructs the first IPA word from decoded tokenizer tokens.
    """
    ipa_parts = [part.strip() for part in ipa_text.split("WORD_BOUNDARY") if part.strip()]
    if len(ipa_parts) != 2:
        raise ValueError(f"Expected exactly 2 IPA words, got {len(ipa_parts)} in: {ipa_text}")

    non_head_ipa = compact_ipa(ipa_parts[0])

    start_idx = 1 if decoded_tokens and decoded_tokens[0] == "UTT_BOUNDARY" else 0

    reconstructed = ""
    non_n = 0

    for k in range(start_idx, len(decoded_tokens)):
        piece = decoded_tokens[k].lstrip("Ġ").replace(" ", "")
        reconstructed += piece
        non_n += 1
        if reconstructed == non_head_ipa:
            break

    if reconstructed != non_head_ipa:
        raise ValueError(
            f"Could not reconstruct first IPA word.\n"
            f"Target: {non_head_ipa}\n"
            f"Got:    {reconstructed}\n"
            f"Tokens: {decoded_tokens}"
        )

    total_real_tokens = len(decoded_tokens) - start_idx
    head_n = total_real_tokens - non_n

    surprisal_non_head = sum(surprisal_values[start_idx:start_idx + non_n])
    surprisal_head = sum(surprisal_values[start_idx + non_n:start_idx + non_n + head_n])

    return surprisal_non_head, surprisal_head


def process_pairs(lm, decoder_tokenizer, data):
    for non_heads, heads in compound_groups:
        # Loop over HEADS first
        for head in heads:
            for i, non_head in enumerate(non_heads):
                category_name = cat_labels[i]

                sentence = f"{non_head} {head}"

                ipa_text = transcribe_utterances(
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

                raw_tokens = [tok for tok, s, *_ in tok_scores]
                surprisal_values = [s for tok, s, *_ in tok_scores]
                decoded_tokens = [
                    token_to_ipa(tok, decoder_tokenizer.byte_decoder)
                    for tok in raw_tokens
                ]

                # --- Print Block ---
                print("\nORTHO:", sentence)
                print("IPA   :", ipa_text)
                print(' '.join(f'{tok:>12}' for tok in decoded_tokens))
                print(' '.join(f'{s:>12.3f}' for s in surprisal_values))
                print(surprisal_values)

                surprisal_non_head, surprisal_head = split_surprisal_by_ipa_words(
                    decoded_tokens,
                    surprisal_values,
                    ipa_text
                )

                data.append([
                    category_name,
                    non_head,
                    head,
                    surprisal_non_head,
                    surprisal_head
                ])

                print(f"{sentence}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")


# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

    force_bow_g_marker(lm)

    # Matching slow tokenizer only for readable byte-decoding of output tokens
    decoder_tokenizer = AutoTokenizer.from_pretrained(
        "phonemetransformers/babble-tokenizers",
        subfolder="BABYLM-TOKENIZER-BPE-PHON",
        use_fast=False
    )

    data = []
    process_pairs(lm, decoder_tokenizer, data)

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f'\nresults in results_experiment_1 folder.\n')