import json
import re
from collections import defaultdict

import pandas as pd
from g2p_plus import transcribe_utterances
from minicons import scorer
from transformers import AutoTokenizer

models = [
    "phonemetransformers/GPT2-85M-BPE-PHON"
]

BOS = False
input_file = "berent&pinker/compounds_with_stories_experiment_3.json"
output_file = "results_berent&pinker/100M/results_experiment_3_babble_phonetic_BPE_with_spaces_stories.csv"
translations_file = "text_to_phonemes/translations_compounds_with_stories_experiment_3_babble_phonetic_BPE_with_spaces.txt"

with open(input_file, "r", encoding="utf-8") as f:
    compound_groups_data = json.load(f)

compound_groups = [
    (group["non_heads"], group["heads"], group["stories"])
    for group in compound_groups_data
]

cat_labels = {
    0: "Sibilant Singular",
    1: "Sibilant Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

failed_words = []
failed_words_seen = set()

manual_ipa = {
    "10,000": "t ɛ n WORD_BOUNDARY θ aʊ z ə n d WORD_BOUNDARY",
    "h&r": "eɪ t̠ʃ WORD_BOUNDARY æ n d WORD_BOUNDARY ɑ ɹ WORD_BOUNDARY",
}

manual_pattern = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(manual_ipa, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)

def register_failed_word(word):
    word = word.strip()
    if word != "" and word not in failed_words_seen:
        failed_words_seen.add(word)
        failed_words.append(word)

def basic_clean_text(text):
    text = str(text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\bFedEx\b", "fedex", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_ws(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def strip_edge_word_boundaries(ipa_text):
    ipa_text = normalize_ws(ipa_text)
    ipa_text = re.sub(r"^(WORD_BOUNDARY\s+)+", "", ipa_text)
    ipa_text = re.sub(r"(\s+WORD_BOUNDARY)+$", "", ipa_text)
    return normalize_ws(ipa_text)

def safe_transcribe_once(text, keep_word_boundaries=True):
    text = basic_clean_text(text)
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

def text_to_ipa_full_or_word_by_word(text, keep_word_boundaries=True):
    text = basic_clean_text(text)
    if text == "":
        return ""

    matches = list(manual_pattern.finditer(text))

    if len(matches) == 0:
        ipa = safe_transcribe_once(text, keep_word_boundaries=keep_word_boundaries)
        if ipa != "":
            return ipa
    else:
        parts = []
        last = 0
        full_failed = False

        for m in matches:
            left = basic_clean_text(text[last:m.start()])
            matched_text = m.group(0)
            matched_ipa = manual_ipa[matched_text.lower()]

            if left != "":
                left_ipa = safe_transcribe_once(left, keep_word_boundaries=keep_word_boundaries)
                if left_ipa == "":
                    full_failed = True
                    break
                parts.append(left_ipa)

            parts.append(normalize_ws(matched_ipa))
            last = m.end()

        if not full_failed:
            right = basic_clean_text(text[last:])
            if right != "":
                right_ipa = safe_transcribe_once(right, keep_word_boundaries=keep_word_boundaries)
                if right_ipa == "":
                    full_failed = True
                else:
                    parts.append(right_ipa)

        if not full_failed:
            if keep_word_boundaries:
                cleaned_parts = []
                for part in parts:
                    cleaned_parts.append(strip_edge_word_boundaries(part))
                return normalize_ws(" WORD_BOUNDARY ".join(cleaned_parts) + " WORD_BOUNDARY")
            return normalize_ws(" ".join(parts))

    words = text.split()
    ipa_words = []

    for word in words:
        word_l = word.lower()

        if word_l in manual_ipa:
            ipa_word = normalize_ws(manual_ipa[word_l])
        else:
            ipa_word = safe_transcribe_once(word, keep_word_boundaries=keep_word_boundaries)

        if ipa_word == "":
            register_failed_word(word)
            continue

        if keep_word_boundaries:
            ipa_word = strip_edge_word_boundaries(ipa_word)

        ipa_words.append(ipa_word)

    if len(ipa_words) == 0:
        return ""

    if keep_word_boundaries:
        return normalize_ws(" WORD_BOUNDARY ".join(ipa_words) + " WORD_BOUNDARY")

    return normalize_ws(" ".join(ipa_words))

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

def find_unique_substring_span(text, target):
    matches = []
    start = 0
    while True:
        pos = text.find(target, start)
        if pos == -1:
            break
        matches.append(pos)
        start = pos + 1

    if len(matches) == 0:
        raise ValueError(f"Could not find target IPA in story IPA.\nTARGET={target}\nSTORY={text}")
    if len(matches) > 1:
        raise ValueError(f"Target IPA appears multiple times in story IPA.\nTARGET={target}")

    return matches[0], matches[0] + len(target)

def find_compound_ipa_span(ipa_story, ipa_compound):
    ipa_story = normalize_ws(ipa_story)
    ipa_compound = normalize_ws(ipa_compound)

    ipa_parts = [part.strip() for part in ipa_compound.split("WORD_BOUNDARY") if part.strip()]
    if len(ipa_parts) != 2:
        raise ValueError(f"Expected exactly 2 IPA words in compound, got {len(ipa_parts)}: {ipa_compound}")

    non_head_ipa = normalize_ws(ipa_parts[0])
    head_ipa = normalize_ws(ipa_parts[1])
    target = normalize_ws(f"{non_head_ipa} WORD_BOUNDARY {head_ipa}")

    target_start_raw, target_end_raw = find_unique_substring_span(ipa_story, target)
    head_start_raw = target_start_raw + len(non_head_ipa) + len(" WORD_BOUNDARY ")

    return target_start_raw, head_start_raw, target_end_raw

def split_surprisal_in_story(lm, ipa_story, tok_scores, target_start_raw, head_start_raw, target_end_raw):
    enc = lm.tokenizer(
        ipa_story,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    offsets = enc["offset_mapping"]

    if len(tok_scores) == len(offsets) + 1:
        toks_only = tok_scores[1:]
    elif len(tok_scores) == len(offsets):
        toks_only = tok_scores
    else:
        raise ValueError(
            f"Offsets/token mismatch: offsets={len(offsets)} vs tok_scores={len(tok_scores)}"
        )

    non_head_sum = 0.0
    head_sum = 0.0

    for tok_info, (start, end) in zip(toks_only, offsets):
        s = tok_info[1]

        if end <= target_start_raw or start >= target_end_raw:
            continue

        if end <= head_start_raw:
            non_head_sum += s
        else:
            head_sum += s

    return non_head_sum, head_sum

def process_pairs(lm, decoder_tokenizer, data, translation_records):
    for group_idx, (non_heads, heads, stories) in enumerate(compound_groups, start=1):
        group_records = []

        for head in heads:
            for i, (non_head, story) in enumerate(zip(non_heads, stories)):
                category_name = cat_labels[i]

                ortho_compound = f"{non_head} {head}"

                ipa_non_head = text_to_ipa_full_or_word_by_word(non_head, keep_word_boundaries=False)
                ipa_head = text_to_ipa_full_or_word_by_word(head, keep_word_boundaries=False)
                ipa_story = text_to_ipa_full_or_word_by_word(story, keep_word_boundaries=True)

                if ipa_non_head == "" or ipa_head == "" or ipa_story == "":
                    raise ValueError(
                        f"Empty IPA after conversion.\n"
                        f"NON_HEAD={non_head}\nHEAD={head}\nSTORY={story}"
                    )

                ipa_compound = normalize_ws(f"{ipa_non_head} WORD_BOUNDARY {ipa_head}")

                tok_scores = lm.token_score(
                    ipa_story,
                    bos_token=BOS,
                    prob=False,
                    surprisal=True,
                    bow_correction=True
                )[0]

                raw_tokens = [tok for tok, s, *_ in tok_scores]
                decoded_tokens = [
                    token_to_ipa(tok, decoder_tokenizer.byte_decoder)
                    for tok in raw_tokens
                ]

                target_start_raw, head_start_raw, target_end_raw = find_compound_ipa_span(
                    ipa_story, ipa_compound
                )

                print("\nSTORY:", story)
                print("ORTHO COMPOUND:", ortho_compound)
                print("IPA STORY:", ipa_story)
                print("IPA COMPOUND:", ipa_compound)

                print("TOK_SCORES:")
                for tok_info, decoded_tok in zip(tok_scores, decoded_tokens):
                    s = tok_info[1]
                    print(f"{repr(decoded_tok):<20} {s:.7f}")

                surprisal_non_head, surprisal_head = split_surprisal_in_story(
                    lm,
                    ipa_story,
                    tok_scores,
                    target_start_raw,
                    head_start_raw,
                    target_end_raw
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

                group_records.append({
                    "item_idx": i + 1,
                    "compound_ortho": ortho_compound,
                    "compound_ipa": ipa_compound,
                    "story_ortho": story,
                    "story_ipa": ipa_story,
                })

        translation_records.append({
            "group_idx": group_idx,
            "items": group_records
        })

for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

    force_bow_g_marker(lm)

    decoder_tokenizer = AutoTokenizer.from_pretrained(
        "phonemetransformers/babble-tokenizers",
        subfolder="BABYLM-TOKENIZER-BPE-PHON",
        use_fast=False
    )

    data = []
    translation_records = []
    process_pairs(lm, decoder_tokenizer, data, translation_records)

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    with open(translations_file, "w", encoding="utf-8") as f:
        for group in translation_records:
            f.write(f"GROUP {group['group_idx']}\n")
            f.write("------------------------------------------------------------\n")

            for item in group["items"]:
                f.write(f"ITEM {item['item_idx']}\n")
                f.write(f"COMPOUND ORTHO: {item['compound_ortho']}\n")
                f.write(f"COMPOUND IPA:   {item['compound_ipa']}\n")
                f.write(f"STORY ORTHO:    {item['story_ortho']}\n")
                f.write(f"STORY IPA:      {item['story_ipa']}\n")
                f.write("\n")

            f.write("\n")

        f.write("FAILED WORDS\n")
        f.write("------------------------------------------------------------\n")
        if len(failed_words) == 0:
            f.write("None\n")
        else:
            for word in failed_words:
                f.write(f"{word}\n")

    print(f"\nresults in results_berent&pinker folder.\n")