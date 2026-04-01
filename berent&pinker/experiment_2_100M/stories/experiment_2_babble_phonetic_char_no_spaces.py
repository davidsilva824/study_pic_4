### This code is complete.
# BOS = False
# IMPORTANT note: the tokenizer of the model removes the spaces automatically. So it is indiferent if you put keep_word_boundaries=False or true.
# Since this model does not use word separation at all, the BOW correction should be kept False.
# The story and the compound are converted as whole texts first, and only translated word by word if the full translation returns empty.

import json
import re
import pandas as pd
from g2p_plus import transcribe_utterances
from minicons import scorer

models = [
    "phonemetransformers/GPT2-85M-CHAR-PHON-SPACELESS"
]

BOS = False
output_file = "results_berent&pinker/100M/results_experiment_2_babble_phonetic_char_no_spaces_stories.csv"
translations_file = "text_to_phonemes/translation_with_stories_experiment_2_babble_phonetic_char_no_spaces_stories.txt"

with open("berent&pinker/compounds_with_stories_experiment_2.json", "r", encoding="utf-8") as f:
    stimuli_data = json.load(f)

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
    r"(" + "|".join(re.escape(k) for k in sorted(manual_ipa, key=len, reverse=True)) + r")",
    flags=re.IGNORECASE
)

def register_failed_word(word):
    word = word.strip()
    if word != "" and word not in failed_words_seen:
        failed_words_seen.add(word)
        failed_words.append(word)

def normalize_ws(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def basic_clean_text(text):
    text = str(text)
    text = re.sub(r"\bFedEx\b", "fedex", text)
    return normalize_ws(text)

def safe_transcribe_once(text, keep_word_boundaries=False):
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

def strip_word_boundaries(ipa_text):
    ipa_text = normalize_ws(ipa_text)
    ipa_text = ipa_text.replace("WORD_BOUNDARY", "")
    return normalize_ws(ipa_text)

def get_manual_ipa(text, keep_word_boundaries=False):
    ipa_text = manual_ipa[text.lower()]
    if keep_word_boundaries:
        return normalize_ws(ipa_text)
    return strip_word_boundaries(ipa_text)

def text_to_ipa_full_or_word_by_word(text, keep_word_boundaries=False):
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
            matched_ipa = get_manual_ipa(matched_text, keep_word_boundaries=keep_word_boundaries)

            if left != "":
                left_ipa = safe_transcribe_once(left, keep_word_boundaries=keep_word_boundaries)
                if left_ipa == "":
                    full_failed = True
                    break
                parts.append(left_ipa)

            parts.append(matched_ipa)
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
            return normalize_ws(" ".join(parts))

    words = text.split()
    ipa_words = []

    for word in words:
        word_l = word.lower()

        if word_l in manual_ipa:
            ipa_word = get_manual_ipa(word, keep_word_boundaries=keep_word_boundaries)
        else:
            ipa_word = safe_transcribe_once(word, keep_word_boundaries=keep_word_boundaries)

        if ipa_word == "":
            register_failed_word(word)
            continue

        ipa_words.append(ipa_word)

    if len(ipa_words) == 0:
        return ""

    return normalize_ws(" ".join(ipa_words))

def compact_string(s):
    return "".join(ch for ch in s if not ch.isspace())

def compact_to_raw_map(s):
    return [i for i, ch in enumerate(s) if not ch.isspace()]

def find_compound_span_in_story(ipa_story, ipa_compound, ipa_non_head):
    story_compact = compact_string(ipa_story)
    compound_compact = compact_string(ipa_compound)
    non_head_compact = compact_string(ipa_non_head)

    idx_map = compact_to_raw_map(ipa_story)

    start_compact = story_compact.find(compound_compact)
    if start_compact == -1:
        raise ValueError(
            f"Compound not found in story IPA.\n"
            f"TARGET={compound_compact}\n"
            f"STORY={story_compact}"
        )

    end_compact = start_compact + len(compound_compact)

    target_start_raw = idx_map[start_compact]
    target_end_raw = idx_map[end_compact - 1] + 1

    head_start_compact = start_compact + len(non_head_compact)
    head_start_raw = idx_map[head_start_compact]

    return target_start_raw, head_start_raw, target_end_raw

def split_surprisal_in_story(lm, ipa_story, tok_scores, target_start_raw, head_start_raw, target_end_raw):
    enc = lm.tokenizer(
        ipa_story,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    offsets = enc["offset_mapping"]

    toks_only = tok_scores[1:]  # drop UTT_BOUNDARY
    if len(offsets) != len(toks_only):
        raise ValueError(f"Offsets/token mismatch: offsets={len(offsets)} vs toks={len(toks_only)}")

    non_head_sum = 0.0
    head_sum = 0.0

    for (tok, s, *_), (start, end) in zip(toks_only, offsets):
        if end <= target_start_raw or start >= target_end_raw:
            continue

        if end <= head_start_raw:
            non_head_sum += s
        else:
            head_sum += s

    return non_head_sum, head_sum

def process_pairs(lm, data, translation_records):
    for group_idx, group in enumerate(stimuli_data, start=1):
        non_heads = group["non_heads"]
        heads = group["heads"]
        stories = group["stories"]

        if len(heads) != 1:
            raise ValueError(f"Expected exactly one head in experiment 2 item, got {len(heads)}")

        head = str(heads[0]).strip()
        group_records = []

        for i, (non_head, story) in enumerate(zip(non_heads, stories)):
            category_name = cat_labels[i]

            sentence = f"{non_head} {head}"

            ipa_story = text_to_ipa_full_or_word_by_word(story, keep_word_boundaries=False)
            ipa_compound = text_to_ipa_full_or_word_by_word(sentence, keep_word_boundaries=False)
            ipa_non_head = text_to_ipa_full_or_word_by_word(non_head, keep_word_boundaries=False)

            if ipa_story == "" or ipa_compound == "" or ipa_non_head == "":
                raise ValueError(
                    f"Empty IPA after conversion.\n"
                    f"STORY={story}\nCOMPOUND={sentence}"
                )

            tok_scores = lm.token_score(
                ipa_story,
                bos_token=BOS,
                prob=False,
                surprisal=True,
                bow_correction=False
            )[0]

            target_start_raw, head_start_raw, target_end_raw = find_compound_span_in_story(
                ipa_story, ipa_compound, ipa_non_head
            )

            print("\nSTORY:", story)
            print("ORTHO COMPOUND:", sentence)
            print("IPA STORY:", ipa_story)

            print("TOK_SCORES:")
            for tok, s in tok_scores:
                print(f"{repr(tok):<20} {s:.7f}")

            surprisal_non_head, surprisal_head = split_surprisal_in_story(
                lm, ipa_story, tok_scores, target_start_raw, head_start_raw, target_end_raw
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
                "compound_ortho": sentence,
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

    data = []
    translation_records = []
    process_pairs(lm, data, translation_records)

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

    print(f'\nresults in results_berent&pinker folder.\n')