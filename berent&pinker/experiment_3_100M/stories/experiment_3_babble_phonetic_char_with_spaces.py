### This code seems to be working, but it wouldnt hurt a final reverification. 
# It uses athe conversion from text to phonemes g2plus. 
# IMPORTANT note: The Forced BOW correction is working. This can be observe in the file 'surprisal_by_token_babble_phonetic_char_with_spaces_forced_BOW.py'.
# With the correction the suprisal of the token 'WORD_BOUNDARY' becomes zero.
### This code seems to be working, but it wouldnt hurt a final reverification. 
# It uses athe conversion from text to phonemes g2plus. 
# IMPORTANT note: The Forced BOW correction is working. This can be observe in the file 'surprisal_by_token_babble_phonetic_char_with_spaces_forced_BOW.py'.
# With the correction the suprisal of the token 'WORD_BOUNDARY' becomes zero.

BOS = False

import json
import re
import pandas as pd
from collections import defaultdict
from g2p_plus import transcribe_utterances
from minicons import scorer

output_file = "results_berent&pinker/100M/results_experiment_3_babble_phonetic_char_with_spaces.csv"
translations_file = "text_to_phonemes/translation_with_stories_experiment_3_babble_phonetic_char_with_spaces.txt"
model_name = "phonemetransformers/GPT2-85M-CHAR-PHON"
json_file = "berent&pinker/compounds_with_stories_experiment_3.json"

with open(json_file, "r", encoding="utf-8") as f:
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
    return " ".join(str(text).split())

def basic_clean_text(text):
    text = str(text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\bFedEx\b", "fedex", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def safe_transcribe_once(text, keep_word_boundaries=True):
    text = basic_clean_text(text)
    if text == "":
        return ""
    try:
        out = transcribe_utterances(
            [text],
            backend="phonemizer",
            language="en-us",
            keep_word_boundaries=keep_word_boundaries,
            allow_possibly_faulty_word_boundaries=True
        )[0]
        return normalize_ws(out)
    except Exception:
        return ""

def strip_edge_word_boundaries(ipa_text):
    ipa_text = normalize_ws(ipa_text)
    ipa_text = re.sub(r"^(WORD_BOUNDARY\s+)+", "", ipa_text)
    ipa_text = re.sub(r"(\s+WORD_BOUNDARY)+$", "", ipa_text)
    return normalize_ws(ipa_text)

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

def ipa_text_to_word_list(ipa_text):
    tokens = normalize_ws(ipa_text).split()
    words = []
    current = []

    for tok in tokens:
        if tok == "WORD_BOUNDARY":
            if current:
                words.append(" ".join(current))
                current = []
        else:
            current.append(tok)

    if current:
        words.append(" ".join(current))

    return words

def find_compound_ipa_span(story_ipa_text, compound_ipa_text):
    story_words = ipa_text_to_word_list(story_ipa_text)
    compound_words = ipa_text_to_word_list(compound_ipa_text)

    if len(compound_words) != 2:
        raise ValueError(f"Expected 2 IPA words for compound, got {len(compound_words)}: {compound_ipa_text}")

    matches = []
    for i in range(len(story_words) - len(compound_words) + 1):
        if story_words[i:i + len(compound_words)] == compound_words:
            matches.append(i)

    if len(matches) == 0:
        raise ValueError(f"Could not find compound IPA inside story IPA.\nCOMPOUND IPA: {compound_ipa_text}")
    if len(matches) > 1:
        raise ValueError(f"Compound IPA appears multiple times in story IPA.\nCOMPOUND IPA: {compound_ipa_text}")

    return matches[0], len(compound_words)

def ipa_word_surprisals(lm, ipa_text):
    tok_scores = lm.token_score(
        ipa_text,
        bos_token=BOS,
        prob=False,
        surprisal=True,
        bow_correction=True
    )[0]

    print("TOK_SCORES:")
    for tok, s in tok_scores:
        print(f"{tok:<20} {s:.7f}")

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

    if current != 0.0 or not word_surps:
        word_surps.append(current)

    return word_surps

print(f"\nLoading phoneme model: {model_name}...")
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

# --- Forced BOW setup for WORD_BOUNDARY ---
bow_symbol = "WORD_BOUNDARY"
lm.is_bow_tokenizer = True
lm.bow_symbol = bow_symbol

bow_subwords = defaultdict(bool)

vocab = lm.tokenizer.get_vocab()
bow_id = vocab.get(bow_symbol, None)

for _, idx in vocab.items():
    bow_subwords[idx] = False

for idx in lm.tokenizer.get_added_vocab().values():
    bow_subwords[idx] = False

if bow_id is not None:
    bow_subwords[bow_id] = True

lm.bow_subwords = bow_subwords
lm.bow_subword_idx = [int(bow_id)] if bow_id is not None else []

print("bow_symbol =", bow_symbol)
print("bow_id =", bow_id)
print("len(bow_subword_idx) =", len(lm.bow_subword_idx))
print("-" * 30)

data = []
translation_records = []

for group_idx, group in enumerate(stimuli_data, start=1):
    non_heads = group["non_heads"]
    heads = group["heads"]
    stories = group["stories"]

    if len(heads) != 1:
        raise ValueError(f"Expected exactly one head in experiment 2 item, got {len(heads)}")

    head = str(heads[0]).strip()
    group_records = []

    for i, (non_head, story_text) in enumerate(zip(non_heads, stories)):
        category_name = cat_labels[i]

        non_head = str(non_head).strip()
        story_text = str(story_text).strip()
        compound = f"{non_head} {head}"

        print("\nORTHO:", story_text)

        ipa_story = text_to_ipa_full_or_word_by_word(story_text, keep_word_boundaries=True)
        ipa_compound = text_to_ipa_full_or_word_by_word(compound, keep_word_boundaries=True)

        print("IPA (folded):", ipa_story)

        word_surps = ipa_word_surprisals(lm, ipa_story)

        compound_start_idx, compound_len = find_compound_ipa_span(ipa_story, ipa_compound)

        s_non_head = word_surps[compound_start_idx]
        s_head = word_surps[compound_start_idx + 1]

        print(f"COMPOUND: {compound}")
        print(f"  Non-Head ({non_head}): {s_non_head}")
        print(f"  Head     ({head}): {s_head}")

        data.append([
            category_name,
            non_head,
            head,
            s_non_head,
            s_head
        ])

        group_records.append({
            "item_idx": i + 1,
            "compound_ortho": compound,
            "compound_ipa": ipa_compound,
            "story_ortho": story_text,
            "story_ipa": ipa_story,
        })

    translation_records.append({
        "group_idx": group_idx,
        "items": group_records
    })

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