import json
import re
import pandas as pd
from g2p import make_g2p
from minicons import scorer

output_file = "results_berent&pinker/100M/results_experiment_3_phoneme_llama_with_spaces_stories.csv"
translations_file = "text_to_phonemes/compounds_with_stories_experiment_3_ipa.txt"
model_name = "bbunzeck/phoneme-llama"

with open("berent&pinker/compounds_with_stories_experiment_3.json", "r", encoding="utf-8") as f:
    stimuli_data = json.load(f)

cat_labels = {
    0: "Sibilant Singular",
    1: "Sibilant Plural",
    2: "Regular Singular",
    3: "Regular Plural",
}

g2p = make_g2p("eng", "eng-ipa")

translation_warnings = []

manual_ipa = {
    "noblemen": "noʊbʌlmɛn",
    "firemen": "faɪrmɛn",
    "boatmen": "boʊtmɛn",
    "labourer": "leɪbɜ˞ɜ˞",
    "labourers": "leɪbɜ˞ɜ˞z",
    "classifier": "klæsʌfaɪɜ˞",
    "evaluators": "ɪvæljueɪtɜ˞z",
    "attractor": "ʌtɹæktɜ˞",
    "evaluator": "ɪvæljueɪtɜ˞",
    "avoiders": "ʌvɔɪdɜ˞z",
    "10,000": "tɛn θaʊzʌnd"
}

manual_pattern = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(manual_ipa, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)

def add_warning(kind, text, ipa="", context=""):
    translation_warnings.append({
        "kind": kind,
        "text": text,
        "ipa": ipa,
        "context": context
    })

def basic_clean_text(text):
    text = str(text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_ws(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def looks_suspicious_ipa(ipa_text):
    ipa_text = str(ipa_text).strip()

    if ipa_text == "":
        return True
    if any(ch.isdigit() for ch in ipa_text):
        return True
    if "[" in ipa_text or "]" in ipa_text or "{" in ipa_text or "}" in ipa_text:
        return True
    if "<" in ipa_text or ">" in ipa_text:
        return True

    return False

def word_to_ipa(word):
    word_l = word.lower()

    if word_l in manual_ipa:
        return manual_ipa[word_l]

    try:
        out = g2p(word)
        ipa_word = normalize_ws(str(out))
    except Exception:
        ipa_word = ""
        add_warning("g2p_exception", word, "", "exception during isolated word conversion")

    if looks_suspicious_ipa(ipa_word):
        add_warning("suspicious_translation", word, ipa_word, "isolated word conversion")

    return ipa_word

def text_to_ipa_full(text):
    text = basic_clean_text(text)
    if text == "":
        return ""

    matches = list(manual_pattern.finditer(text))

    if len(matches) == 0:
        try:
            out = g2p(text)
            ipa_text = normalize_ws(str(out))
        except Exception:
            ipa_text = ""
            add_warning("g2p_exception", text, "", "exception during full-text conversion")

        if looks_suspicious_ipa(ipa_text):
            add_warning("suspicious_translation", text, ipa_text, "full-text conversion")

        return ipa_text

    parts = []
    last = 0

    for m in matches:
        left = basic_clean_text(text[last:m.start()])
        matched_word = m.group(0)
        matched_ipa = manual_ipa[matched_word.lower()]

        if left != "":
            try:
                out = g2p(left)
                left_ipa = normalize_ws(str(out))
            except Exception:
                left_ipa = ""
                add_warning("g2p_exception", left, "", "exception during full-text conversion around manual override")

            if looks_suspicious_ipa(left_ipa):
                add_warning("suspicious_translation", left, left_ipa, "full-text conversion around manual override")

            if left_ipa == "":
                return ""

            parts.append(left_ipa)

        parts.append(matched_ipa)
        last = m.end()

    right = basic_clean_text(text[last:])
    if right != "":
        try:
            out = g2p(right)
            right_ipa = normalize_ws(str(out))
        except Exception:
            right_ipa = ""
            add_warning("g2p_exception", right, "", "exception during full-text conversion around manual override")

        if looks_suspicious_ipa(right_ipa):
            add_warning("suspicious_translation", right, right_ipa, "full-text conversion around manual override")

        if right_ipa == "":
            return ""

        parts.append(right_ipa)

    return normalize_ws(" ".join(parts))

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

def find_compound_ipa_span(ipa_story, ipa_non_head, ipa_head):
    ipa_story = normalize_ws(ipa_story)
    ipa_non_head = normalize_ws(ipa_non_head)
    ipa_head = normalize_ws(ipa_head)

    target = f"{ipa_non_head} {ipa_head}"
    target_start_raw, target_end_raw = find_unique_substring_span(ipa_story, target)
    head_start_raw = target_start_raw + len(ipa_non_head) + 1

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

print(f"\nLoading phoneme model: {model_name}...")
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

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

    for item_idx, (non_head, story_text) in enumerate(zip(non_heads, stories), start=1):
        category_name = cat_labels[item_idx - 1]
        non_head = str(non_head).strip()
        story_text = str(story_text).strip()

        ipa_non_head = word_to_ipa(non_head)
        ipa_head = word_to_ipa(head)
        ipa_story = text_to_ipa_full(story_text)

        if ipa_non_head == "" or ipa_head == "" or ipa_story == "":
            raise ValueError(
                f"Empty IPA after conversion.\n"
                f"NON_HEAD={non_head}\nHEAD={head}\nSTORY={story_text}"
            )

        target_start_raw, head_start_raw, target_end_raw = find_compound_ipa_span(
            ipa_story, ipa_non_head, ipa_head
        )

        tok_scores = lm.token_score(
            ipa_story,
            bos_token=True,
            prob=False,
            surprisal=True,
            bow_correction=True
        )[0]

        print("\nORTHO STORY:", story_text)
        print("COMPOUND:", f"{non_head} {head}")
        print("IPA (folded):", ipa_story)

        print("TOK_SCORES:")
        for tok, s in tok_scores:
            print(f"{repr(tok):<20} {s:.7f}")

        surprisal_non_head, surprisal_head = split_surprisal_in_story(
            lm, ipa_story, tok_scores, target_start_raw, head_start_raw, target_end_raw
        )

        print(f"  Non-Head ({non_head}): {surprisal_non_head}")
        print(f"  Head     ({head}): {surprisal_head}")

        data.append([
            category_name,
            non_head,
            head,
            surprisal_non_head,
            surprisal_head
        ])

        group_records.append({
            "item_idx": item_idx,
            "compound_ortho": f"{non_head} {head}",
            "compound_ipa": f"{ipa_non_head} {ipa_head}",
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

print(f"\nresults in results_berent&pinker folder.\n")
print(f"translations file saved in: {translations_file}\n")

print("\nTRANSLATION WARNINGS:")
if len(translation_warnings) == 0:
    print("None.")
else:
    for w in translation_warnings:
        print(
            f"- kind={w['kind']} | text={w['text']} | ipa={w['ipa']} | context={w['context']}"
        )