from g2p_plus import transcribe_utterances
import json
import re

input_file = "berent&pinker/compounds_with_stories_experiment_3.json"
output_file = "text_to_phonemes/compounds_with_stories_experiment_3_ipa_g2p_plus.txt"

with open(input_file, "r", encoding="utf-8") as f:
    stimuli_data = json.load(f)

failed_words = []
failed_words_seen = set()

def register_failed_word(word):
    word = word.strip()
    if word != "" and word not in failed_words_seen:
        failed_words_seen.add(word)
        failed_words.append(word)

def strip_edge_boundaries(ipa_text):
    ipa_text = ipa_text.strip()
    ipa_text = re.sub(r"^(WORD_BOUNDARY\s+)+", "", ipa_text)
    ipa_text = re.sub(r"(\s+WORD_BOUNDARY)+$", "", ipa_text)
    ipa_text = re.sub(r"\s+", " ", ipa_text).strip()
    return ipa_text

def translate_one(text):
    ipa = transcribe_utterances(
        [text],
        "phonemizer",
        "en-us",
        keep_word_boundaries=True
    )[0]
    return ipa.strip()

def translate_full_or_word_by_word(text):
    ipa_full = translate_one(text)

    if ipa_full != "":
        return ipa_full

    words = text.split()
    ipa_words = []

    for word in words:
        ipa_word = translate_one(word)

        if ipa_word != "":
            ipa_word = strip_edge_boundaries(ipa_word)
            if ipa_word != "":
                ipa_words.append(ipa_word)
            else:
                register_failed_word(word)
        else:
            register_failed_word(word)

    if len(ipa_words) == 0:
        return ""

    return " WORD_BOUNDARY ".join(ipa_words) + " WORD_BOUNDARY"

with open(output_file, "w", encoding="utf-8") as out:
    for group_idx, group in enumerate(stimuli_data, start=1):
        non_heads = group["non_heads"]
        heads = group["heads"]
        stories = group["stories"]

        out.write(f"GROUP {group_idx}\n")
        out.write("-" * 60 + "\n")

        for head in heads:
            for i, (non_head, story) in enumerate(zip(non_heads, stories), start=1):
                compound = f"{non_head} {head}"

                compound_ipa = translate_full_or_word_by_word(compound)
                story_ipa = translate_full_or_word_by_word(story)

                out.write(f"ITEM {i}\n")
                out.write(f"COMPOUND ORTHO: {compound}\n")
                out.write(f"COMPOUND IPA:   {compound_ipa}\n")
                out.write(f"STORY ORTHO:    {story}\n")
                out.write(f"STORY IPA:      {story_ipa}\n")
                out.write("\n")

        out.write("\n")

    out.write("FAILED WORDS\n")
    out.write("-" * 60 + "\n")
    if len(failed_words) == 0:
        out.write("None\n")
    else:
        for word in failed_words:
            out.write(f"{word}\n")

print(f"IPA translations written to: {output_file}")

print("\nFAILED WORDS:")
if len(failed_words) == 0:
    print("None")
else:
    for word in failed_words:
        print(word)