### same thing but with the spliting of text in case it is too big


from g2p_plus import transcribe_utterances
import json

input_file = "berent&pinker/compounds_with_stories_experiment_2.json"
output_file = "berent&pinker/compounds_with_stories_experiment_2_ipa_g2p_plus.txt"

with open(input_file, "r", encoding="utf-8") as f:
    stimuli_data = json.load(f)

def normalize_ws(text):
    return " ".join(str(text).split())

def nearest_middle_space(text):
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if not spaces:
        return None
    mid = len(text) / 2
    return min(spaces, key=lambda i: abs(i - mid))

def phonemize_story_recursive(text):
    text = normalize_ws(text)

    try:
        ipa = transcribe_utterances(
            [text],
            "phonemizer",
            "en-us",
            keep_word_boundaries=True
        )[0]
        return normalize_ws(ipa)

    except Exception:
        split_idx = nearest_middle_space(text)
        if split_idx is None:
            raise

        left = text[:split_idx].strip()
        right = text[split_idx + 1:].strip()

        if not left or not right:
            raise

        left_ipa = phonemize_story_recursive(left)
        right_ipa = phonemize_story_recursive(right)

        return normalize_ws(f"{left_ipa} WORD_BOUNDARY {right_ipa}")

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

                compound_ipa = transcribe_utterances(
                    [compound],
                    "phonemizer",
                    "en-us",
                    keep_word_boundaries=True
                )[0]

                story_ipa = phonemize_story_recursive(story)

                out.write(f"ITEM {i}\n")
                out.write(f"COMPOUND ORTHO: {compound}\n")
                out.write(f"COMPOUND IPA:   {compound_ipa}\n")
                out.write(f"STORY ORTHO:    {story}\n")
                out.write(f"STORY IPA:      {story_ipa}\n")
                out.write("\n")

        out.write("\n")

print(f"IPA translations written to: {output_file}")