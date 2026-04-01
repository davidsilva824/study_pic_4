from g2p import make_g2p
import json

input_file = "berent&pinker/compounds_with_stories_experiment_3.json"
output_file = "text_to_phonemes/compounds_with_stories_experiment_3_ipa.txt"

with open(input_file, "r", encoding="utf-8") as f:
    stimuli_data = json.load(f)

transducer = make_g2p("eng", "eng-ipa")

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

                compound_ipa = transducer(compound).output_string
                story_ipa = transducer(story).output_string

                out.write(f"ITEM {i}\n")
                out.write(f"COMPOUND ORTHO: {compound}\n")
                out.write(f"COMPOUND IPA:   {compound_ipa}\n")
                out.write(f"STORY ORTHO:    {story}\n")
                out.write(f"STORY IPA:      {story_ipa}\n")
                out.write("\n")

        out.write("\n")

print(f"IPA translations written to: {output_file}")