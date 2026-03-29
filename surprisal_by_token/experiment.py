### This code is complete.
# Reads the stimuli from the JSON file.
# For each item, it generates all non_head + head combinations.
# Prints token-by-token surprisal to the terminal and saves the same output to a text file.
# For the SPACELESS phoneme model.

import json
from g2p_plus import transcribe_utterances
from minicons import scorer
from transformers import AutoTokenizer

BOS = False

model_name = "phonemetransformers/GPT2-85M-BPE-PHON-SPACELESS"
STIMULI_FILE = "compounds_experiment_1.json"
OUTPUT_FILE = "token_surprisal_per_example.txt"

# Load JSON stimuli
with open(STIMULI_FILE, "r", encoding="utf-8") as f:
    compound_groups = json.load(f)

# Load model once
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

# Load matching tokenizer once
tok = AutoTokenizer.from_pretrained(
    "phonemetransformers/babble-tokenizers",
    subfolder="BABYLM-TOKENIZER-BPE-PHON",
    use_fast=False
)

def token_to_ipa(token_str, byte_decoder):
    """
    Converts a GPT-2 byte-level token string back to readable IPA text.
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

example_id = 1

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for group_id, group in enumerate(compound_groups, start=1):
        non_heads = group["non_heads"]
        heads = group["heads"]

        for non_head in non_heads:
            for head in heads:
                text = f"{non_head} {head}"

                ipa_text = transcribe_utterances(
                    [text],
                    backend="phonemizer",
                    language="en-us",
                    keep_word_boundaries=False
                )[0]

                surprisals = lm.token_score(
                    ipa_text,
                    bos_token=BOS,
                    prob=False,
                    surprisal=True,
                    bow_correction=False
                )[0]

                header_1 = "=" * 60
                header_2 = f"EXAMPLE {example_id} | GROUP {group_id}"
                line_1 = f"TEXT: {text}"
                line_2 = f"IPA : {ipa_text}"
                line_3 = f"NON-HEAD: {non_head}"
                line_4 = f"HEAD: {head}"
                line_5 = "SURPRISAL PER TOKEN (IPA-decoded token)"

                print(header_1)
                print(header_2)
                print(line_1)
                print(line_2)
                print(line_3)
                print(line_4)
                print(line_5)

                fout.write(header_1 + "\n")
                fout.write(header_2 + "\n")
                fout.write(line_1 + "\n")
                fout.write(line_2 + "\n")
                fout.write(line_3 + "\n")
                fout.write(line_4 + "\n")
                fout.write(line_5 + "\n")

                for tok_str, s in surprisals:
                    ipa_tok = token_to_ipa(tok_str, tok.byte_decoder)
                    out_line = f"{ipa_tok}\t{s:.7f}"
                    print(out_line)
                    fout.write(out_line + "\n")

                print()
                fout.write("\n")

                example_id += 1