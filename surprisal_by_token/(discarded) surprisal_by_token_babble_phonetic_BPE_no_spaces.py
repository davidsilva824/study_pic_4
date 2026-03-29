### This code is not to be used.
# Like in the text, the BPE without spaces does not separate the head and non-head clanly, invalidating the test for the PiC effect. 

# Because of a difference between the way the phonetic symbols from the G2P+ poackage,
# it has a special function to convert a  GPT-2 byte-level token string back to readable IPA text, after the surprisal is obtained for each token. 
# IMPORTANT note: the tokenizer of the model removes the spaces automatically. So it is indiferent if you put keep_word_boundaries=False or true.
# Forced BOW correction is working.

from g2p_plus import transcribe_utterances
from minicons import scorer
from transformers import AutoTokenizer
from collections import defaultdict

BOS = False


# 1) Input text
text = "this monster is a rat eater"
model_name = "phonemetransformers/GPT2-85M-BPE-PHON-SPACELESS"


# 2) Text -> IPA (g2p-plus)
ipa_text = transcribe_utterances(
    [text],
    backend="phonemizer",
    language="en-us",
    keep_word_boundaries=True
)[0]

print("TEXT:", text)
print("IPA :", ipa_text)
print()

lm = scorer.IncrementalLMScorer(model_name, device="cuda")


# 4) Get surprisal per ACTUAL tokenizer token
surprisals = lm.token_score(
    ipa_text,
    bos_token=BOS,
    prob=False,
    surprisal=True,
    bow_correction=False
)[0]

# 5) Load MATCHING tokenizer (slow version) so we can access byte_decoder
tok = AutoTokenizer.from_pretrained(
    "phonemetransformers/babble-tokenizers",
    subfolder="BABYLM-TOKENIZER-BPE-PHON",
    use_fast=False
)

def token_to_ipa(token_str, byte_decoder):
    """
    Converts a GPT-2 byte-level token string (weird symbols) back to readable IPA-ish text.
    Keeps Ġ as word-start marker in the output.
    """
    if token_str == "UTT_BOUNDARY":
        return "UTT_BOUNDARY"

    has_word_start = token_str.startswith("Ġ")
    core = token_str[1:] if has_word_start else token_str

    # Convert visible GPT-2 byte chars -> raw bytes -> UTF-8 text
    try:
        b = bytes([byte_decoder[c] for c in core])
        decoded = b.decode("utf-8")
    except Exception:
        # fallback: leave as-is if something unexpected appears
        decoded = core

    # Put Ġ back (as requested, glued to token)
    if has_word_start:
        return "Ġ" + decoded
    return decoded

print("SURPRISAL PER TOKEN (IPA-decoded token)")
for tok_str, s in surprisals:
    ipa_tok = token_to_ipa(tok_str, tok.byte_decoder)
    print(f"{ipa_tok}\t{s:.7f}")