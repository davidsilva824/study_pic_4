### This code is complete.
# BOS = False
# IMPORTANT note: the tokenizer of the model removes the spaces automatically. So it is indiferent if you put keep_word_boundaries=False or true.
# Since this model does not use word separation at all, the BOW correction should be kept False. 

from g2p_plus import transcribe_utterances
from minicons import scorer

print()

BOS = False

model_name = "phonemetransformers/GPT2-85M-CHAR-PHON-SPACELESS"


text = "this monster is a rat eater"
lines = [text]

ipa_list = transcribe_utterances(
    lines,
    backend="phonemizer",
    language="en-us",
    keep_word_boundaries=False
)

ipa_text = ipa_list[0]
print("IPA (folded):", ipa_text)

lm = scorer.IncrementalLMScorer(model_name, device="cuda")


surprisals = lm.token_score(
    ipa_text,
    bos_token=BOS,
    prob=False,
    surprisal=True,
    bow_correction=True
)[0]

print(f"{'TOKEN':<20} {'SURPRISAL':<10}")
print("-" * 35)
for tok, s in surprisals:
    print(f"{tok:<20} {s:.7f}")