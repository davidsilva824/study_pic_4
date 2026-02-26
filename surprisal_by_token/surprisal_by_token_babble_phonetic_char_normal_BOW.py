### Surprisal for fonetic models. 
### Important note. It does not need a changed version, because it already sparates the 

from g2p_plus import transcribe_utterances
from minicons import scorer

print()

text = "this monster is a rat eater"
lines = [text]

# Converts text to phonemes
ipa_list = transcribe_utterances(
    lines,
    backend="phonemizer",
    language="en-us",                      
    keep_word_boundaries=True
)

ipa_text = ipa_list[0]
print("IPA (folded):", ipa_text)

model_name = "phonemetransformers/GPT2-85M-CHAR-PHON"
lm = scorer.IncrementalLMScorer(model_name, device="cuda") 

surprisals = lm.token_score(
    ipa_text,
    bos_token=False,
    prob=False,
    surprisal=True,
    bow_correction=True
)[0]

print(f"{'TOKEN':<15} {'SURPRISAL':<10}")
print("-" * 30)
for tok, s in surprisals:
    print(f"{tok:<15} {s:.7f}")
