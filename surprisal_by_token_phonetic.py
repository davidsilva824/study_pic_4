### Surprisal for fonetic models. 

from g2p_plus import transcribe_utterances
from minicons import scorer

print()

text = "lifeguards patrol"
lines = [text]

# Converts text to phonemes
ipa_list = transcribe_utterances(
    lines,
    backend="phonemizer",
    language="en-us",                      
    keep_word_boundaries=True,
    allow_possibly_faulty_word_boundaries=True
)

ipa_text = ipa_list[0]
print("IPA (folded):", ipa_text)

model_name = "phonemetransformers/GPT2-85M-CHAR-PHON"
lm = scorer.IncrementalLMScorer(model_name, device="cpu") #change cpu to cuda for GPU usage. 

surprisals = lm.token_score(
    ipa_text,
    bos_token=False,
    prob=False,
    surprisal=True,
    bow_correction=False
)[0]

print("\nSURPRISAL PER TOKEN:")
for tok, s in surprisals:
    print(f"{tok}\t{s:.7f}") #to adjust decimal units

print()
