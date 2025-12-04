### This one is fully working. 

from g2p_plus import transcribe_utterances
from minicons import scorer

# 1. Convert to phonemes with the SAME config used for IPA-BabyLM
text = "lifeguards patrol"
lines = [text]

ipa_list = transcribe_utterances(
    lines,
    backend="phonemizer",
    language="en-us",                      # <- match training (en-us)
    keep_word_boundaries=True,
    allow_possibly_faulty_word_boundaries=True
    # NOTE: do NOT set uncorrected=True, so folding stays ON
)

ipa_text = ipa_list[0]
print("IPA (folded):", ipa_text)

# 2. Load phoneme model
model_name = "phonemetransformers/GPT2-85M-CHAR-PHON"
lm = scorer.IncrementalLMScorer(model_name, device="cpu")

# 3. Get surprisal values (no bow-correction needed)
surprisals = lm.token_score(
    ipa_text,
    bos_token=False,
    prob=False,
    surprisal=True,
    bow_correction=False
)[0]

# 4. Print results
print("\nSURPRISAL PER TOKEN:")
for tok, s in surprisals:
    print(f"{tok}\t{s:.3f}")
