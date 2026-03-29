### This code is working.
# Uses g2p to convert the text to phonemes
# Bow correction is working normally, as can be seen by the zero value atributed to spaces.

from g2p import make_g2p
from minicons import scorer

print()

text = "this monster is a rat eater" # no spaces

model_name = "bbunzeck/phoneme-llama-no-whitespace"


lines = [text]

g2p = make_g2p("eng", "eng-ipa")

ipa_list = []
for line in lines:
    out = g2p(line)
    ipa_text = str(out)  # keep as string

    # DO NOT replace spaces with WORD_BOUNDARY for phoneme-llama
    ipa_text = "".join(ipa_text.split())  # normalize whitespace only

    ipa_list.append(ipa_text)

ipa_text = ipa_list[0]
print("IPA (folded):", ipa_text)


lm = scorer.IncrementalLMScorer(model_name, device="cuda")

surprisals = lm.token_score(
    ipa_text,
    bos_token=True,
    prob=False,
    surprisal=True,
    bow_correction=False
)[0]

print(f"{'TOKEN':<15} {'SURPRISAL':<10}")
print("-" * 30)
for tok, s in surprisals:
    print(f"{repr(tok):<15} {s:.7f}")