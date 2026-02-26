from g2p import make_g2p
from minicons import scorer

print()

text = "this monster is a rat eater killer"
lines = [text]

g2p = make_g2p("eng", "eng-ipa")

ipa_list = []
for line in lines:
    out = g2p(line)
    ipa_text = str(out)  # keep as string

    # DO NOT replace spaces with WORD_BOUNDARY for phoneme-llama
    ipa_text = " ".join(ipa_text.split())  # normalize whitespace only

    ipa_list.append(ipa_text)

ipa_text = ipa_list[0]
print("IPA (folded):", ipa_text)

model_name = "bbunzeck/phoneme-llama"
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

surprisals = lm.token_score(
    ipa_text,
    bos_token=True,
    prob=False,
    surprisal=True,
    bow_correction=True
)[0]

print(f"{'TOKEN':<15} {'SURPRISAL':<10}")
print("-" * 30)
for tok, s in surprisals:
    print(f"{repr(tok):<15} {s:.7f}")