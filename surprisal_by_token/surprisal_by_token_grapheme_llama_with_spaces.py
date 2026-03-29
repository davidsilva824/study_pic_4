# This code is complete. 
# BOS = True
# Normal Bow correction is working, as can be seen by the 0 surprisal atributed to spaces, with bow_correction=True.


from minicons import scorer

model_name = "bbunzeck/grapheme-llama"
text = "this monster is a rat eater"
BOS = True

lm = scorer.IncrementalLMScorer(model_name, device="cuda")

surprisals = lm.token_score(
    text,
    bos_token=BOS,
    prob=False,
    surprisal=True,
    bow_correction=True
)[0]

print(f"{'TOKEN':<15} {'SURPRISAL':<10}")
print("-" * 30)
for tok, s in surprisals:
    print(f"{tok:<15} {s:.7f}")