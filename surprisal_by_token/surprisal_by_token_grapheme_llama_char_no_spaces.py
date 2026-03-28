### This code is complete.
# BOS = True
# IMPORTANT note: the tokenizer of the model does not remove the spaces automatically.
# Since this model does not use word separation at all, the BOW correction should be kept False. 
# With BOW correction true it also massively increases the surprisal of the last character. so avoid it. 

from minicons import scorer

model_name = "bbunzeck/grapheme-llama-no-whitespace"
text = "thismonsterisarateater"
BOS = True

lm = scorer.IncrementalLMScorer(model_name, device="cuda")

surprisals = lm.token_score(
    text,
    bos_token=BOS,
    prob=False,
    surprisal=True,
    bow_correction=False
)[0]

print(f"{'TOKEN':<15} {'SURPRISAL':<10}")
print("-" * 30)
for tok, s in surprisals:
    print(f"{tok:<15} {s:.7f}")