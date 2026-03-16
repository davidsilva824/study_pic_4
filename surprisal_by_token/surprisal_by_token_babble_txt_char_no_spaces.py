### This code seems to be correct, but a reverification wouldn't hurt.  
# BOS = False
# IMPORTANT note: the tokenizer of the model removes the spaces automatically. No need to remove them before inputing the model.
# Since this model does not use word separation at all, the BOW correction should be kept False. 

from minicons import scorer

model_name = "phonemetransformers/GPT2-85M-CHAR-TXT-SPACELESS"
text = "this monster is a rat eater"
BOS = False

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