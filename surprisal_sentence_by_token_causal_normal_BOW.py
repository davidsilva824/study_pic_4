### This one is fully working. 

###this works only for models that have a specific saving space indiating their word boundry token. 

from minicons import scorer

model_name = "bbunzeck/gpt-wee-large"
text = "lifeguards patrol"

BOS = True
# load incremental (causal) LM
lm = scorer.IncrementalLMScorer(model_name, device="cpu")

# Get token-level surprisal using all specified parameters
surprisals = lm.token_score(
    text,
    bos_token=BOS,
    prob=False,
    surprisal=True,
    bow_correction=False
)[0]

# Print each token and its surprisal
for tok, s in surprisals:
    print(f"{tok}\t{s:.7f}")