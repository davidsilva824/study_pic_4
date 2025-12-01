### This one is fully working. 

from minicons import scorer
print('True')

model_name = "BabyLM-community/babylm-baseline-100m-gpt2"
text = "this monster is a rat eater"

BOS = False
# load incremental (causal) LM
lm = scorer.IncrementalLMScorer(model_name, device="cpu")

# Get token-level surprisal using all specified parameters
surprisals = lm.token_score(
    text,
    bos_token=BOS,
    prob=False,
    surprisal=True,
    bow_correction=True
)[0]

# Print each token and its surprisal
for tok, s in surprisals:
    print(f"{tok}\t{s:.3f}")