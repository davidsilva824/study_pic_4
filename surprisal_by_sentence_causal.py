from minicons import scorer

model_name = "EleutherAI/gpt-neo-2.7B"
text = "this monster is a ducks feeder"

BOS = True

# load incremental (causal) LM
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

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