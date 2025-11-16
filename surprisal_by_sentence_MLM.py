from minicons import scorer

model_name = "babylm/ltgbert-10m-2024"
text = "this monster is a ducks feeder"

BOS = True

# load incremental (causal) LM
lm = scorer.MaskedLMScorer(model_name, device="cpu", trust_remote_code=True)

# Get token-level surprisal using all specified parameters
surprisals = lm.token_score(
    text,
    prob=False,
    surprisal=True
)[0]

# Print each token and its surprisal
for tok, s in surprisals:
    print(f"{tok}\t{s:.3f}")