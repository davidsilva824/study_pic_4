from minicons import scorer

model = "babylm/ltgbert-10m-2024"
text = "this moster is a rat eater"

from minicons import scorer

# 1. Load the Masked Model
lm_masked = scorer.MaskedLMScorer(model, device="cpu", trust_remote_code=True)

# 2. Extract Token Surprisals
# Note: For basic token extraction, we don't need special metrics yet.
# This uses the default strategy: Mask one token, predict it, move to next.
scores = lm_masked.token_score(text, surprisal=True, base_two=True)[0]


for tok, s in scores:
    print(f"{tok}\t{s:.3f}")