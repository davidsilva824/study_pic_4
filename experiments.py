from minicons import scorer

model_name = "babylm/ltgbert-10m-2024"
text = "this monster is a rat eater"

BOS = True

# load incremental (causal) LM
lm = scorer.MaskedLMScorer(model_name, device="cpu", trust_remote_code=True)

print(dir(lm))