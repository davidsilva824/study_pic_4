### For causal models. 
# Works only in more conventional models that provide a specific set of parameters about word separation.
# If the bow correction does not work, use 'surprisal_by_token_causal_changed_BOW.py'

from minicons import scorer

model_name = "NeTS-lab/babylm-mop-10m-gpt2"
text = "this monster is a rat eater"

BOS = True
# load incremental (causal) LM
lm = scorer.IncrementalLMScorer(model_name, device="cpu", trust_remote_code=True)


# Get token-level surprisal using all specified parameters
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