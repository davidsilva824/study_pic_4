### For causal models. 
# Works only in more conventional models that provide a specific set of parameters about word separation.
# If the bow correction does not work, use 'surprisal_by_token_causal_changed_BOW.py'

from minicons import scorer

model_name = "phonemetransformers/GPT2-85M-CHAR-TXT-SPACELESS"
text = "this monster is a rat eater"

BOS = False
# load incremental (causal) LM
lm = scorer.IncrementalLMScorer(model_name, device="cpu", trust_remote_code=True)

lm.model.config.return_dict = True
lm.model.config.torchscript = False

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
    print(f"{tok}\t{s:.7f}")