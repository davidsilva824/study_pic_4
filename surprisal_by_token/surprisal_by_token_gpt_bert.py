### Deals with the format of the output logits. Seems to make sense and to be working fine. 
# Necessary to work with gpt-bert.
# In all the rest, these models work as normal causal models. 

import torch
from types import SimpleNamespace
from minicons import scorer

model_name = "BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus"
text = "protector"
BOS = False

lm = scorer.IncrementalLMScorer(model_name, device="cpu", trust_remote_code=True)

# --- MINIMAL FIX: wrap tuple outputs so minicons can read .logits ---
class _WrapOutputsWithLogits:
    def __init__(self, model):
        self._m = model

    def __call__(self, *args, **kwargs):
        out = self._m(*args, **kwargs)
        if hasattr(out, "logits"):
            return out
        if isinstance(out, tuple):
            return SimpleNamespace(logits=out[0])
        return out

    def __getattr__(self, name):
        return getattr(self._m, name)

lm.model = _WrapOutputsWithLogits(lm.model)
# --- end fix ---

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
