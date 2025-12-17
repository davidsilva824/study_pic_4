### For causal models. 
# For models in which the bow correction does not work.

import torch
from collections import defaultdict
from minicons import scorer

model_name = "phonemetransformers/GPT2-85M-BPE-TXT"
text = "mazes decoder"
BOS = False

lm = scorer.IncrementalLMScorer(model_name, device="cpu")

bow_symbol = "Ġ"
lm.is_bow_tokenizer = True
lm.bow_symbol = bow_symbol

# Use a defaultdict that defaults to False for any unknown token
bow_subwords = defaultdict(bool)

# Mark standard vocabulary
for word, idx in lm.tokenizer.get_vocab().items():
    if len(word) > 0 and word[0] == bow_symbol:
        bow_subwords[idx] = True
    else:
        bow_subwords[idx] = False

# Mark special/added tokens (like BOS/EOS) as False to be safe
# (This ensures ID 633 doesn't crash the script)
for idx in lm.tokenizer.get_added_vocab().values():
    bow_subwords[idx] = False

lm.bow_subwords = bow_subwords
lm.bow_subword_idx = [k for k, v in lm.bow_subwords.items() if v]

print("Forced BOW settings applied successfully.")
print("-" * 30)

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