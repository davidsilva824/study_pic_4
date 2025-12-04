### this is especially for models where the correction is not being performed even when activated in the scorer. 

import torch
from collections import defaultdict
from minicons import scorer

model_name = "bbunzeck/gpt-wee-large"
text = "lifeguards patrol"
BOS = True

lm = scorer.IncrementalLMScorer(model_name, device="cpu")

#  FORCE BOW SETTINGS
# ------------------------------------------------
# We manually tell minicons: "Treat 'Ġ' as the space marker."
bow_symbol = "Ġ"
lm.is_bow_tokenizer = True
lm.bow_symbol = bow_symbol

# Use a defaultdict that defaults to False for any unknown token (fixes KeyError: 633)
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

# Assign it back to the model object
# CRITICAL: We keep it as a defaultdict so it never crashes on unseen IDs
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