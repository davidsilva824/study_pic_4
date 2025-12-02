import torch
from collections import defaultdict
from minicons import scorer

# ------------------------------------------------
# 1. Create scorer as usual
# ------------------------------------------------
model_name = "bbunzeck/gpt-wee-small"
text = "this moster is a rat eater"
BOS = True

lm = scorer.IncrementalLMScorer(model_name, device="cpu")

# ------------------------------------------------
# 2. FORCE BOW SETTINGS (use 'Ġ' as space marker)
#    Keep the rest of the logic EXACTLY as in __init__
# ------------------------------------------------
tokenizer = lm.tokenizer
bow_symbol = "Ġ"   # force this as the BOW symbol

lm.is_bow_tokenizer = True
lm.bow_symbol = bow_symbol

bow_subwords = defaultdict(lambda: False)

# same pattern as in __init__: mark all vocab entries
for word, idx in tokenizer.get_vocab().items():
    if len(word) > 0 and word[0] == bow_symbol:
        bow_subwords[idx] = True
    else:
        bow_subwords[idx] = False

# handle added tokens with lstrip=True, same as in __init__
for idx, details in tokenizer.added_tokens_decoder.items():
    if getattr(details, "lstrip", False):
        bow_subwords[idx] = True

lm.bow_subwords = dict(bow_subwords)
lm.bow_subword_idx = [k for k, v in lm.bow_subwords.items() if v]

print("Forced is_bow_tokenizer:", lm.is_bow_tokenizer)
print("bow_symbol:", repr(lm.bow_symbol))
print("num bow_subword_idx:", len(lm.bow_subword_idx))

# ------------------------------------------------
# 3. Compare raw vs corrected surprisals
#    (compute_stats + bow_correction logic is UNCHANGED)
# ------------------------------------------------
raw = lm.token_score(
    text,
    bos_token=BOS,
    prob=False,
    surprisal=True,
    bow_correction=False,
    decode=False,
)[0]

corr = lm.token_score(
    text,
    bos_token=BOS,
    prob=False,
    surprisal=True,
    bow_correction=False,   # now uses forced BOW settings
    decode=False,
)[0]

print("\nRAW (bow_correction=False):")
for t, s in raw:
    print(f"{t:10s} {s:.3f}")


print("\nCORRECTED (forced Ġ-BOW):")
for tok, s in corr:
    print(f"{tok:10s} {s:.3f}")
