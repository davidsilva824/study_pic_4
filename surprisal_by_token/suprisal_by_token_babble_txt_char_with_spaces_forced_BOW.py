### This code seems to be working, but it wouldnt hurt a final reverification. 

# BOS = False
# Forced BOW settings block, for separator token "W"
# Splitting head and non-head based on the token W.
# IMPORTANT note: The BOW correction is working. This can be observe in the file 'suprisal_by_token_babble_txt_char_with_spaces.py'.
# With the correction the suprisal of the token 'W' becomes zero. 


from minicons import scorer
from collections import defaultdict

model_name = "phonemetransformers/GPT2-85M-CHAR-TXT"
text = "this monster is a rat eater"
BOS = False

lm = scorer.IncrementalLMScorer(model_name, device="cpu", trust_remote_code=True)

# Force minicons to treat "W" as the word-boundary marker
bow_symbol = "W"
lm.is_bow_tokenizer = True
lm.bow_symbol = bow_symbol

bow_subwords = defaultdict(bool)

vocab = lm.tokenizer.get_vocab()
bow_id = vocab.get(bow_symbol, None)

for _, idx in vocab.items():
    bow_subwords[idx] = False

for idx in lm.tokenizer.get_added_vocab().values():
    bow_subwords[idx] = False

if bow_id is not None:
    bow_subwords[bow_id] = True

lm.bow_subwords = bow_subwords
lm.bow_subword_idx = [int(bow_id)] if bow_id is not None else []

print("bow_symbol =", bow_symbol)
print("bow_id =", bow_id)
print("len(bow_subword_idx) =", len(lm.bow_subword_idx))
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