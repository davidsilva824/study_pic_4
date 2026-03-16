### This code seems to be working, but it wouldnt hurt a final reverification. 
# It uses athe conversion from text to phonemes g2plus. 
# IMPORTANT note: The Forced BOW correction is working. 
# With the correction the suprisal of the token 'WORD_BOUNDARY' becomes zero.


from g2p_plus import transcribe_utterances
from minicons import scorer
from collections import defaultdict

print()

text = "this monster is a rat eater"
lines = [text]

ipa_list = transcribe_utterances(
    lines,
    backend="phonemizer",
    language="en-us",
    keep_word_boundaries=True
)

ipa_text = ipa_list[0]
print("IPA (folded):", ipa_text)

model_name = "phonemetransformers/GPT2-85M-CHAR-PHON"
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

# ---- forced BOW correction with WORD_BOUNDARY ----
bow_symbol = "WORD_BOUNDARY"
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
# -----------------------------------------------

surprisals = lm.token_score(
    ipa_text,
    bos_token=False,
    prob=False,
    surprisal=True,
    bow_correction=True
)[0]

print(f"{'TOKEN':<20} {'SURPRISAL':<10}")
print("-" * 35)
for tok, s in surprisals:
    print(f"{tok:<20} {s:.7f}")