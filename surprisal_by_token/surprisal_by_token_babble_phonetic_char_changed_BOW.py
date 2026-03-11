### Surprisal for fonetic models. 
### Important note. It does not need a changed version, because it already sparates the 

from g2p_plus import transcribe_utterances
from minicons import scorer
from collections import defaultdict


print()

text = "this monster is a rat eater"
lines = [text]

# Converts text to phonemes
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



#### the correction

bow_symbol = "WORD_BOUNDARY"
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

print("len(bow_subword_idx) =", len(lm.bow_subword_idx)) 

print("Forced BOW settings applied successfully.")
print("-" * 30)

################################################################

surprisals = lm.token_score(
    ipa_text,
    bos_token=False,
    prob=False,
    surprisal=True,
    bow_correction=True
)[0]

print(f"{'TOKEN':<15} {'SURPRISAL':<10}")
print("-" * 30)
for tok, s in surprisals:
    print(f"{tok:<15} {s:.7f}")
