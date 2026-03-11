from collections import defaultdict
from minicons import scorer
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "NeTS-lab/babylm-mop-10m-gpt2"
text = "this monster is a rat eater"

BOS = True

# ---- previous strategy: load tokenizer/model manually ----
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    return_dict=True
)

lm = scorer.IncrementalLMScorer(
    model,
    tokenizer=tokenizer,
    device="cpu"
)

# ---- forced BOW correction ----
# Keep "Ġ" only if this tokenizer really uses it as the word-start marker.
bow_symbol = "Ġ"

lm.is_bow_tokenizer = True
lm.bow_symbol = bow_symbol

bow_subwords = defaultdict(bool)

for word, idx in lm.tokenizer.get_vocab().items():
    bow_subwords[idx] = len(word) > 0 and word[0] == bow_symbol

for idx in lm.tokenizer.get_added_vocab().values():
    bow_subwords[idx] = False

lm.bow_subwords = bow_subwords
lm.bow_subword_idx = [k for k, v in lm.bow_subwords.items() if v]

print("len(bow_subword_idx) =", len(lm.bow_subword_idx))
print("Forced BOW settings applied successfully.")
print("-" * 30)

# ---- token surprisals ----
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