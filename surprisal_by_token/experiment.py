from minicons import scorer
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "NeTS-lab/babylm-mop-10m-gpt2"
text = "this monster is a rat eater"

BOS = True

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

surprisals = lm.token_score(
    text,
    bos_token=BOS,
    prob=False,
    surprisal=True,
    bow_correction=False
)[0]

print(f"{'TOKEN':<15} {'SURPRISAL':<10}")
print("-" * 30)
for tok, s in surprisals:
    print(f"{tok:<15} {s:.7f}")