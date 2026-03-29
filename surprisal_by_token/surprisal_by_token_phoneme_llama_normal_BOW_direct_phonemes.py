### This code is complete. 
# This code should be used to directly input phonetic symbols in the phoneme llama model. 
# Used in order to understand which symbols it accepts and it doesn't accept.
# Bow correction is working normally, as can be seen by the zero value atributed to spaces. 

from minicons import scorer

print()

# Write the phonemes directly here.
ipa_text = "leɪbɜ˞ɜ˞z leɪbɜ˞ɜ˞z"

print("IPA (manual):", ipa_text)

model_name = "bbunzeck/phoneme-llama"
lm = scorer.IncrementalLMScorer(model_name, device="cuda")

surprisals = lm.token_score(
    ipa_text,
    bos_token=True,
    prob=False,
    surprisal=True,
    bow_correction=True
)[0]

print(f"{'TOKEN':<15} {'SURPRISAL':<10}")
print("-" * 30)
for tok, s in surprisals:
    print(f"{repr(tok):<15} {s:.7f}")