### For MLM models. 
# Returns the surpisal of tokens.
# Uses 'within_word_l2r' metric, that corrects the surprisal of multi-token words. 

from minicons import scorer

model = "babylm/ltgbert-10m-2024"
text = "this man is heroic stranger"

from minicons import scorer

lm_masked = scorer.MaskedLMScorer(model, device="cpu", trust_remote_code=True)

scores = lm_masked.token_score(text, surprisal=True, base_two=True)[0]


for tok, s in scores:
    print(f"{tok}\t{s:.3f}")