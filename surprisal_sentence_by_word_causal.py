from minicons import scorer
from nltk.tokenize import TweetTokenizer

model_name = "BabyLM-community/babylm-baseline-100m-gpt2"
text = "rat catcher"

BOS = True
# load incremental (causal) LM
lm = scorer.IncrementalLMScorer(model_name, device="cpu")
# Get token-level surprisal using all specified parameters

word_tokenizer = TweetTokenizer().tokenize

surprisals = lm.word_score(
    text,
    prob=False,
    surprisal=True,
    bow_correction=True
)[0]

# Print each token and its surprisal
for tok, s in surprisals:
    print(f"{tok}\t{s:.3f}")