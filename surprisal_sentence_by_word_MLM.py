from minicons import scorer

model_name = "bert-base-uncased"
text = "this moster is a rat eater"

def get_word_surprisal_from_wordpieces(model, text):
    """
    Word-level surprisal for masked LMs using Kauf & Ivanova (2023):
    - PLL_metric='within_word_l2r'
    - words reconstructed from WordPiece tokens (##-continuations)
    """
    # 1. Get per-subtoken surprisals with the correct PLL metric
    token_scores = model.token_score(
        text,
        surprisal=True,
        base_two=True,
        PLL_metric="within_word_l2r",
    )[0]

    word_results = []
    current_word = ""
    current_surprisal = 0.0

    # 2. Group wordpieces into words using '##' convention
    for tok, s in token_scores:
        # skip special tokens if they appear
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue

        if tok.startswith("##"):
            piece = tok[2:]
            current_word += piece
            current_surprisal += s
        else:
            # close previous word if any
            if current_word != "":
                word_results.append((current_word, current_surprisal))
            current_word = tok
            current_surprisal = s

    # 3. Add the last word
    if current_word != "":
        word_results.append((current_word, current_surprisal))

    return word_results


print(f"Loading model: {model_name}...")
lm_bert = scorer.MaskedLMScorer(model_name, device="cpu")

print(f"Analyzing sentence: '{text}'")
results = get_word_surprisal_from_wordpieces(lm_bert, text)

print("\n" + "=" * 30)
print(f"{'WORD':<15} {'SURPRISAL(bits)':<15}")
print("-" * 30)
for w, s in results:
    print(f"{w:<15} {s:.3f}")
print("=" * 30)
