### This code is complete. (Adapted for bbunzeck/phoneme-llama using g2p text->IPA conversion)

import pandas as pd
from minicons import scorer
from g2p import make_g2p

models = [
    "bbunzeck/phoneme-llama"
]

BOS = True

# --- Read compounds from CSV ---
stimuli_file = "berent&pinker/stimuli_compounds_experiment_2.csv"
stimuli_df = pd.read_csv(stimuli_file)

cat_labels = {
    "a": "singular_1",
    "b": "plural_1",
    "c": "singular_2",
    "d": "plural_2",
}

# --- g2p converters (rule-based + neural fallback) ---
g2p_rule = make_g2p("eng", "eng-ipa")
g2p_neural = make_g2p("eng", "eng-ipa", neural=True)


def _g2p_output_string(transducer, text):
    out = transducer(text)
    # Prefer output_string when available (more reliable than str(out))
    ipa = getattr(out, "output_string", str(out))
    return " ".join(str(ipa).split()).strip()


def phonemize_word(word):
    # First try rule-based
    ipa_rule = _g2p_output_string(g2p_rule, word)

    # If rule-based failed (empty output), fallback to neural
    if ipa_rule == "":
        ipa_neural = _g2p_output_string(g2p_neural, word)
        return ipa_neural
    return ipa_rule


def phonemize_sentence_with_fallback(sentence):
    words = sentence.split()
    ipa_words = [phonemize_word(w) for w in words]
    return " ".join(ipa_words)


def process_pairs(lm, pairs, data):
    for _, row in stimuli_df.iterrows():
        suffix = str(row["story_id"]).strip().split("_")[-1]
        category_name = cat_labels[suffix]

        sentence = str(row["compound"]).strip()
        non_head, head = sentence.split(" ", 1)

        # --- Text -> IPA with per-word fallback (rule-based first, neural only if needed) ---
        ipa_sentence = phonemize_sentence_with_fallback(sentence)

        tok_scores = lm.token_score(
            ipa_sentence,
            bos_token=BOS,
            prob=False,
            surprisal=True,
            bow_correction=False
        )[0]

        tokens = [tok for tok, s, *_ in tok_scores]
        surprisal_values = [s for tok, s, *_ in tok_scores]

        # --- Original Print Block ---
        print(' '.join(f'{tok:>10}' for tok in tokens))
        print(' '.join(f'{s:>10.3f}' for s in surprisal_values))
        print(surprisal_values)

        # --- Reconstruct non-head from IPA tokens ---
        non_head_ipa = phonemize_word(non_head)
        head_ipa = phonemize_word(head)

        # If first token is special BOS-like token, skip it; otherwise start at 0.
        start_idx = 1 if (len(tokens) > 0 and str(tokens[0]).startswith("<")) else 0

        non_n = 0
        reconstructed = ""

        for k in range(start_idx, len(tokens)):
            tok = str(tokens[k])

            # skip explicit special tokens if they appear
            if tok.startswith("<") and tok.endswith(">"):
                continue

            reconstructed += tok
            non_n += 1

            if reconstructed == non_head_ipa:
                break

        total_real_tokens = len(tokens) - start_idx
        head_n = total_real_tokens - non_n

        surprisal_non_head = sum(surprisal_values[start_idx : start_idx + non_n])
        surprisal_head = sum(surprisal_values[start_idx + non_n : start_idx + non_n + head_n])

        data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])

        print(f"TEXT: {sentence}")
        print(f"IPA : {ipa_sentence}")
        print(f"IPA split target -> non-head: {non_head_ipa} | head: {head_ipa}")
        print(f"Non-Head: {surprisal_non_head}, Head: {surprisal_head}")


# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda")

    data = []
    process_pairs(lm, None, data)

    output_file = "results_berent&pinker/results_experiment_2_phoneme_llama_compounds.csv"

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")