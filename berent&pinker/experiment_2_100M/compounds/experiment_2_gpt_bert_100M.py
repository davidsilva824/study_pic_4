### This code is complete.
# Minimal adaptation: uses the SAME GPT-BERT method for all 3 models:
# - BOS = False
# - trust_remote_code = True
# - tuple-output wrapper so minicons can read .logits

import pandas as pd
from types import SimpleNamespace
from minicons import scorer

models = [
    "BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus",
    "BabyLM-community/babylm-baseline-100m-gpt-bert-mixed",
    "BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus"
]

BOS = False

stimuli_file = "berent&pinker/stimuli_compounds_experiment_2.csv"
stimuli_df = pd.read_csv(stimuli_file)

cat_labels = {
    "a": "singular_1",
    "b": "plural_1",
    "c": "singular_2",
    "d": "plural_2",
}

# --- FIX: wrap tuple outputs so minicons can read .logits ---
class _WrapOutputsWithLogits:
    def __init__(self, model):
        self._m = model

    def __call__(self, *args, **kwargs):
        out = self._m(*args, **kwargs)
        if hasattr(out, "logits"):
            return out
        if isinstance(out, tuple):
            return SimpleNamespace(logits=out[0])
        return out

    def __getattr__(self, name):
        return getattr(self._m, name)
# --- end fix ---


def process_pairs(lm, data):

    for _, row in stimuli_df.iterrows():
        suffix = str(row["story_id"]).strip().split("_")[-1]
        category_name = cat_labels[suffix]

        sentence = str(row["compound"]).strip()
        non_head, head = sentence.split(" ", 1)

        tok_scores = lm.token_score(
            sentence,
            bos_token=BOS,
            prob=False,
            surprisal=True,
            bow_correction=True
        )[0]

        tokens = [tok for tok, s, *_ in tok_scores]
        surprisal_values = [s for tok, s, *_ in tok_scores]

        # --- Original Print Block ---
        print(' '.join(f'{tok:>10}' for tok in tokens))
        print(' '.join(f'{s:>10.3f}' for s in surprisal_values))
        print(surprisal_values)

        non_n = 0
        reconstructed_word = ""

        cleaned_tokens = [tok.lstrip('Ġ ') for tok in tokens]

        for k in range(1, len(cleaned_tokens)):
            reconstructed_word += cleaned_tokens[k]
            non_n += 1
            if reconstructed_word == non_head:
                break

        total_real_tokens = len(tokens) - 1
        head_n = total_real_tokens - non_n

        surprisal_non_head = sum(surprisal_values[1: 1 + non_n])
        surprisal_head = sum(surprisal_values[1 + non_n: 1 + non_n + head_n])

        data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])

        # --- Original Sentence Print ---
        print(f"{sentence}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")


# --- MAIN EXECUTION ---
for model_name in models:
    print(f"\nLoading model: {model_name}...")
    lm = scorer.IncrementalLMScorer(model_name, device="cuda", trust_remote_code=True)

    # apply tuple-output wrapper
    lm.model = _WrapOutputsWithLogits(lm.model)

    data = []
    process_pairs(lm, data)

    # filename per model (same style as your experiment 1 script)
    if "causal" in model_name:
        output_file = "results_berent&pinker/results_experiment_2_gpt_bert_100M_causal_compounds.csv"
    elif "mixed" in model_name:
        output_file = "results_berent&pinker/results_experiment_2_gpt_bert_100M_mixed_compounds.csv"
    else:
        output_file = "results_berent&pinker/results_experiment_2_gpt_bert_100M_masked_compounds.csv"

    df = pd.DataFrame(
        data,
        columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"]
    )
    df.to_csv(output_file, index=False)

    print(f"\nresults in results_berent&pinker folder.\n")