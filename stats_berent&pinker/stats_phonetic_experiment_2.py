import pandas as pd
import scipy.stats as stats
import numpy as np
from pathlib import Path


### Add the files for which you want to obtain the stats
file_list = [
    "results_berent&pinker/results_experiment_2_babble_phonetic_BPE_stories.csv",
    "results_berent&pinker/results_experiment_2_babble_phonetic_cha_stories.csv",
    "results_berent&pinker/results_experiment_2_babble_txt_BPE_stories.csv",
    "results_berent&pinker/results_experiment_2_grapheme_llama_stories.csv",
    "results_berent&pinker/results_experiment_2_phoneme_llama_stories.csv",
]


# Function to compute 95% CI (same logic as before)
def compute_95ci(data):
    data = pd.Series(data).dropna()
    n = len(data)

    if n < 2:
        return (np.nan, np.nan)

    mean = np.mean(data)
    sem = stats.sem(data, nan_policy="omit")

    if sem == 0 or np.isnan(sem):
        return (np.nan, np.nan)

    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
    return ci


# Helper: detect condition labels robustly (supports different capitalizations/separators)
def normalize_label(x):
    x = str(x).strip().lower()
    x = x.replace("-", "_").replace(" ", "_")
    while "__" in x:
        x = x.replace("__", "_")
    return x


def get_surprisal_column(df):
    # Tries common variants
    candidates = ["Surprisal head", "Surprisal.head", "surprisal head", "surprisal_head"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Could not find surprisal-head column. Checked common names like 'Surprisal head' and 'Surprisal.head'.")


def get_category_column(df):
    candidates = ["Category", "category"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Could not find category column ('Category').")


def summarize_condition(df, condition_name, category_col, surprisal_col):
    sub = df[df["_Category_norm"] == condition_name].reset_index(drop=True)
    values = sub[surprisal_col].dropna()

    ci_low, ci_high = compute_95ci(values)

    return {
        "Category": condition_name,
        "N": len(values),
        "Mean_Surprisal": values.mean() if len(values) > 0 else np.nan,
        "Std_Surprisal": values.std(ddof=1) if len(values) > 1 else np.nan,
        "CI_95_Lower": ci_low,
        "CI_95_Upper": ci_high
    }


def process_results_files(file_list):
    for filename in file_list:
        df = pd.read_csv(filename)

        category_col = get_category_column(df)
        surprisal_col = get_surprisal_column(df)

        # Normalize category labels to make filtering robust
        df["_Category_norm"] = df[category_col].apply(normalize_label)

        # Expected Experiment 2 conditions
        target_conditions = ["singular_1", "plural_1", "singular_2", "plural_2"]

        # Build combined stats table (similar "single output table" style)
        combined_stats = []
        for cond in target_conditions:
            combined_stats.append(
                summarize_condition(df, cond, category_col, surprisal_col)
            )

        final_result_df = pd.DataFrame(combined_stats)

        print("\n--- Combined Statistical Analysis ---")
        print(final_result_df)

        # Output name: replace "results_" with "res_stats_"
        p = Path(filename)
        output_filename = p.with_name(p.name.replace("results_", "res_stats_", 1))

        final_result_df.to_csv(output_filename, index=False)

        print(f"\nAll results have been saved to a single file: '{output_filename}'.")


process_results_files(file_list)