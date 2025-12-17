import pandas as pd
import scipy.stats as stats
import numpy as np

df = pd.read_csv("results_berent&pinker_phonemes_with_spaces_experiment_2.csv")

# Split by Pair and Number (singular/plural inferred from Non-Head)
pair1_singular_df = df[(df['Category'] == 'Pair 1') & (~df['Non-Head'].str.endswith('s'))].reset_index(drop=True)
pair1_plural_df   = df[(df['Category'] == 'Pair 1') & ( df['Non-Head'].str.endswith('s'))].reset_index(drop=True)
pair2_singular_df = df[(df['Category'] == 'Pair 2') & (~df['Non-Head'].str.endswith('s'))].reset_index(drop=True)
pair2_plural_df   = df[(df['Category'] == 'Pair 2') & ( df['Non-Head'].str.endswith('s'))].reset_index(drop=True)

# Make sure lengths match before subtracting
assert len(pair1_singular_df) == len(pair1_plural_df), "Pair 1 sing/plur length mismatch"
assert len(pair2_singular_df) == len(pair2_plural_df), "Pair 2 sing/plur length mismatch"
assert len(pair1_singular_df) == len(pair2_singular_df), "Pair 1 vs Pair 2 singular length mismatch"

# Pairwise differences
diff_pair1 = pair1_plural_df["Surprisal head"] - pair1_singular_df["Surprisal head"]
diff_pair2 = pair2_plural_df["Surprisal head"] - pair2_singular_df["Surprisal head"]
diff_sing  = pair1_singular_df["Surprisal head"] - pair2_singular_df["Surprisal head"]

# Function to compute 95% CI
def compute_95ci(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    if len(data) < 2 or sem == 0:
        return (np.nan, np.nan)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
    return ci

ci_pair1 = compute_95ci(diff_pair1)
ci_pair2 = compute_95ci(diff_pair2)
ci_sing  = compute_95ci(diff_sing)

# Create a list of dictionaries, one row per "Category"
combined_stats = [
    {
        "Category": "Pair 1",
        "Mean_Surprisal_Singular": pair1_singular_df["Surprisal head"].mean(),
        "Std_Surprisal_Singular":  pair1_singular_df["Surprisal head"].std(),
        "Mean_Surprisal_Plural":   pair1_plural_df["Surprisal head"].mean(),
        "Std_Surprisal_Plural":    pair1_plural_df["Surprisal head"].std(),
        "Mean_Difference":         diff_pair1.mean(),   # plural - singular
        "Std_Difference":          diff_pair1.std(),
        "CI_95_Lower":             ci_pair1[0],
        "CI_95_Upper":             ci_pair1[1]
    },
    {
        "Category": "Pair 2",
        "Mean_Surprisal_Singular": pair2_singular_df["Surprisal head"].mean(),
        "Std_Surprisal_Singular":  pair2_singular_df["Surprisal head"].std(),
        "Mean_Surprisal_Plural":   pair2_plural_df["Surprisal head"].mean(),
        "Std_Surprisal_Plural":    pair2_plural_df["Surprisal head"].std(),
        "Mean_Difference":         diff_pair2.mean(),   # plural - singular
        "Std_Difference":          diff_pair2.std(),
        "CI_95_Lower":             ci_pair2[0],
        "CI_95_Upper":             ci_pair2[1]
    },
    {
        # New row: difference between singulars (Pair 1 - Pair 2)
        "Category": "Pair1_minus_Pair2_Singular",
        "Mean_Surprisal_Singular": np.nan,
        "Std_Surprisal_Singular":  np.nan,
        "Mean_Surprisal_Plural":   np.nan,
        "Std_Surprisal_Plural":    np.nan,
        "Mean_Difference":         diff_sing.mean(),
        "Std_Difference":          diff_sing.std(),
        "CI_95_Lower":             ci_sing[0],
        "CI_95_Upper":             ci_sing[1]
    }
]

final_result_df = pd.DataFrame(combined_stats)

print("\n--- Combined Statistical Analysis ---")
print(final_result_df)

output_filename = "res_stats_berent_pinker_phonemes_with_spaces_experiment_2.csv"
final_result_df.to_csv(output_filename, index=False)

print(f"\nAll results have been saved to a single file: '{output_filename}'.")
