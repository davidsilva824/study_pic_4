import pandas as pd
import scipy.stats as stats
import numpy as np


df = pd.read_csv("study_gpt_wee_small_experiment_3.csv")

# Correctly filter the DataFrame into four distinct, properly indexed groups
regular_singular_df = df[df['Category'] == 'Regular Singular'].reset_index(drop=True)
regular_plural_df = df[df['Category'] == 'Regular Plural'].reset_index(drop=True)
irregular_singular_df = df[df['Category'] == 'Irregular Singular'].reset_index(drop=True)
irregular_plural_df = df[df['Category'] == 'Irregular Plural'].reset_index(drop=True)



# Calculate the pairwise differences
diff_reg = regular_plural_df["Surprisal head"] - regular_singular_df["Surprisal head"]
diff_irreg = irregular_plural_df["Surprisal head"] - irregular_singular_df["Surprisal head"]

# Function to compute 95% CI
def compute_95ci(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    # Check if there's enough data to compute CI
    if len(data) < 2 or sem == 0:
        return (np.nan, np.nan)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
    return ci

ci_reg = compute_95ci(diff_reg)
ci_irreg = compute_95ci(diff_irreg)


# Create a list of dictionaries, with each dictionary representing a row
combined_stats = [
    {
        "Category": "Regular",
        "Mean_Surprisal_Singular": regular_singular_df["Surprisal head"].mean(),
        "Std_Surprisal_Singular": regular_singular_df["Surprisal head"].std(),
        "Mean_Surprisal_Plural": regular_plural_df["Surprisal head"].mean(),
        "Std_Surprisal_Plural": regular_plural_df["Surprisal head"].std(),
        "Mean_Difference": diff_reg.mean(),
        "Std_Difference": diff_reg.std(),
        "CI_95_Lower": ci_reg[0],
        "CI_95_Upper": ci_reg[1]
    },
    {
        "Category": "Irregular",
        "Mean_Surprisal_Singular": irregular_singular_df["Surprisal head"].mean(),
        "Std_Surprisal_Singular": irregular_singular_df["Surprisal head"].std(),
        "Mean_Surprisal_Plural": irregular_plural_df["Surprisal head"].mean(),
        "Std_Surprisal_Plural": irregular_plural_df["Surprisal head"].std(),
        "Mean_Difference": diff_irreg.mean(),
        "Std_Difference": diff_irreg.std(),
        "CI_95_Lower": ci_irreg[0],
        "CI_95_Upper": ci_irreg[1]
    }
]

final_result_df = pd.DataFrame(combined_stats)

print("\n--- Combined Statistical Analysis ---")
print(final_result_df)


output_filename = "stats_gpt_wee_small_experiment_3.csv"
final_result_df.to_csv(output_filename, index=False)

print(f"\nAll results have been saved to a single file: '{output_filename}'.")