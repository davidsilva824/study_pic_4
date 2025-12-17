import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load stats from the two experiments ---
df2 = pd.read_csv("res_stats_berent_pinker_phonemes_no_spaces_experiment_2.csv")
df3 = pd.read_csv("res_stats_berent_pinker_phonemes_no_spaces_experiment_3.csv")

# Helper: grab Mean_Difference and CI for a given Category from a stats file
def extract_diff_ci(df, category_name):
    row = df[df["Category"] == category_name].iloc[0]
    mean = row["Mean_Difference"]
    ci_low = row["CI_95_Lower"]
    ci_high = row["CI_95_Upper"]
    return mean, ci_low, ci_high

# Models on the x-axis = the two experiments
models = ["Experiment 2", "Experiment 3"]

# Regular-sounding (Pair 1)
p1_mean_2, p1_low_2, p1_high_2 = extract_diff_ci(df2, "Pair 1")
p1_mean_3, p1_low_3, p1_high_3 = extract_diff_ci(df3, "Pair 1")

regular_means = np.array([p1_mean_2, p1_mean_3])
regular_ci_lower = np.array([p1_low_2,  p1_low_3])
regular_ci_upper = np.array([p1_high_2, p1_high_3])

# Control (Pair 2)
p2_mean_2, p2_low_2, p2_high_2 = extract_diff_ci(df2, "Pair 2")
p2_mean_3, p2_low_3, p2_high_3 = extract_diff_ci(df3, "Pair 2")

irregular_means = np.array([p2_mean_2, p2_mean_3])
irregular_ci_lower = np.array([p2_low_2,  p2_low_3])
irregular_ci_upper = np.array([p2_high_2, p2_high_3])

# Convert to asymmetric error bars (distance from mean)
regular_err = np.vstack([
    regular_means - regular_ci_lower,
    regular_ci_upper - regular_means
])

irregular_err = np.vstack([
    irregular_means - irregular_ci_lower,
    irregular_ci_upper - irregular_means
])

# --- Plotting (same structure as your example) ---
x_positions = np.arange(len(models))
bar_width = 0.40

fig, ax = plt.subplots(figsize=(10, 6))

# Pair 1 (regular-sounding)
ax.bar(
    x_positions - bar_width/2,
    regular_means,
    bar_width,
    yerr=regular_err,
    capsize=4,
    label="Pair 1 (regular-sounding)"
)

# Pair 2 (control)
ax.bar(
    x_positions + bar_width/2,
    irregular_means,
    bar_width,
    yerr=irregular_err,
    capsize=4,
    label="Pair 2 (control)"
)

ax.set_title("Plural penalty on head surprisal (phoneme model)", fontsize=16, pad=10)
ax.set_ylabel("Mean Difference in Surprisal (plural − singular)", fontsize=14)

ax.set_xticks(x_positions)
ax.set_xticklabels(models, fontsize=12)

ax.tick_params(axis="y", labelsize=12)
ax.legend(fontsize=12)

ax.yaxis.grid(True, linestyle="--", which="major", alpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("graphic_display_berent&pinker_2.png")
print("Chart saved as 'chart_berent_pinker_phonemes_spaces.png'")
