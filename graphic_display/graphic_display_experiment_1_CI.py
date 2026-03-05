import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Files you uploaded (stats CSVs) ---
files = [
    "results_experiment_1_10M/res_stats_experiment_1_babyLlama_2_10M.csv",
    "results_experiment_1_10M/res_stats_experiment_1_babyLlama_10M.csv",
    "results_experiment_1_10M/res_stats_experiment_1_gpt_2_10M.csv",
    "results_experiment_1_10M/res_stats_experiment_1_gpt_bert_10M_causal.csv",
    "results_experiment_1_10M/res_stats_experiment_1_gpt_bert_10M_masked.csv",
    "results_experiment_1_10M/res_stats_experiment_1_gpt_bert_10M_mixed.csv",
    "results_experiment_1_10M/res_stats_experiment_1_gpt_wee_large.csv",
    "results_experiment_1_10M/res_stats_experiment_1_gpt_wee_medium.csv",
    "results_experiment_1_10M/res_stats_experiment_1_gpt_wee_small.csv",
    "results_experiment_1_10M/res_stats_experiment_1_MOEP.csv",
    "results_experiment_1_10M/res_stats_experiment_1_OPT_10M.csv",
    "results_experiment_1_10M/res_stats_experiment_1_ZLATA.csv",
]

labels = [
    "BabyLLaMA 2-10M",
    "BabyLLaMA 10M",
    "GPT-2 10M",
    "GPT-BERT causal 10M",
    "GPT-BERT masked 10M",
    "GPT-BERT mixed 10M",
    "GPT-wee large",
    "GPT-wee medium",
    "GPT-wee small",
    "MOEP",
    "OPT 10M",
    "ZLATA",
]

def read_one(path):
    df = pd.read_csv(path)
    reg = df[df["Category"].str.lower() == "regular"].iloc[0]
    irr = df[df["Category"].str.lower() == "irregular"].iloc[0]
    return (
        float(reg["Mean_Difference"]), float(reg["CI_95_Lower"]), float(reg["CI_95_Upper"]),
        float(irr["Mean_Difference"]), float(irr["CI_95_Lower"]), float(irr["CI_95_Upper"]),
    )

regular_means, regular_low, regular_up = [], [], []
irregular_means, irregular_low, irregular_up = [], [], []

for f in files:
    r_m, r_l, r_u, i_m, i_l, i_u = read_one(f)
    regular_means.append(r_m); regular_low.append(r_l); regular_up.append(r_u)
    irregular_means.append(i_m); irregular_low.append(i_l); irregular_up.append(i_u)

regular_means = np.array(regular_means)
regular_low   = np.array(regular_low)
regular_up    = np.array(regular_up)

irregular_means = np.array(irregular_means)
irregular_low   = np.array(irregular_low)
irregular_up    = np.array(irregular_up)

regular_err = np.vstack([regular_means - regular_low, regular_up - regular_means])
irregular_err = np.vstack([irregular_means - irregular_low, irregular_up - irregular_means])

def plot_block(idx, out_png, title):
    block_labels = [labels[i] for i in idx]
    x = np.arange(len(idx))
    bar_width = 0.40

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.bar(x - bar_width/2, regular_means[idx], bar_width,
           yerr=regular_err[:, idx], capsize=4,
           label="Regular plurals", color="#8B0000")

    ax.bar(x + bar_width/2, irregular_means[idx], bar_width,
           yerr=irregular_err[:, idx], capsize=4,
           label="Irregular plurals", color="#9400D3")

    ax.set_title(title, fontsize=22, pad=20)
    ax.set_ylabel("Mean Difference in Surprisal", fontsize=18)

    ax.set_xticks(x)
    ax.set_xticklabels(block_labels, rotation=20, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)

    ax.legend(fontsize=14)
    ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)

# split into two plots (6 + 6)
plot_block(np.arange(0, 6), "chart_models_part1.png", "Mean Surprisal Difference (Models 1–6)")
plot_block(np.arange(6, 12), "chart_models_part2.png", "Mean Surprisal Difference (Models 7–12)")

print("Saved: chart_models_part1.png and chart_models_part2.png")