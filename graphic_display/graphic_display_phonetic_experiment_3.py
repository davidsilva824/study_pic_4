import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Folder where your res_stats CSV files are saved
stats_folder = Path(r"results_berent&pinker")

# Pick all summary files created by your previous script
file_list = sorted(stats_folder.glob("res_stats_experiment_3_*_stories.csv"))

# Order of conditions on the x-axis
condition_order = ["singular_1", "plural_1", "singular_2", "plural_2"]

# Nice labels for display
label_map = {
    "singular_1": "Singular 1",
    "plural_1": "Plural 1",
    "singular_2": "Singular 2",
    "plural_2": "Plural 2"
}

for file_path in file_list:
    df = pd.read_csv(file_path)

    # Keep only expected rows and enforce order
    df["Category"] = df["Category"].astype(str).str.strip().str.lower()
    df = df[df["Category"].isin(condition_order)].copy()
    df["Category"] = pd.Categorical(df["Category"], categories=condition_order, ordered=True)
    df = df.sort_values("Category")

    # Means and CI
    means = df["Mean_Surprisal"].to_numpy(dtype=float)
    ci_low = df["CI_95_Lower"].to_numpy(dtype=float)
    ci_high = df["CI_95_Upper"].to_numpy(dtype=float)

    # Asymmetric error bars
    yerr = np.vstack([
        means - ci_low,
        ci_high - means
    ])

    x = np.arange(len(df))

    # Title from filename
    model_name = file_path.stem.replace("res_stats_experiment_2_", "").replace("_stories", "")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.bar(
        x,
        means,
        yerr=yerr,
        capsize=4
    )

    ax.set_title(f"Head surprisal by condition ({model_name})", fontsize=14, pad=10)
    ax.set_ylabel("Mean head surprisal", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[c] for c in df["Category"]], fontsize=11)

    ax.yaxis.grid(True, linestyle="--", which="major", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()

    out_png = file_path.with_name(file_path.stem.replace("res_stats_", "graphic_") + ".png")
    plt.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_png}")