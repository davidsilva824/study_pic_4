import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
# --- 1. Data Setup ---
models = ['GPT-wee (S)', 'GPT-wee (M)', 'GPT-wee (L)',
          'BabyLlama (10M)', 'BabyLlama (100M)', 'Babble', 'GPT-J']

regular_means   = np.array([0.2757, 0.5768, 1.1663, 1.2629, 0.7816, 1.9651, 1.8121])
irregular_means = np.array([0.1479, -0.1498, 1.0937, 0.0885, 0.4884, 1.2598, 0.6665])

# 95% CI (lower, upper)
regular_ci_lower   = np.array([-0.4769, -0.4991,  0.3685,  0.5898,  0.2121,  0.9808,  1.0874])
regular_ci_upper   = np.array([ 1.0283,  1.6528,  1.9641,  1.9360,  1.3511,  2.9495,  2.5368])

irregular_ci_lower = np.array([-0.2982, -0.9060,  0.2784, -0.8999, -0.1340, -0.4446, -0.3955])
irregular_ci_upper = np.array([ 0.5940,  0.6063,  1.9089,  1.0770,  1.1109,  2.9643,  1.7285])


# Convert to asymmetric error bars
regular_err = np.vstack([
    regular_means - regular_ci_lower,
    regular_ci_upper - regular_means
])

irregular_err = np.vstack([
    irregular_means - irregular_ci_lower,
    irregular_ci_upper - irregular_means
])

# --- 2. Plotting the Chart ---
x_positions = np.arange(len(models))

bar_width = 0.40

fig, ax = plt.subplots(figsize=(12, 8))

ax.bar(x_positions - bar_width/2, regular_means, bar_width,
       yerr=regular_err, capsize=4,
       label='Regular plurals', color='#8B0000')

ax.bar(x_positions + bar_width/2, irregular_means, bar_width,
       yerr=irregular_err, capsize=4,
       label='Irregular plurals', color='#9400D3')


# --- 3. Styling and Labels ---
ax.set_title('Comparison of Mean Surprisal Difference Across Models', fontsize=22, pad=20)

ax.set_ylabel('Mean Difference in Surprisal', fontsize=18)

ax.set_xticks(x_positions)
ax.set_xticklabels(models, rotation=20, ha="right", fontsize=20)

ax.tick_params(axis='y', labelsize=14)

ax.legend(fontsize=14)

ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_axisbelow(True)

# --- 4. Save the File ---
plt.tight_layout()
plt.savefig('chart_2_CI.png')

print("Chart with CIs saved as 'chart_2.png'")
