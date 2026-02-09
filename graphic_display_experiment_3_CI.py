import matplotlib.pyplot as plt
import numpy as np


models = ['Babble TEXT-BPE (spaces)', 
          'Babble TEXT-single-char (spaces)', 
          'Babble TEXT-single-char (no spaces)',
          'Grapheme-LLaMA-single-char (spaces)']

# Mean Difference in Surprisal
regular_means   = np.array([1.0579, 1.5147, -0.8099, 0.9539])
irregular_means = np.array([0.1448, 1.0429, -0.6212, 0.1562])

# 95% CI (lower, upper)
regular_ci_lower   = np.array([0.3518, 1.0130, -1.3724, 0.5325])
regular_ci_upper   = np.array([1.7640, 2.0163, -0.2473, 1.3752])

irregular_ci_lower = np.array([-0.5185, 0.5474, -1.2277, -0.2387])
irregular_ci_upper = np.array([ 0.8081, 1.5384, -0.0147, 0.5512])


# Convert to asymmetric error bars (distance from mean)
regular_err = np.vstack([
    regular_means - regular_ci_lower,
    regular_ci_upper - regular_means
])

irregular_err = np.vstack([
    irregular_means - irregular_ci_lower,
    irregular_ci_upper - irregular_means
])

# Plotting the Chart
x_positions = np.arange(len(models))

bar_width = 0.40

fig, ax = plt.subplots(figsize=(12, 8))

# Regular Plurals Bar
ax.bar(x_positions - bar_width/2, regular_means, bar_width,
       yerr=regular_err, capsize=4,
       label='Regular plurals', color='#8B0000')

# Irregular Plurals Bar
ax.bar(x_positions + bar_width/2, irregular_means, bar_width,
       yerr=irregular_err, capsize=4,
       label='Irregular plurals', color='#9400D3')


# Styling and Labels
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

# Saving file
plt.tight_layout()
plt.savefig('chart_babble_models.png')

print("Updated chart saved as 'chart_babble_models.png'")
