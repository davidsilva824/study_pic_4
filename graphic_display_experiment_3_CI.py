import matplotlib.pyplot as plt
import numpy as np


models = ['GPT-wee (S)', 'GPT-wee (M)', 'GPT-wee (L)', 
          'GPT 2 (10M)', 'GPT 2 (100M)']

# Mean Difference in Surprisal
regular_means   = np.array([0.6160, 0.5711, 0.6698, 1.4068, 1.6823])
irregular_means = np.array([0.1007, 0.0647, 0.0274, 0.3649, 0.9181])

# 95% CI (lower, upper)
regular_ci_lower   = np.array([0.4062, 0.1569, 0.1776, 0.9672, 1.2092])
regular_ci_upper   = np.array([0.8259, 0.9853, 1.1621, 1.8465, 2.1554])

irregular_ci_lower = np.array([-0.0601, -0.2096, -0.4065, -0.0512, 0.4319])
irregular_ci_upper = np.array([ 0.2615,  0.3389,  0.4613,  0.7811, 1.4042])


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
plt.savefig('chart_2_updated.png')

print("Updated chart saved as 'chart_2_updated.png'")