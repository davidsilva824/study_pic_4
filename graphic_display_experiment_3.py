import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
models = ['GPT-wee (S)', 'GPT-wee (M)', 'GPT-wee (L)']

regular_means   = np.array([0.1464, 0.1113, 0.2459])
irregular_means = np.array([-0.0181, -0.1709, -0.5020])

# --- 2. Plotting the Chart ---
x_positions = np.arange(len(models))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(12, 8)) # Increased figure height slightly to accommodate larger text

ax.bar(x_positions - bar_width/2, regular_means, bar_width,
       label='Regular plurals', color='#8B0000')

ax.bar(x_positions + bar_width/2, irregular_means, bar_width,
       label='Irregular plurals', color='#9400D3')


# --- 3. Styling and Labels (with even larger font sizes) ---
ax.set_title('Comparison of Mean Surprisal Difference Across Models', fontsize=22, pad=20)

ax.set_ylabel('Mean Difference in Surprisal', fontsize=18)

ax.set_xticks(x_positions)
ax.set_xticklabels(models, rotation=20, ha="right", fontsize=14) # Increased rotation slightly

ax.legend(fontsize=14)

ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_axisbelow(True)


# --- 4. Save the File ---
plt.tight_layout()
plt.savefig('chart_2.png')

print("Chart with even larger text saved as 'chart_even_larger_text.png'")