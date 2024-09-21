import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

df = pd.read_pickle('body_word_proportions.pkl')

def create_boxplot_and_test(df, category, output_file, alpha):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=category, y='proportion', data=df, palette="Greys")
    plt.title(f'Proportions of "body words" by {category.capitalize()}')
    plt.savefig(f'{output_file}_boxplot.png')
    plt.close()

    unique_categories = df[category].unique()
    if len(unique_categories) == 2:
        group1 = df[df[category] == unique_categories[0]]['proportion']
        group2 = df[df[category] == unique_categories[1]]['proportion']
        t_stat, p_value = ttest_ind(group1, group2)
        with open(f'{output_file}_ttest.txt', 'w') as f:
            f.write(f'T-test results for {category.capitalize()}:\n')
            f.write(f'T-statistic: {t_stat}\n')
            f.write(f'P-value: {p_value}\n')
        return p_value, p_value < alpha
    else:
        with open(f'{output_file}_ttest.txt', 'w') as f:
            f.write(f'T-test not performed for {category.capitalize()} due to more than two categories.\n')
        return None, False

categories = ['gender', 'genre', 'speciality']
alpha = 0.05 / len(categories)  # Bonferroni correction
p_values = {}
significant_results = {}

for category in categories:
    p_value, is_significant = create_boxplot_and_test(df, category, category, alpha)
    if p_value is not None:
        p_values[category] = p_value
        significant_results[category] = is_significant

fig, axes = plt.subplots(1, 3, figsize=(16, 9))
for ax, category in zip(axes, categories):
    sns.violinplot(x=category, y='proportion', data=df, palette="Greys", ax=ax)
    ax.set_title(f'Proportions of "body words" by {category.capitalize()}')
    if category in significant_results and significant_results[category]:
        color = 'red'
    else:
        color = 'black'

    ax.text(0.5, 0.95, f'p-value: {p_values[category]:.4f}\n0.05 sig. level (after Bonferroni): 0.0167', ha='center', va='center', transform=ax.transAxes,
            fontsize=12, color=color, bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.savefig('combined_boxplots_16_9.png')
plt.show()

print("Box plots and significance testing results saved to the working directory.")