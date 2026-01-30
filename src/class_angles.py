import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

df_family_original = pd.read_csv('disparity_family.csv')
df_order_original = pd.read_csv('disparity_order.csv')

df_family_lt1 = df_family_original[df_family_original['size'] > 1]
df_order_lt1 = df_order_original[df_order_original['size'] > 1]

df_family_lt2 = df_family_original[df_family_original['size'] > 2]
df_order_lt2 = df_order_original[df_order_original['size'] > 2]

fig, axes = plt.subplots(2, 4, figsize=(24, 10))

def to_formal_scientific(number, precision=2):
    s = f"{number:.{precision}e}"
    base, exponent = s.split('e')
    exponent = int(exponent)
    return f"{base}×10^{exponent}"

# plot 1: size vs pairwise angle variance (order)
rho, p_value = stats.spearmanr(df_order_lt2['size'], df_order_lt2['pairwise angle variance'], nan_policy='omit')
sns.scatterplot(data=df_order_lt2, x='size', y='pairwise angle variance', ax=axes[0, 0])
axes[0, 0].set_title(f'(a) Order Size vs Angle Variance\n(ρ={rho:.2f}, p={p_value:.2f})')
axes[0, 0].set_xscale('log')

# plot 2: size vs pairwise angle variance (family)
rho, p_value = stats.spearmanr(df_family_lt2['size'], df_family_lt2['pairwise angle variance'], nan_policy='omit')
sns.scatterplot(data=df_family_lt2, x='size', y='pairwise angle variance', ax=axes[1, 0])
axes[1, 0].set_title(f'(b) Family Size vs Angle Variance\n(ρ={rho:.2f}, p={to_formal_scientific(p_value, 2)})')
axes[1, 0].set_xscale('log')

# plot 3: size vs pairwise angle variance (order)
rho, p_value = stats.spearmanr(df_order_lt1['size'], df_order_lt1['mean pairwise angle'], nan_policy='omit')
sns.scatterplot(data=df_order_lt1, x='size', y='mean pairwise angle', ax=axes[0, 1])
axes[0, 1].set_title(f'(c) Order Size vs Mean Angle\n(ρ={rho:.2f}, p={to_formal_scientific(p_value, 2)})')
axes[0, 1].set_xscale('log')

# plot 4: size vs pairwise angle variance (family)
rho, p_value = stats.spearmanr(df_family_lt1['size'], df_family_lt1['mean pairwise angle'], nan_policy='omit')
sns.scatterplot(data=df_family_lt1, x='size', y='mean pairwise angle', ax=axes[1, 1])
axes[1, 1].set_title(f'(d) Family Size vs Mean Angle\n(ρ={rho:.2f}, p={to_formal_scientific(p_value, 2)})')
axes[1, 1].set_xscale('log')

# plot 5: mean pairwise angle vs pairwise angle variance (order)
rho, p_value = stats.spearmanr(df_order_lt1['mean pairwise angle'], df_order_lt1['pairwise angle variance'], nan_policy='omit')
sns.scatterplot(data=df_order_lt1, x='mean pairwise angle', y='pairwise angle variance', ax=axes[0, 2])
axes[0, 2].set_title(f'(e) Mean Angle vs Angle Variance (Orders)\n(ρ={rho:.2f}, p={p_value:.3f})')

# plot 6: mean pairwise angle vs pairwise angle variance (family)
rho, p_value = stats.spearmanr(df_family_lt1['mean pairwise angle'], df_family_lt1['pairwise angle variance'], nan_policy='omit')
sns.scatterplot(data=df_family_lt1, x='mean pairwise angle', y='pairwise angle variance', ax=axes[1, 2])
axes[1, 2].set_title(f'(f) Mean Angle vs Angle Variance (Families)\n(ρ={rho:.2f}, p={to_formal_scientific(p_value, 2)})')

# plot 7: compare  order vs family distribution of mean angle

sns.kdeplot(df_family_lt1['mean pairwise angle'], label='Family', ax=axes[0, 3])
sns.kdeplot(df_order_lt1['mean pairwise angle'], label='Order', ax=axes[0, 3])
axes[0, 3].axvline(1.57066, color='r', linestyle='--', label='Global (All Birds)')
axes[0, 3].set_title('(g) Distribution of Mean Pairwise Angle')
axes[0, 3].legend()

# plot 8: Compare order vs family distributions of variance
sns.kdeplot(df_family_lt2['pairwise angle variance'], label='Family', ax=axes[1, 3])
sns.kdeplot(df_order_lt2['pairwise angle variance'], label='Order', ax=axes[1, 3])
axes[1, 3].axvline(0.00968, color='r', linestyle='--', label='Global (All Birds)')
axes[1, 3].set_title('(h) Distribution of Angle Variance')
axes[1, 3].legend()

plt.tight_layout()
# plt.show()
plt.savefig('../document/angles_analysis.pdf')

# Calculate correlations
corr_mean_var = df_family_lt1['mean pairwise angle'].corr(df_family_lt1['pairwise angle variance'], method='spearman')
