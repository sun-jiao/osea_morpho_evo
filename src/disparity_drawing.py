import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from scipy.stats import spearmanr, pearsonr, linregress

data = []

LEVEL = 'family'

with open(f"disparity_{LEVEL}.csv", 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row) if row[2] != '1' else None

names = [item[0] for item in data]
y_values = np.array([float(item[1]) for item in data])
y_log = np.array([math.log10(float(item[1])) for item in data])
x_values = np.array([int(item[2]) for item in data])
x_log = np.array([math.log10(int(item[2])) for item in data])

spearmanr_corr, p_value = spearmanr(x_values, y_values)

print(f"Spearman's rank correlation coefficient: {spearmanr_corr}")
print(f"p-value: {p_value}")

plt.figure(figsize=(18, 8))

plt.subplot(1, 2, 1)
plt.scatter(x_log, y_log, color='blue', s=50, label='taxon')

texts = []
for i, name in enumerate(names):
    texts.append(plt.text(x_log[i], y_log[i], name, fontsize=6))

adjust_text(texts)
plt.title(f"Diversity vs Disparity (birds in {LEVEL}-level)", fontsize=20)
plt.xlabel("species number (log10)", fontsize=16)
plt.ylabel("disparity (log10)", fontsize=16)

plt.grid(True, linestyle='--', alpha=0.6)

plt.legend()

# fitting log(Y) = log(a) + b * log(X)
slope, intercept, r_value, p_value, std_err = linregress(x_log, y_log)

b = slope
a = 10 ** intercept
print(f"Fitted model: Y = {a:.4f} * X^{b:.4f}")
print(f"Pearson R: {r_value:.4f}")
print(f"p-value: {p_value:.4e}")

# Drawing the image of function
x_fitted = np.linspace(min(x_values), max(x_values), 500)
y_fitted = a * x_fitted ** b

# Y~X fitted function
plt.subplot(1, 2, 2)
plt.scatter(x_values, y_values, color='blue', label='taxon')
plt.plot(x_fitted, y_fitted, color='red', label=f'Fit: Y = {a:.4f} * X^{b:.4f}')
plt.xlabel('Diversity (species richness) (X)', fontsize=16)
plt.ylabel('Morphological disparity (vector variances) (Y)', fontsize=16)
plt.title(f'Power Law Fit ({LEVEL}-level)', fontsize=20)
plt.legend()

plt.tight_layout()
plt.show()
