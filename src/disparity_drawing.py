import csv
import math

import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.stats import spearmanr, pearsonr

data = []

LEVEL = 'family'

with open(f"disparity_{LEVEL}.csv", 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row) if row[2] != '1' else None

names = [item[0] for item in data]
y_values = [float(item[1]) for item in data]
x_values = [int(item[2]) for item in data]
x_values_log = [math.log10(int(item[2])) for item in data]

spearmanr_corr, p_value = spearmanr(x_values, y_values)

print(f"Spearman's rank correlation coefficient: {spearmanr_corr}")
print(f"p-value: {p_value}")

plt.figure(figsize=(16, 12))
plt.scatter(x_values_log, y_values, color='blue', s=50)

texts = []
for i, name in enumerate(names):
    texts.append(plt.text(x_values_log[i], y_values[i], name, fontsize=10))

adjust_text(texts)
plt.title(f"Diversity vs Disparity (birds in {LEVEL}-level)", fontsize=20)
plt.xlabel("species number (log10)", fontsize=16)
plt.ylabel("disparity ()", fontsize=16)

plt.grid(True, linestyle='--', alpha=0.6)

plt.show()