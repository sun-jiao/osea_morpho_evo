import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import curve_fit

data = []

LEVEL = 'order'
INPUT_FILE = f"disparity_{LEVEL}.csv"

# content of data:
# row[0]: name of taxa; row[1]: disparity (sphere_variance), 2: mean pairwise angle,
# 3: pairwise angle variance; row[4] size (species richness)
with open(INPUT_FILE, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == f"{LEVEL} name": continue

        data.append(row) if row[4] != '1' else None

names = [item[0] for item in data]
diversity = np.array([int(item[4]) for item in data])
disparity = np.array([float(item[1]) for item in data])

spearmanr_corr, p_value = spearmanr(diversity, disparity)

print(f"Spearman's rank correlation coefficient: {spearmanr_corr}")
print(f"p-value: {p_value}")


# Model A: Power Law
# f(x) = 1 - x^b
# generally we use a * x^b,
# but because spherical variance is definitely == 0 when x == 1
# the coefficient is unneeded
def model_power(x, b):
    return 1 - x ** b


# Model B: Stretched Exponential
# f(x) = 1 − exp(−λ*(x−1)^β)
def model_stretched_exp(x, lam, beta):
    val = np.maximum(x - 1, 0)
    return 1 - np.exp(-lam * (val ** beta))


# Model C: Hill Equation
# f(x) = ((x − 1)^n) / (k + (x − 1)^n)
def model_hill(x, k, n):
    val = np.maximum(x - 1, 0)
    return 1 - k / (k + (val ** n))


# Model D: Logarithmic Rational
# f(x) = ln(x) / (k + ln(x))
def model_log_rational(x, k):
    val = np.log(np.maximum(x, 1e-9))
    return 1 - k / (k + val)


def fit_and_evaluate(model_func, x_data, y_data, model_name, p0=None, bounds=(-np.inf, np.inf)):
    try:
        popt, pcov = curve_fit(model_func, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
        y_pred = model_func(x_data, *popt)

        # RSS (Residual Sum of Squares)
        rss = np.sum((y_data - y_pred) ** 2)
        n = len(y_data)
        k_param = len(popt)

        # AIC (Akaike Information Criterion)
        # AIC = n * ln(RSS/n) + 2k
        aic = n * np.log(rss / n) + 2 * k_param

        # R^2
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2 = 1 - (rss / ss_tot)

        return {
            'name': model_name,
            'aic': aic,
            'r2': r2,
            'params': popt,
            'y_pred': y_pred
        }
    except Exception as e:
        return {'name': model_name, 'error': str(e)}


print(f"--- Model Comparison for {LEVEL}-level Data ---")

# A. Power Law
res_power = fit_and_evaluate(
    model_power, diversity, disparity,
    model_name="Power Law",
    p0=[-0.5],
    bounds=(-np.inf, 0),
)

# B. Stretched Exponential
res_strexp = fit_and_evaluate(
    model_stretched_exp, diversity, disparity,
    model_name="Stretched Exp",
    p0=[0.1, 0.5],
    bounds=([0, 0], [np.inf, np.inf]),
)

# C. Hill Equation
res_hill = fit_and_evaluate(
    model_hill, diversity, disparity,
    model_name="Hill Eq",
    p0=[10, 1],
    bounds=([0, 0], [np.inf, np.inf]),
)

# D. Logarithmic Rational
res_log = fit_and_evaluate(
    model_log_rational, diversity, disparity,
    model_name="Log Rational",
    p0=[1],
    bounds=(0, np.inf),
)

results = [res_power, res_strexp, res_hill, res_log]
# sorting by AIC
results.sort(key=lambda x: x.get('aic', float('inf')))

print(f"{'Model':<30} | {'AIC':<25} | {'R^2':<10} | {'Params'}")
print("-" * 80)
for res in results:
    if 'error' in res:
        print(f"{res['name']:<30} | {'Fit Failed':<25} | {'-':<10} | {res['error']}")
    else:
        param_str = ", ".join([f"{p:.4f}" for p in res['params']])
        print(f"{res['name']:<30} | {res['aic']:<25.2f} | {res['r2']:<10.4f} | {param_str}")

# get the best model
best_res = results[0]
print(f"\nSelected Best Model: {best_res['name']} (AIC={best_res['aic']:.2f})")

# residual = observed - predicted
observed_vals = disparity
predicted_vals = best_res['y_pred']
residuals = observed_vals -  predicted_vals

# {'name': scientific name, 'x': diversity, 'y': disparity, 'residual': residual}
points_data = []
for i, name in enumerate(names):
    points_data.append({
        'name': name,
        'x': diversity[i],
        'y': disparity[i],
        'residual': residuals[i]
    })

points_data.sort(key=lambda item: item['residual'])

bottom_10 = points_data[:5]
top_10 = points_data[-5:]

print("\n--- Top 5 High Residuals ---")
for p in reversed(top_10):
    print(f"{p['name']}: {p['residual']:.4f}")

print("\n--- Top 5 Low Residuals ---")
for p in bottom_10:
    print(f"{p['name']}: {p['residual']:.4f}")


plt.figure(figsize=(12, 8))

plt.scatter(diversity, disparity, color='gray', alpha=0.5, label='Observed Mean Vector Length')

for p in bottom_10 + top_10:
    plt.text(p['x'], p['y'], p['name'], fontsize=9, color='black',
             ha='left', va='bottom', alpha=0.8)

x_plot = np.logspace(np.log10(min(diversity)), np.log10(max(diversity)), 1000)

if 'y_pred' in res_power:
    plt.plot(x_plot, model_power(x_plot, *res_power['params']), 'r-', linewidth=2,
             label=f"Power Law (AIC={res_power['aic']:.0f})")

if 'y_pred' in res_strexp:
    plt.plot(x_plot, model_stretched_exp(x_plot, *res_strexp['params']), 'g--', linewidth=2,
             label=f"Stretched Exponential (AIC={res_strexp['aic']:.0f})")

if 'y_pred' in res_hill:
    plt.plot(x_plot, model_hill(x_plot, *res_hill['params']), 'b-.', linewidth=2,
             label=f"Hill Equation (AIC={res_hill['aic']:.0f})")

if 'y_pred' in res_log:
    plt.plot(x_plot, model_log_rational(x_plot, *res_log['params']), 'm:', linewidth=3,
             label=f"Logarithmic Rational (AIC={res_log['aic']:.0f})")

# both x and y are in log scale.
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Diversity (Species Richness, log scale)", fontsize=14)
plt.ylabel("Spherical Variance (1 - ||R||, log scale)", fontsize=14)
plt.title(f"Model Comparison: Decay of Morphological Cohesion ({LEVEL}-level)", fontsize=16)
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(f"model_comparison_{LEVEL}.png", dpi=300)
plt.show()