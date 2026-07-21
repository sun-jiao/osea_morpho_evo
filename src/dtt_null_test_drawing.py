import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# output_full_null, output_one_fifth_null, output_paca_exclude_top_20_null, output_paca_null, output_paca_top_5_null, output_paca_top_20_null
data_dir = 'output_paca_null'
data_type = 'interval' # 'num_slice'
output_image_name = '../document/dtt_null_test_combined.pdf'

empirical_data = []
null_data = []

print(f"Loading data from {data_dir}...")

files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
if not files:
    raise FileNotFoundError(f"No csv files found in {data_dir}")

for filename in files:
    filepath = os.path.join(data_dir, filename)

    current_file_lines = []
    with open(filepath) as f:
        # read the whole file
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            try:
                nums = list(map(float, parts))
                current_file_lines.append(nums)
            except ValueError:
                continue

    if not current_file_lines:
        continue

    for idx, lines in enumerate(current_file_lines):
        time_len = lines[0]
        vals = lines[1:]

        if data_type == 'num_slice':
            time = np.linspace(0 - time_len, 0, len(vals))
        else:
            # suppose interval=1.0, and the timeline is: -Length, -Length + 1, ..., -1, 0
            time = np.linspace(-time_len, 0, len(vals))

        # first line: empirical data, other lines: null data
        (empirical_data if idx == 0 else null_data).append((time, np.array(vals)))

print(f"Loaded {len(empirical_data)} empirical curves and {len(null_data)} null simulations.")

all_times = [x for x, y in empirical_data] + [x for x, y in null_data]
all_x_flat = np.concatenate(all_times)
x_min, x_max = np.nanmin(all_x_flat), np.nanmax(all_x_flat)

x_common = np.linspace(x_min, x_max, 500)


# Interpolation
def interpolate_to_common(data_list, common_x):
    interpolated_results = []
    for x, y in data_list:
        # Deduplication and sorting data
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[unique_idx]

        if len(x_unique) < 2:
            continue

        try:
            # Key: fill_value=(0.0, np.nan)
            # Solves the NaN problem when some trees have not yet started, setting it to 0
            f = interp1d(x_unique, y_unique, kind='linear', bounds_error=False, fill_value=(0.0, np.nan))
            y_interp = f(common_x)
            interpolated_results.append(y_interp)
        except Exception:
            continue

    return np.array(interpolated_results)


print("Interpolating Empirical data...")
emp_matrix = interpolate_to_common(empirical_data, x_common)

print("Interpolating Null data (this might take a moment)...")
null_matrix = interpolate_to_common(null_data, x_common)

# Normalisation
print("Normalizing curves to Relative Disparity (End point = 1.0)...")

# Normalising Empirical data
emp_finals = emp_matrix[:, -1]
emp_finals[emp_finals == 0] = 1e-9
emp_matrix = emp_matrix / emp_finals[:, None]

# Normalising Null Simulation
null_finals = null_matrix[:, -1]
null_finals[null_finals == 0] = 1e-9
null_matrix = null_matrix / null_finals[:, None]

# Empirical mean value
mean_empirical = np.nanmean(emp_matrix, axis=0)

# Null mean value and 95% CI (2.5% - 97.5%)
mean_null = np.nanmean(null_matrix, axis=0)
lower_null = np.nanpercentile(null_matrix, 2.5, axis=0)
upper_null = np.nanpercentile(null_matrix, 97.5, axis=0)

print("Plotting...")
plt.rcParams['font.size'] = 15
plt.figure(figsize=(15, 10))

# Null Range (grey shadow)
plt.fill_between(x_common, lower_null, upper_null, color='gray', alpha=0.3, label='95% Null Range')

# Null Mean (grey line)
plt.plot(x_common, mean_null, color='gray', linewidth=1.5, label='Mean Brownian Motion Simulation')

# Empirical Mean (blue line)
plt.plot(x_common, mean_empirical, color='blue', linewidth=2.5, label='Observed')

# K-Pg boundary
target_x = -66
line_color = 'red'
line_label_text = 'K-Pg boundary'

plt.vlines(x=target_x, ymin = 0, ymax = 0.9, color=line_color, linestyle='--', linewidth=2, alpha=0.8, zorder=5)

ax = plt.gca()

# Add text displayed on the line
plt.text(target_x, 0.5, line_label_text,
         rotation=90,
         color=line_color,
         horizontalalignment='center',
         verticalalignment='center',
         fontweight='regular',
         fontsize=20,
         bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=4.0),
         transform=ax.get_xaxis_transform(),
         zorder=6)

# =========================================

plt.xlabel('Time (Mya)')
plt.ylabel('Relative Morphological Disparity')
plt.title('Disparity Through Time: Empirical vs. Null Model (Aggregated)')
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xlim(x_min - 0.5, 0)
plt.ylim(0, 1.01)
plt.tight_layout()
plt.savefig(output_image_name, dpi=300)
print(f"Done! Plot saved to {output_image_name}")
# plt.show()