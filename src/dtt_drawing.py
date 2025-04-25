import os

import matplotlib.pyplot as plt
import numpy as np
# from scipy.stats import sem, t
from scipy.interpolate import interp1d

data_dir = 'output_intervals'
data_type = 'interval' # 'num_slice'
time_interval = 1.0

data = []
times = []

plt.figure(figsize=(15, 10))

for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(data_dir, filename)
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                try:
                    nums = list(map(float, parts))
                except ValueError:
                    continue

                time_length = nums[0]
                nums = nums[1:]
                data.append(nums)

                if data_type == 'num_slice':
                    time = np.linspace(0 - time_length, 0, 200)
                elif data_type == 'interval':
                    time = range(2 - len(nums), 1)
                    time = [0 - time_length, *time]
                times.append(time)

                rand_num = np.random.randint(0, 50)
                if rand_num == 0:  # random drawing, otherwise lines will be hard to distinguish
                    plt.plot(time, nums, color='grey', linewidth=0.5, alpha=0.5)

# get the range of data for interpolating sampling
all_y = np.concatenate(data)
y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)

# interpolating points
y_common = np.linspace(y_min, y_max, 300)

# interpolated result
interpolated_x = []

for x, y in zip(times, data):
    y = np.array(y)
    x = np.array(x)

    if len(x) < 2 or len(y) < 2:
        continue

    sort_idx = np.argsort(y)
    y_sorted = y[sort_idx]
    x_sorted = x[sort_idx]

    y_unique, unique_idx = np.unique(y_sorted, return_index=True)
    x_unique = x_sorted[unique_idx]

    if len(y_unique) < 2:
        continue

    try:
        f = interp1d(y_unique, x_unique, bounds_error=False, fill_value=np.nan)
        x_interp = f(y_common)
        interpolated_x.append(x_interp)
    except Exception as e:
        continue

interpolated_x = np.array(interpolated_x)

mean_x = np.nanmean(interpolated_x, axis=0)
# ci = sem(interpolated_x, axis=0, nan_policy='omit') * t.ppf(0.975, df=interpolated_x.shape[0] - 1)

plt.plot(mean_x, y_common, color='blue', label='Mean X=f(Y)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Mean Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'mean_x.png'))
plt.show()
