import os

import matplotlib.pyplot as plt
import numpy as np
# from scipy.stats import sem, t
from scipy.interpolate import interp1d

data_dir = 'output'
data_type = 'interval'
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
                    # time = np.linspace(0 - time_length, 0, 200)
                    time = np.linspace(0 - time_length, 0, len(nums))
                elif data_type == 'interval':
                    time = range(2 - len(nums), 1)
                    time = [0 - time_length, *time]
                times.append(time)

                # rand_num = np.random.randint(0, 50)
                # if rand_num == 0:  # random drawing, otherwise lines will be hard to distinguish
                plt.plot(time, nums, color='grey', linewidth=0.5, alpha=0.5)

# get the range of data for interpolating sampling
all_x = np.concatenate(times)
x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)

# interpolating points
x_common = np.linspace(x_min, x_max, 300)

interpolated_y = []

for x, y in zip(times, data):
    x = np.array(x)
    y = np.array(y)

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]

    if len(x_unique) < 2:
        continue

    try:
        f = interp1d(x_unique, y_unique, kind='linear', bounds_error=False, fill_value=(0.0, np.nan))
        y_interp = f(x_common)
        interpolated_y.append(y_interp)
    except Exception as e:
        continue

interpolated_y = np.array(interpolated_y)

mean_y = np.nanmean(interpolated_y, axis=0)

plt.plot(x_common, mean_y, color='blue', label='Mean Disparity (Average Y)')
plt.xlabel('Time (Myr)')
plt.ylabel('Disparity (Spherical Variance)')
plt.title('Disparity Through Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'dtt_plot.png'))
plt.show()
