import os

import matplotlib.pyplot as plt
import numpy as np
# from scipy.stats import sem, t
from scipy.interpolate import interp1d

data_dir = 'output_null'
filename = 'disparity_through_time-HackettStage1Full_1.tre-tree4-1745573018.csv'
data_type = 'interval' # 'num_slice'
time_interval = 1.0

data = []

plt.figure(figsize=(15, 10))

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

        if data_type == 'num_slice':
            time = np.linspace(0 - time_length, 0, 200)
        elif data_type == 'interval':
            time = range(2 - len(nums), 1)
            time = [0 - time_length, *time]

        data.append([(t, v) for t, v in zip(time, nums)])

        plt.plot(time, nums, color='grey', linewidth=0.1, alpha=0.5)


reconstructed_values = data[0]
all_null_simulations = data[1:]


# Assuming all lists (reconstructed and simulations) have the same time points
# and are ordered consistently. Extract times and reconstructed values.
times = np.array([item[0] for item in reconstructed_values])
reconstructed_vals = np.array([item[1] for item in reconstructed_values])

# Convert each simulation list to a numpy array of values
sim_values_only = [np.array([item[1] for item in sim]) for sim in all_null_simulations]

# Stack these arrays. This assumes all simulations have the same length.
# If simulations have different lengths, a more complex grouping by time is needed.
sim_values_stacked = np.vstack(sim_values_only)

# Calculate 2.5th and 97.5th percentiles along the first axis (across simulations)
# This gives the lower and upper bounds for each time point.
lower_bound = np.percentile(sim_values_stacked, 2.5, axis=0)
upper_bound = np.percentile(sim_values_stacked, 97.5, axis=0)

# Plot the reconstructed values
plt.plot(times, reconstructed_vals, label='Reconstructed Values', color='blue', linestyle='-')

# Plot the 95% range as a shaded area
plt.fill_between(times, lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Null Range')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Reconstructed Values vs. 95% Null Simulation Range')
plt.legend()
plt.grid(True)
plt.show()

