import csv

import pandas as pd
import torch
import torch.nn.functional as F

LEVEL = "family"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_df = pd.read_csv("pca_weights.csv", header=None)
weights = torch.tensor(weights_df.values, dtype=torch.float16, device=device)
weights = F.normalize(weights, p=2, dim=1)

bird_info = []
with open("bird_info.csv", 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        bird_info.append(row)

num_species = len(bird_info)

excluded = []
with open("excluded_species.csv", 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        excluded.append(row)
excluded = [bird[0] for bird in excluded]

if LEVEL == 'order':
    level_index = 4
elif LEVEL == 'family':
    level_index = 5
else:
    raise ValueError('level must be either "order" or "family"')

groups_info = [bird[level_index] for bird in bird_info]
unique_groups = list(set(groups_info))
groups_and_vectors = []
for group in unique_groups:
    indices = [
        i for i, g in enumerate(groups_info)
        if g == group and i not in excluded and i < len(weights)
    ]

    if len(indices) > 0:
        group_vectors = weights[indices]
        groups_and_vectors.append((group, group_vectors))

# Spherical Variance = 1 - length(Mean_Vector)
group_variances = []
for group, vectors in groups_and_vectors:
    mean_vec = vectors.mean(dim=0)
    mean_vec_len = mean_vec.norm(p=2)
    sphere_variance = (1.0 - mean_vec_len).item()
    group_variances.append((group, sphere_variance, vectors.size(0)))

group_variances.sort(key=lambda x: x[1], reverse=True)

with open(f'disparity_{LEVEL}.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(group_variances)