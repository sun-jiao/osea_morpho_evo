import csv
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

LEVEL = "order"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_df = pd.read_csv("pca_weights.csv", header=None)
weights = torch.tensor(weights_df.values, dtype=torch.float16, device=device)
weights = F.normalize(weights, p=2, dim=1)

root_state_file =  open('root_state.pkl', 'rb')
root_state = pickle.load(root_state_file)
if not isinstance(root_state, np.ndarray):
    root_state = np.array(root_state)

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


def calculate_disparity(_vectors):
        # Spherical Variance
        mean_vec = _vectors.mean(dim=0)
        mean_vec_len = mean_vec.norm(p=2)
        sphere_variance = (1.0 - mean_vec_len).item()

        n = len(_vectors)

        if n < 2:
            mean_angle = 0
            var_angle = 0
        else:
            vectors_np = _vectors.detach().cpu().float().numpy()

            # cosine similarity
            cos_sim = np.dot(vectors_np, vectors_np.T)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            angles = np.arccos(cos_sim)
            iu = np.triu_indices(n, k=1)
            pairwise_angles = angles[iu]

            mean_angle = np.mean(pairwise_angles)
            var_angle = np.var(pairwise_angles)

        norm_root = np.linalg.norm(root_state)

        if mean_vec_len > 1e-9 and norm_root > 1e-9:
            mean_vec_np = mean_vec.detach().cpu().numpy()
            cos_sim_with_root = np.dot(mean_vec_np, root_state) / (mean_vec_len.item() * norm_root)
        else:
            cos_sim_with_root = 0.0

        return sphere_variance, mean_angle, var_angle, cos_sim_with_root



if LEVEL == "class":
    result = calculate_disparity(weights)

    print(f'Sphere variance: {result[0]}, Mean angle: {result[1]}, Variance angle: {result[2]}, Cosine Similarity with root: {result[3]}')
    exit()
elif LEVEL == 'order':
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

group_variances = []
for group, vectors in groups_and_vectors:
    results = calculate_disparity(vectors)
    group_variances.append((group, results[0], results[1], results[2], len(vectors), results[3]))

group_variances.sort(key=lambda x: x[1], reverse=True)

with open(f'disparity_{LEVEL}.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow([f'{LEVEL} name', 'sphere_variance', 'mean pairwise angle', 'pairwise angle variance', 'size', 'cos sim with root'])
    writer.writerows(group_variances)