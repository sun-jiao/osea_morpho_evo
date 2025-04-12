import csv

import torch
from torchvision.models import resnet34

LEVEL = "order"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 11000
model = resnet34(num_classes=num_classes)
weights_path = 'model20240824.pth'

model.load_state_dict(torch.load(weights_path))
model = model.to(device)
model.eval()

weights = model.fc.weight.data.to(torch.float16).to(device)
# weights = list(torch.unbind(weights, dim=0))

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
    level_index = 3
elif LEVEL == 'family':
    level_index = 4
else:
    raise ValueError('level must be either "order" or "family"')

groups_info = [bird[level_index] for bird in bird_info]
unique_groups = list(set(groups_info))
groups_size = {}
groups_and_vectors = []
for group in unique_groups:
    vectors = torch.stack([weights[index] for index, group0 in enumerate(groups_info) if group0 == group and index not in excluded], dim=0)
    groups_and_vectors.append((group, vectors))
    groups_size[group] = vectors.size(0)

group_variances = {}
for group, vectors in groups_and_vectors:
    group_variance = vectors.var(dim=0, unbiased=False).mean()
    group_variances[group] = group_variance.item()

mean_pairwise_distances = {}
for group, vectors in groups_and_vectors:
    vectors = vectors.float()
    if vectors.size(0) > 1:
        dists = torch.cdist(vectors, vectors, p=2)
        triu_indices = torch.triu_indices(vectors.size(0), vectors.size(0), offset=1)
        mean_dist = dists[triu_indices[0], triu_indices[1]].mean()
        mean_pairwise_distances[group] = mean_dist.item()
    else:
        mean_pairwise_distances[group] = 0.0

mean_dist_to_centroid = {}
for group, vectors in groups_and_vectors:
    if vectors.size(0) > 1:
        centroid = vectors.mean(dim=0)
        dists = torch.norm(vectors - centroid, dim=1)
        mean_dist = dists.mean()
        mean_dist_to_centroid[group] = mean_dist.item()
    else:
        mean_dist_to_centroid[group] = 0.0

morpho_range = {}
for group, vectors in groups_and_vectors:
    if vectors.size(0) > 1:
        range_per_dim = vectors.max(dim=0).values - vectors.min(dim=0).values
        morpho_range[group] = range_per_dim.mean().item()
    else:
        morpho_range[group] = 0.0

with open(f'disparity_{LEVEL}.csv', 'w') as file:
    writer = csv.writer(file)
    for group in unique_groups:
        writer.writerow([group, groups_size[group], group_variances[group], mean_pairwise_distances[group], mean_dist_to_centroid[group], morpho_range[group]])