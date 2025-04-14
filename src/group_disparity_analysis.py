import csv

import torch
from torchvision.models import resnet34

LEVEL = "family"

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
groups_and_vectors = []
for group in unique_groups:
    groups_and_vectors.append((group, [weights[index] for index, group0 in enumerate(groups_info) if group0 == group and index not in excluded]))

group_variances = []
for group, vectors in groups_and_vectors:
    vectors = torch.stack(vectors, dim=0)
    group_variance = vectors.var(dim=0, unbiased=False).mean()
    group_variances.append((group, group_variance.item(), vectors.size(0)))

with open(f'disparity_{LEVEL}.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(group_variances)