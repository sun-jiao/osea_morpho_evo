import csv

import torch
import umap
from sklearn.decomposition import PCA
from torchvision.models import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bird_info = []
with open("bird_info.csv", 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        bird_info.append(row)

num_species = len(bird_info)

model = resnet34(num_classes=11000)
weights_path = 'model20240824.pth'

model.load_state_dict(torch.load(weights_path))
model = model.to(device)
model.eval()

weights = model.fc.weight.to(device)[:num_species].detach().cpu().numpy()

pca_result = None

for n in range(1, weights.shape[1] - 1):
    pca = PCA(n_components=n)
    pca.fit(weights)
    total_explained = sum(pca.explained_variance_ratio_)
    print(f"{n}-dimensionality explained: {total_explained}")
    if 0.80 <= total_explained:
        best_n_components = n
        pca_result = pca.fit_transform(weights)
        break

if pca_result is None:
    pca_result = weights

with open(f'pca_weights.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(pca_result)
