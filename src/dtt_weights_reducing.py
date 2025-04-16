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

pca = PCA(n_components=50)
pca_result = pca.fit_transform(weights)

reducer = umap.UMAP(n_components=5, random_state=42)

reduced_weights = reducer.fit_transform(pca_result)

with open(f'reduced_weights.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(reduced_weights)
