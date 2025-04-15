import csv

import torch
from sklearn.decomposition import PCA
from torchvision.models import resnet34

def weights_pca():
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

    weights = model.fc.weight.to(device)[:num_species]

    pca = PCA(n_components=10)
    return pca.fit_transform(weights.detach().cpu().numpy())

# with open(f'pca_weights.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerows(pca_result)
