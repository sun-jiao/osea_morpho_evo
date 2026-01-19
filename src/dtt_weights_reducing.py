import csv
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize # 引入归一化工具
from torchvision.models import resnet34

TARGET_VARIANCE = 0.80
MODEL_PATH = 'model20240824.pth'
BIRD_INFO_PATH = "bird_info.csv"
WEIGHTS_OUT_PATH = "all_weights.csv"
PCA_OUTPUT_PATH = "pca_weights.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

bird_info = []
with open(BIRD_INFO_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        bird_info.append(row)
num_species = len(bird_info)
print(f"Number of species: {num_species}")

model = resnet34(num_classes=11000)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # 增加 map_location 防止跨设备加载报错
model = model.to(device)
model.eval()

raw_weights = model.fc.weight.data[:num_species].detach().cpu().numpy()
print(f"Raw weights shape: {raw_weights.shape}")

print(f"Saving to {WEIGHTS_OUT_PATH}...")
with open(WEIGHTS_OUT_PATH, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(raw_weights)

# normalisation
print("Applying Pre-Normalization (L2)...")
weights_norm = normalize(raw_weights, norm='l2', axis=1)

print("Fitting PCA...")

pca_full = PCA()
pca_full.fit(weights_norm)

cumsum = np.cumsum(pca_full.explained_variance_ratio_)

d_target = np.argmax(cumsum >= TARGET_VARIANCE) + 1
d_90 = np.argmax(cumsum >= 0.90) + 1
d_95 = np.argmax(cumsum >= 0.95) + 1

print("-" * 30)
print(f"Dimensions needed for {TARGET_VARIANCE*100}% variance: {d_target}")
print(f"Dimensions needed for 90% variance: {d_90}")
print(f"Dimensions needed for 95% variance: {d_95}")
print("-" * 30)

pca_final = PCA(n_components=d_target)
weights_pca = pca_final.fit_transform(weights_norm)

# Re-normalisation after PCA
print("Applying Post-Normalization (L2)...")
weights_final = normalize(weights_pca, norm='l2', axis=1)

print(f"Final weights shape: {weights_final.shape}")

print(f"Saving to {PCA_OUTPUT_PATH}...")
with open(PCA_OUTPUT_PATH, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(weights_final)

print("Done.")