import csv
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize # 引入归一化工具
from torchvision.models import resnet34
from disparity_through_time import read_csv, create_trait_mapping
from excluded_species import is_excluded_species, load_excluded_species

TARGET_VARIANCE = 0.80
MODEL_PATH = 'model20240824.pth'
BIRD_INFO_PATH = "bird_info.csv"
WEIGHTS_OUT_PATH = "avian_timetree_all_weights.csv"
PCA_OUTPUT_PATH = "avian_timetree_pca_weights.csv"
PCA_OUTPUT_PATH_95 = "avian_timetree_pca_weights_95.csv"
PCA_OUTPUT_PATH_100 = "avian_timetree_pca_weights_100.csv"
ONE_FIFTH_OF_SPECIES_PATH = "avian_timetree_one_fifth_species_pca_weights.csv"

name_match_file = "avian_timetree_name_match.csv"
new_name_match_file = "avian_timetree_name_match_trimmed.csv"

# the relationship between labels in the tree and indexes of vectors
name_match = read_csv(name_match_file)

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
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

raw_weights = model.fc.weight.data[:num_species].detach().cpu().numpy()
print(f"Raw weights shape: {raw_weights.shape}")

# normalisation
print("Applying Pre-Normalization (L2)...")
weights_norm = normalize(raw_weights, norm='l2', axis=1)

trait_mapping = create_trait_mapping(name_match, weights_norm)

excluded_indices, excluded_names = load_excluded_species()
analysis_trait_mapping = {
    label: vector
    for label, vector in trait_mapping.items()
    if not is_excluded_species(excluded_indices, excluded_names, name=label)
}
print(f"Excluded {len(trait_mapping) - len(analysis_trait_mapping)} species from DTT/PCA analyses.")

mapped_weights = np.array(list(analysis_trait_mapping.values()))
mapped_labels = list(analysis_trait_mapping.keys())

with open(new_name_match_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i, val in enumerate(mapped_labels):
        writer.writerow([val, "empty", i])

print(f"Saving to {WEIGHTS_OUT_PATH}...")
with open(WEIGHTS_OUT_PATH, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(mapped_weights)

print("Fitting PCA...")

pca_full = PCA()
pca_full.fit(mapped_weights)

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
weights_pca = pca_final.fit_transform(mapped_weights)

# Re-normalisation after PCA
print("Applying Post-Normalization (L2)...")
weights_final = normalize(weights_pca, norm='l2', axis=1)

print(f"Final weights shape: {weights_final.shape}")

print(f"Saving to {PCA_OUTPUT_PATH}...")
with open(PCA_OUTPUT_PATH, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(weights_final)

dims = round(len(mapped_labels) / 5)
pca_one_fifth = PCA(n_components=dims)
weights_one_fifth = pca_one_fifth.fit_transform(mapped_weights)

weights_one_fifth_final = normalize(weights_one_fifth, norm='l2', axis=1)

print(f"Final weights shape for one-fifth of species: {weights_one_fifth_final.shape}")

print(f"Saving to {ONE_FIFTH_OF_SPECIES_PATH}...")
with open(ONE_FIFTH_OF_SPECIES_PATH, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(weights_one_fifth_final)

pca_final_95 = PCA(n_components=d_95)
weights_pca_95 = pca_final_95.fit_transform(mapped_weights)
weights_pca_95_final = normalize(weights_pca_95, norm='l2', axis=1)
print(f"Final weights shape for 95% variance: {weights_pca_95_final.shape}")

print(f"Saving to {PCA_OUTPUT_PATH_95}...")
with open(PCA_OUTPUT_PATH_95, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(weights_pca_95_final)

pca_final_100 = PCA(n_components=len(cumsum))
weights_pca_100 = pca_final_100.fit_transform(mapped_weights)
weights_pca_100_final = normalize(weights_pca_100, norm='l2', axis=1)
print(f"Final weights shape for 100% variance: {weights_pca_100_final.shape}")

print(f"Saving to {PCA_OUTPUT_PATH_100}...")
with open(PCA_OUTPUT_PATH_100, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(weights_pca_100_final)

# Similarity clustering intentionally retains all species.  Preserve its PCA
# basis as a separate fit to the complete mapped dataset, while the DTT files
# above are based only on analysis_trait_mapping.
clustering_weights = np.array(list(trait_mapping.values()))
clustering_pca = PCA(n_components=min(clustering_weights.shape))
clustering_pca.fit(clustering_weights)
projection_matrix = clustering_pca.components_.T
data_mean = clustering_pca.mean_

np.save('pca_final_100_projection_matrix.npy', projection_matrix)
np.save('pca_final_100_data_mean.npy', data_mean)

print("Done.")
