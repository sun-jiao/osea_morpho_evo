import pandas as pd
import numpy as np
import torch
from torchvision.models import resnet34
from sklearn.preprocessing import normalize

run_type = 'paca'

MODEL_PATH = 'model20240824.pth'
BIRD_INFO_PATH = "bird_info.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 11000
model = resnet34(num_classes=num_classes)

model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval()

bird_info = pd.read_csv(BIRD_INFO_PATH, header=None).values
num_species = len(bird_info)

raw_weights = model.fc.weight.data[:num_species].detach().cpu().numpy()
weights_norm = normalize(raw_weights, norm='l2', axis=1)

if run_type == 'paca':
    # PACA rotation matrix
    w_paca_df = pd.read_feather("PACA_rot.feather")
    w_paca = w_paca_df.to_numpy()
    mean_paca_df = pd.read_feather("PACA_center.feather")
    mean_paca = mean_paca_df.to_numpy().flatten()

    # PCA projection matrix
    w_pca = np.load('pca_final_100_projection_matrix.npy')
    mean_pca = np.load('pca_final_100_data_mean.npy')

    x_pca_centered = weights_norm - mean_pca
    x_pca = x_pca_centered @ w_pca 
    x_paca_centered = x_pca - mean_paca
    x_paca = x_paca_centered @ w_paca
    final_weights = normalize(x_paca, norm='l2', axis=1)
elif run_type == 'original':
    final_weights = weights_norm
else:
    raise ValueError("Invalid type. Must be 'original' or 'paca'.")

final_weights = np.asanyarray(final_weights)
similarity_matrix = np.dot(final_weights, final_weights.T)

similarity_matrix = similarity_matrix.float()

# force diagonal points to be 1 to avoid float error
similarity_matrix.fill_diagonal_(1.0)
similarity_np = similarity_matrix.cpu().numpy()

result_df = pd.DataFrame(similarity_np)
result_df.to_feather(f"class_similarity-{run_type}.feather")
