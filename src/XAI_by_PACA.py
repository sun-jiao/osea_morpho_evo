from math import sqrt
import pandas as pd
import numpy as np
import torch
from torchvision.models import resnet34
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr, spearmanr, kruskal
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = 'model20240824.pth'
BIRD_INFO_PATH = "bird_info.csv"
AVONET_TRAITS_PATH = "ELEData/TraitData/AVONET3_BirdTree.xlsx"
NUMERIC_TRAIT_LABELS = [
    "Beak.Length_Culmen", "Beak.Length_Nares", "Beak.Width", "Beak.Depth", "Tarsus.Length", "Wing.Length", "Kipps.Distance", "Secondary1", "Hand-Wing.Index", "Tail.Length", "Mass"
]
ENUMERATED_TRAIT_LABELS = [
    "Habitat", "Migration", "Trophic.Level", "Trophic.Niche", "Primary.Lifestyle"
]

# ==============================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

bird_info = pd.read_csv(BIRD_INFO_PATH, header=None).values
num_species = len(bird_info)
print(f"Number of species: {num_species}")

species_names = [row[0] + " (" + row[2] + ")" for row in bird_info]

model = resnet34(num_classes=11000)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
target_layer = model.avgpool  # 最后一层卷积层的输出

raw_weights = model.fc.weight.data[:num_species].detach().cpu().numpy()
weights_norm = normalize(raw_weights, norm='l2', axis=1)

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
final_weights = np.asanyarray(final_weights)

w_combined = np.dot(w_pca, w_paca)

name_match = pd.read_csv("avian_timetree_name_match.csv", header=None)
name_match = name_match.sort_values(by=name_match.columns[0])

paca_x = pd.read_feather("PACA_x.feather")

paca_x.set_index("Species", inplace=True)
paca_x = paca_x.sort_index()

avonet_data = pd.read_excel(AVONET_TRAITS_PATH, sheet_name="AVONET3_BirdTree", index_col=0, header=0)
avonet_name_match = pd.read_csv("birdtree_name_match.csv", header=None)

avonet_name_mapping = dict(zip(avonet_name_match[1], avonet_name_match[2]))

avonet_data['B_index'] = avonet_data.index.map(avonet_name_mapping)
avonet_data = avonet_data[avonet_data['B_index'] >= 0]
avonet_data['B_index'] = avonet_data['B_index'].astype(int)
new_indexes = [bird_info[b][2] for b in avonet_data['B_index']]

avonet_data.index = new_indexes
avonet_data = avonet_data.drop(columns=['B_index'])

num_dims = len(paca_x.columns)

all_traits = NUMERIC_TRAIT_LABELS + ENUMERATED_TRAIT_LABELS
pac_labels = [i+1 for i in range(num_dims)]

# Initialize an empty DataFrame to store correlation values
# Default fill with NaN, so that cells with p > 0.05 will naturally
# remain blank in the heatmap
heatmap_data_numeric = pd.DataFrame(index=NUMERIC_TRAIT_LABELS, columns=pac_labels, dtype=float)
heatmap_data_enum = pd.DataFrame(index=ENUMERATED_TRAIT_LABELS, columns=pac_labels, dtype=float)

for i in range(num_dims):
    # print(f"\n=== PACA {i+1} ===")
    # The first PACA axes (PAC1) in the original embedding space
    scores_all = final_weights[:, i]

    indices = name_match.iloc[:, 2].to_numpy(dtype=int)
    score_sorted = np.asarray(scores_all)[indices]
    x_value = np.asarray(paca_x)[:, i]

    rho, p_value = pearsonr(x_value, score_sorted)

    # print(f"Calculated values (top 10): {score_sorted[0:10]}\r\nOriginal values produced by geomorph: {x_value[0:10]}")
    # print(f"Relationship with empirical value: Spearman rho = {rho:.6f}, p-value = {p_value:.6g}")

    df_scores = pd.DataFrame({
        'Species': species_names,
        f'PAC{i+1}_Score': scores_all
    })

    df_sorted = df_scores.sort_values(by=f'PAC{i+1}_Score', ascending=False)

    top_10 = df_sorted.head(10)
    bottom_10 = df_sorted.tail(10)

    # print("\n=== Highest scores ===")
    # print(top_10)
    # print("\n=== Lowest scores ===")
    # print(bottom_10)

    # Add PACA values to the avonet_data DataFrame for correlation analysis
    score_mapping = {bird_info[idx][2]: scores_all[idx] for idx in range(num_species)}
    avonet_data['Current_Score'] = avonet_data.index.map(score_mapping)
    
    # print("\n=== Numeric Traits Correlation (Spearman) ===-")
    for item in NUMERIC_TRAIT_LABELS:
        # drop NaN rows
        valid_data = avonet_data[['Current_Score', item]].dropna()
        if len(valid_data) > 2:
            rho_num, p_num = spearmanr(valid_data['Current_Score'], valid_data[item])
            rho_num_squared = rho_num ** 2
            # print(f"  {item:20s}: rho^2 = {rho_num_squared:>7.4f}, p-value = {p_num:.4g}, n = {len(valid_data)}")
            if p_num <= 0.05:
                heatmap_data_numeric.at[item, i + 1] = rho_num_squared
            
    # print("\n=== Enumerated Traits Analysis (Kruskal-Wallis & Eta-squared) ===")
    for item in ENUMERATED_TRAIT_LABELS:
        # drop NaN rows
        valid_data = avonet_data[['Current_Score', item]].dropna()
        
        groups = [group['Current_Score'].values for name, group in valid_data.groupby(item)]
        
        if len(groups) > 1: 
            h_stat, p_cat = kruskal(*groups)
            
            # Eta-squared (η²)
            # η² = H / (n - 1), 
            # n is the total number of observations across all groups
            # 0.01~0.06: weak; 0.06~0.14 median; >0.14 strong correlation
            n = len(valid_data)
            eta_sq = h_stat / (n - 1) if n > 1 else 0
            
            # print(f"  {item:20s}: H-stat = {h_stat:>8.4f}, p-value = {p_cat:.4g}, Eta-squared = {eta_sq:.4f}, n = {n}")

            if p_cat <= 0.05:
                heatmap_data_enum.at[item, i + 1] = sqrt(eta_sq)

    avonet_data = avonet_data.drop(columns=['Current_Score'])

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8 + num_dims * 0.05, 6 + len(all_traits) * 0.25), sharex=True,
    gridspec_kw={"height_ratios": [len(NUMERIC_TRAIT_LABELS), len(ENUMERATED_TRAIT_LABELS)]},
)

sns.heatmap(heatmap_data_numeric, 
            ax=ax1,
            annot=False,
            fmt=".3f",
            cmap="YlOrRd",
            cbar_kws={'label': 'Correlation (|ρ|)'},
            mask=heatmap_data_numeric.isnull(),
            linewidths=0.5,
            linecolor='lightgray')

ax1.set_ylabel("AVONET Traits (Numeric)", fontsize=12, fontweight='bold')

sns.heatmap(heatmap_data_enum, 
            ax=ax2,
            annot=False,
            fmt=".3f",
            cmap="YlOrRd",
            cbar_kws={'label': 'Correlation (η)'},
            mask=heatmap_data_enum.isnull(),
            linewidths=0.5,
            linecolor='lightgray')

ax2.set_ylabel("AVONET Traits (Enumerated)", fontsize=12, fontweight='bold')
ax2.set_xlabel("PACA Dimensions", fontsize=12, fontweight='bold')

ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
fig.suptitle('Trait Correlations across PACA Dimensions\n(Blank cells indicate p > 0.05)', 
          fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('trait_correlation_heatmap.png', dpi=300)
# plt.show()