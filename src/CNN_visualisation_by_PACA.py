import pandas as pd
import numpy as np
import torch
from torchvision.models import resnet34
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr
import cv2
import numpy as np

def save_gradcam_opencv(img_path, cam, save_path):
    """
    img_path: 原始图片的路径 (例如 'dog.jpg')
    cam: 经过 ReLU 后计算出的 2D numpy 数组 (形状如 [H, W], 值在 0~1 之间)
    save_path: 保存结果的路径 (例如 'gradcam_result.jpg')
    """
    # 1. 读取原图
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # 2. 将 CAM 矩阵缩放到与原图同样的大小
    cam = cv2.resize(cam, (w, h))

    # 3. 将 0~1 的 float 转换为 0~255 的 uint8 格式
    cam_img = np.uint8(255 * cam)

    # 4. 施加伪彩色（JET 模式：值大的地方变红，值小的地方变蓝）
    heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

    # 5. 将热力图与原图融合 (alpha 是原图权重，beta 是热力图权重)
    # 这里的 heatmap 和 img 必须都是 BGR 格式
    result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # 6. 保存到本地
    cv2.imwrite(save_path, result)
    print(f"Grad-CAM 结果已成功保存至: {save_path}")

MODEL_PATH = 'model20240824.pth'
BIRD_INFO_PATH = "bird_info.csv"

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
x_paca_final = normalize(x_paca, norm='l2', axis=1)

w_combined = np.dot(w_pca, w_paca)

name_match = pd.read_csv("avian_timetree_name_match.csv", header=None)
name_match = name_match.sort_values(by=name_match.columns[0])

paca_x = pd.read_feather("PACA_x.feather")

paca_x.set_index("Species", inplace=True)
paca_x = paca_x.sort_index()

for i in range(1):
    # The first PACA axes (PAC1) in the original embedding space
    paca_vector = w_combined[:, i]

    scores_all = np.dot(weights_norm, paca_vector)

    indices = name_match.iloc[:, 2].to_numpy(dtype=int)
    score_sorted = np.asarray(scores_all)[indices]
    x_value = np.asarray(paca_x)[:, i]

    rho, p_value = spearmanr(x_value, score_sorted)

    print(f"Spearman rho = {rho:.6f}")
    print(f"p-value = {p_value:.6g}")

    df_scores = pd.DataFrame({
        'Species': species_names,
        f'PAC{i+1}_Score': scores_all
    })

    df_sorted = df_scores.sort_values(by=f'PAC{i+1}_Score', ascending=False)

    top_10 = df_sorted.head(10)
    bottom_10 = df_sorted.tail(10)

    print(f"\n=== PACA {i+1} highest scores ===")
    print(top_10)
    print(f"\n=== PACA {i+1} lowest scores ===")
    print(bottom_10)

    feature_maps = []
    gradients = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_feature(module, input, output):
        feature_maps.append(output)

    target_layer.register_forward_hook(save_feature)
    target_layer.register_full_backward_hook(save_gradient)

    # 3. 前向传播
    inputs = torch.randn(1, 3, 224, 224, requires_grad=True)
    
    x = model.conv1(inputs)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)  # 到这里是最后一层卷积的输出 [B, 512, H, W] 或 [B, 2048, H, W]

    x = model.avgpool(x) # 经过池化层 [B, C, 1, 1]
    features = torch.flatten(x, 1) # 展平得到倒数第二层的特征向量 [B, C]

    paca_vector_normalized = paca_vector / np.linalg.norm(paca_vector)
    direction_vector = torch.from_numpy(paca_vector_normalized).float().to(inputs.device)

    # 5. 计算在该方向上的投影得分 (Scalar)
    score = torch.sum(features * direction_vector)

    # 6. 反向传播计算梯度
    model.zero_grad()
    score.backward()

    # 7. 按照 Grad-CAM 公式聚合
    # gradients[0] 形状: [1, C, H, W]
    # feature_maps[0] 形状: [1, C, H, W]
    grads = gradients[0].cpu().data.numpy()[0]
    f_maps = feature_maps[0].cpu().data.numpy()[0]

    weights = grads.mean(axis=(1, 2)) # 在 H, W 轴上取平均，得到通道权重
    cam = np.zeros(f_maps.shape[1:], dtype=np.float32)

    for j, w in enumerate(weights):
        cam += w * f_maps[j, :, :]

    # ReLU 激活
    cam = np.maximum(cam, 0)
    save_gradcam_opencv('/home/sunjiao/Downloads/images.webp', cam, f'paca_gradcam_output{i+1}.jpg')
