import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet34

# -----------------------------
# Config
# -----------------------------
class_A = [1592]
class_B = list(range(1584, 1592))
class_B.extend(list(range(1593, 1606)))

# -----------------------------
# Preprocessing
# -----------------------------
def load_image(img_path):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
    return img_tensor, np.array(img)

# -----------------------------
# Register Hook
# -----------------------------
features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output.detach()

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

# def get_target_layer(model):
#     return dict([*model.named_modules()])[target_layer_name]

# -----------------------------
# Grad-CAM Core
# -----------------------------
def generate_contrastive_cam(model, input_tensor, target_classes, excluded_classes, mode='diff'):
    model.eval()
    output = model(input_tensor)
    target_score = output[0, target_classes].mean()
    excluded_score = output[0, excluded_classes].mean()

    if mode == 'diff':
        score = target_score - excluded_score
    elif mode == 'common':
        score = target_score + excluded_score
    else:
        raise ValueError("mode must be 'diff' or 'common'")

    model.zero_grad()
    score.backward()

    weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = F.relu((weights * features).sum(dim=1, keepdim=True))  # (1, 1, H, W)

    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = cam ** 1.8  # 增强对比
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

# -----------------------------
# Overlay Heatmap
# -----------------------------
def overlay_cam_on_image(img_np, cam):
    # 1. Resize heatmap to match image
    heatmap = cv2.resize((cam * 255).astype(np.uint8), (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # shape (H, W, 3), dtype=uint8

    # 2. Ensure image is uint8 and 3-channel
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)
    if img_np.shape[2] == 1:  # gray → RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 3 and img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)

    # 3. Overlay
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    return overlay


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    # 1. Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = 11000
    model = resnet34(num_classes=num_classes)
    weights_path = 'model20240824.pth'

    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()
    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # 2. Load image
    img_path = '/home/sunjiao/Pictures/勺嘴鹬/59932061-2.jpg'  # 替换成你的图片路径
    input_tensor, img_np = load_image(img_path)

    # 3. Grad-CAM
    cam = generate_contrastive_cam(model, input_tensor, class_A, class_B, mode='diff')
    overlay = overlay_cam_on_image(img_np, cam)

    # 4. Show result
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title(f"Why class {class_A} > class {class_B}")
    plt.imshow(overlay[..., ::-1])  # Convert BGR to RGB
    plt.axis('off')

    plt.tight_layout()
    plt.show()
