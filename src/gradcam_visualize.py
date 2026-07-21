import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet34
from pytorch_grad_cam import GradCAM  #pip install grad-cam
from pytorch_grad_cam.utils.image import show_cam_on_image
# import scienceplots

# plt.style.use(['science','nature'])
# -----------------------------
# Config
# -----------------------------
class_A = list(range(0, 11000))
class_B = []

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
def generate_contrastive_cam(model, input_tensor, target_classes=None, excluded_classes=None, mode='all'):
    if excluded_classes is None:
        excluded_classes = []
    if target_classes is None:
        target_classes = []

    model.eval()
    output = model(input_tensor)
    target_score = output[0, target_classes].mean()
    excluded_score = output[0, excluded_classes].mean()

    if mode == 'all':
        score = output.mean()
    elif mode == 'common':
        score = target_score + excluded_score
    elif mode == 'diff':
        score = target_score - excluded_score
    else:
        raise ValueError("mode must be 'diff' or 'common'")

    model.zero_grad()
    score.backward()

    weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = F.relu((weights * features).sum(dim=1, keepdim=True))  # (1, 1, H, W)

    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = cam ** 2
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
    overlay = cv2.addWeighted(img_np, 0.8, heatmap, 0.5, 0)
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
    
    image_paths = [
        '../test_images/A9_01683.jpg',
        '../test_images/A9_08275.jpg',
        '../test_images/A9_09326.jpg',
        '../test_images/A9_09426.jpg'
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8)) 

    for idx, img_path in enumerate(image_paths):
        input_tensor, img_np = load_image(img_path)
        cam = generate_contrastive_cam(model, input_tensor, mode='all')
        overlay = overlay_cam_on_image(img_np, cam)

        row = idx // 2
        col_orig = (idx % 2) * 2
        col_cam = col_orig + 1

        axes[row, col_orig].imshow(img_np)
        axes[row, col_orig].set_title(f"Image {idx+1}: Original", fontsize=10)
        axes[row, col_orig].axis('off')

        axes[row, col_cam].imshow(overlay[..., ::-1])  # BGR -> RGB
        axes[row, col_cam].set_title(f"Image {idx+1}: Grad-CAM", fontsize=10)
        axes[row, col_cam].axis('off')

    plt.tight_layout()
    plt.savefig('../document/gradcam_2x2.pdf', dpi=300)
    # plt.show()