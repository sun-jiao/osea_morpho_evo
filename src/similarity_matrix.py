import csv

import torch
from torchvision.models import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 11000
model = resnet34(num_classes=num_classes)
weights_path = 'model20240824.pth'

model.load_state_dict(torch.load(weights_path))
model = model.to(device)
model.eval()

weights = model.fc.weight.data.to(torch.float16).to(device)
# L2 normalize, make sure that all vectors are lengthed 1
weights_normalized = torch.nn.functional.normalize(weights, p=2, dim=1)

# dot product of normalized vectors (cosine similarity)
if torch.cuda.is_available():
    with torch.cuda.amp.autocast():
        similarity_matrix = torch.mm(weights_normalized, weights_normalized.T)
else:
    similarity_matrix = torch.mm(weights_normalized, weights_normalized.T)

similarity_matrix = similarity_matrix.float()

# force diagonal points to be 1 to avoid float error
similarity_matrix.fill_diagonal_(1.0)
similarity_np = similarity_matrix.cpu().numpy()

with open("class_similarity.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class"] + [str(i) for i in range(num_classes)])
    for i in range(num_classes):
        writer.writerow([i] + similarity_np[i].tolist())