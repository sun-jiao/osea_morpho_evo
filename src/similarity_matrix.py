import csv

import torch
from torchvision.models import resnet34

type = 'dimension' # 'species' 'dimension'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 11000
model = resnet34(num_classes=num_classes)
weights_path = 'model20240824.pth'

model.load_state_dict(torch.load(weights_path))
model = model.to(device)
model.eval()

bird_info = []
with open("bird_info.csv", 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        bird_info.append(row)

num_species = len(bird_info)

weights = model.fc.weight.data.to(torch.float16).to(device)
weights = weights[:num_species]

if type == 'species':
    values = weights
elif type == 'dimension':
    # calculate the correlation of dimensions
    values = weights.T
else:
    raise ValueError

num_values = len(values)

# L2 normalize, make sure that all vectors are lengthed 1
normalized = torch.nn.functional.normalize(values, p=2, dim=1)

# dot product of normalized vectors (cosine similarity)
if torch.cuda.is_available():
    with torch.cuda.amp.autocast():
        similarity_matrix = torch.mm(normalized, normalized.T)
else:
    similarity_matrix = torch.mm(normalized, normalized.T)

similarity_matrix = similarity_matrix.float()

# force diagonal points to be 1 to avoid float error
similarity_matrix.fill_diagonal_(1.0)
similarity_np = similarity_matrix.cpu().numpy()

with open(f"class_similarity-{type}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class"] + [str(i) for i in range(num_values)])
    for i in range(num_values):
        writer.writerow([i] + similarity_np[i].tolist())