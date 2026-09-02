import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet34

# 1. Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 11000
model = resnet34(num_classes=num_classes)
weights_path = 'model20240824.pth'

model.load_state_dict(torch.load(weights_path))
model = model.to(device)
model.eval()

# 2. Check for classes with identical (or near-identical) FC weights
fc_weights_all = model.fc.weight.data  # shape: (11000, in_features)
total_classes, in_features = fc_weights_all.shape
print(f"FC layer shape (total): {fc_weights_all.shape}")

# Drop classes 10964-10999 (they do not represent species)
valid_mask = torch.arange(total_classes) < 10964
fc_weights = fc_weights_all[valid_mask]  # shape: (10964, in_features)
num_classes = fc_weights.shape[0]
print(f"FC layer shape (valid species only): {fc_weights.shape}")

# Normalize weights row-wise for cosine similarity computation
fc_norm = F.normalize(fc_weights, p=2, dim=1)

# Compute pairwise cosine similarity matrix: (num_classes, num_classes)
cos_sim = fc_norm @ fc_norm.T  # cosine similarity matrix

# Set diagonal to -1 so we don't flag self-similarity
cos_sim.fill_diagonal_(-1.0)

# Find pairs with similarity above threshold
threshold = 0.9999  # near-identical
identical_pairs = torch.where(cos_sim > threshold)

if len(identical_pairs[0]) == 0:
    threshold = 0.999
    identical_pairs = torch.where(cos_sim > threshold)

def find_connected_groups(pairs):
    """Find connected components from a list of (i, j) pairs."""
    adj = {}
    for i, j in pairs:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    visited = set()
    groups = []
    for node in adj:
        if node not in visited:
            stack = [node]
            visited.add(node)
            group = []
            while stack:
                cur = stack.pop()
                group.append(cur)
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            groups.append(sorted(group))
    return groups


if len(identical_pairs[0]) == 0:
    print(f"\nNo classes have cosine similarity >= {threshold}. Checking top matches...")
    # Show the top-10 most similar pairs
    flat_indices = torch.argsort(cos_sim.flatten(), descending=True)
    top_k = min(20, len(flat_indices))
    for rank, flat_idx in enumerate(flat_indices[:top_k]):
        i = flat_idx // num_classes
        j = flat_idx % num_classes
        if i < j:  # only show each pair once
            sim = cos_sim[i, j].item()
            l2_dist = torch.norm(fc_weights[i] - fc_weights[j], p=2).item()
            print(f"  Rank {rank+1}: class {i} <-> class {j} | cos_sim={sim:.8f} | L2={l2_dist:.6f}")
else:
    pairs = [(int(i), int(j)) for i, j in zip(identical_pairs[0], identical_pairs[1]) if i < j]
    groups = find_connected_groups(pairs)
    print(f"\nFound {len(groups)} group(s) of near-identical classes (cos_sim >= {threshold}):")
    for g_idx, group in enumerate(groups):
        members = ", ".join(str(c) for c in group)
        print(f"  Group {g_idx+1} ({len(group)} classes): {members}")

# Also do an exact equality check (bitwise identical) using connected components
print("\n--- Exact equality check (all elements bitwise equal) ---")
exact_pairs = []
for i in range(num_classes):
    for j in range(i + 1, num_classes):
        if torch.equal(fc_weights[i], fc_weights[j]):
            exact_pairs.append((i, j))

if exact_pairs:
    exact_groups = find_connected_groups(exact_pairs)
    print(f"Found {len(exact_groups)} group(s) with bitwise identical weight vectors:")
    for g_idx, group in enumerate(exact_groups):
        members = ", ".join(str(c) for c in group)
        print(f"  Group {g_idx+1} ({len(group)} classes): {members}")
else:
    print("No classes have bitwise identical weight vectors.")

# Summary statistics
print(f"\n--- Summary ---")
print(f"Total classes (all): {total_classes}")
print(f"Valid classes (species only): {num_classes}")
print(f"Excluded classes: {total_classes - num_classes}  (indices 10964-10999)")
print(f"FC input features: {in_features}")
print(f"Max cosine similarity (excluding self): {cos_sim.max().item():.8f}")
print(f"Min cosine similarity (excluding self): {cos_sim.min().item():.8f}")
print(f"Mean cosine similarity (excluding self): {cos_sim[cos_sim > -0.5].mean().item():.8f}")

