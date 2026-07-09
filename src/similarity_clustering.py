import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import to_tree
from scipy.spatial.distance import squareform

type = 'species' # 'species' 'dimension'
BIRD_INFO_PATH = "bird_info.csv"

filename = f"class_similarity-{type}.feather"

df = pd.read_feather(filename)

similarity_matrix = df.to_numpy()

bird_info = pd.read_csv(BIRD_INFO_PATH, header=None).values

num_species = len(bird_info)

labels = [f"{i}_{bird_info[i][2]}_{bird_info[i][0]}".replace(' ', '_') for i in range(num_species)]

# reduced_matrix = similarity_matrix[:num_species, :num_species]
distance_matrix = 1 - similarity_matrix
distance_matrix = (distance_matrix + distance_matrix.T) / 2  # avoid float error
condensed_matrix = squareform(distance_matrix)

# hierarchy clustering
linkage_matrix = hierarchy.linkage(condensed_matrix, method='average')

# convert to a newick file
def linkage_to_newick(matrix, labels):
    tree = to_tree(matrix, rd=False)
    def build_newick(node, parent_dist, leaf_names):
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{parent_dist - node.dist:.4f}"
        else:
            left = build_newick(node.left, node.dist, leaf_names)
            right = build_newick(node.right, node.dist, leaf_names)
            # subtree sorting
            children = sorted([left, right])
            return f"({children[0]},{children[1]}):{parent_dist - node.dist:.4f}"

    return f"{build_newick(tree, tree.dist, labels)};"

newick_str = linkage_to_newick(linkage_matrix, labels)

# save a file
with open(f"similarity_clustering-{type}.tre", "w") as f:
    f.write(newick_str)