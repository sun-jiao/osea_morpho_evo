import csv

import numpy as np
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import to_tree
from scipy.spatial.distance import squareform

bird_info = []
with open("bird_info.csv", 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        bird_info.append(row)

num_species = len(bird_info)
labels = [f"{i}_{bird_info[i][2]}_{bird_info[i][0]}".replace(' ', '_') for i in range(num_species)]
similarity_matrix = np.loadtxt(
    "class_similarity.csv",
    delimiter=',',
    skiprows=1,          # skip the frist row
    usecols=range(1, len(open("class_similarity.csv").readline().split(','))  # skip the first column
                  ))
reduced_matrix = similarity_matrix[:num_species, :num_species]
distance_matrix = 1 - reduced_matrix
condensed_matrix = squareform(distance_matrix)

# hierarchy clustering
linkage_matrix = hierarchy.linkage(condensed_matrix, method='average')

# convert to newick file
def linkage_to_newick(matrix, labels):
    tree = to_tree(matrix, rd=False)
    def build_newick(node, parent_dist, leaf_names):
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{parent_dist - node.dist:.4f}"
        else:
            left = build_newick(node.left, node.dist, leaf_names)
            right = build_newick(node.right, node.dist, leaf_names)
            return f"({left},{right}):{parent_dist - node.dist:.4f}"
    return f"{build_newick(tree, tree.dist, labels)};"

newick_str = linkage_to_newick(linkage_matrix, labels)

# save file
with open("morphology_cluster.tre", "w") as f:
    f.write(newick_str)