import bisect
import csv
import io
import os
import time
from multiprocessing import Process
import gc

import numpy as np
import pandas as pd
from Bio import Phylo


# ---------------------------
# Data reading and processing
# ---------------------------
def read_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    return df


def read_phylogenetic_trees(tree_string):
    handle = io.StringIO(tree_string)
    tree = Phylo.read(handle, "newick")
    return tree


def get_progress(file_path):
    df = pd.read_csv(file_path, header=None)
    return df.shape[0]


def remove_missing_data_nodes(tree, name_match_df):
    """Remove nodes that have no corresponding vector from the tree
    No excluded species included in the tree, so no need to remove them.
    """
    missing_species = name_match_df[name_match_df.iloc[:, 2] == -1].iloc[:, 0].tolist()
    terminals_to_remove = [terminal for terminal in tree.get_terminals() if terminal.name in missing_species]
    for terminal in terminals_to_remove:
        path = tree.get_path(terminal)
        if len(path) < 2:
            continue  # for root nodes
        parent = path[-2]
        parent.clades.remove(terminal)
    # collapse nodes that have only one child (because another child may have been removed)
    collapse_single_child_nodes(tree)
    if tree.root.branch_length is None:
        tree.root.branch_length = 0.0


def collapse_single_child_nodes(tree):
    """collapse nodes that have only one child, and combine the branch lengths. traversing from the leaf nodes (species)"""
    for clade in tree.get_nonterminals(order='postorder'):
        while len(clade.clades) == 1 and clade != tree.root:
            child = clade.clades[0]
            bl_parent = clade.branch_length if clade.branch_length is not None else 0.0
            bl_child = child.branch_length if child.branch_length is not None else 0.0
            new_bl = bl_parent + bl_child
            parent = find_parent(tree, clade)
            if parent is None:
                break
            index = parent.clades.index(clade)
            parent.clades[index] = child
            child.branch_length = new_bl
            clade = child


def find_parent(tree, target):
    """In biopython, a node has references to its child nodes, but not parent nodes.
    Thus, we have to traverse the tree"""
    for clade in tree.find_clades():
        if target in clade.clades:
            return clade
    return None


def create_trait_mapping(tree, name_match_df, pca_weights_df):
    """create label-trait vector mapping"""
    trait_map = {}
    for leaf in tree.get_terminals():
        label = leaf.name
        match = name_match_df[name_match_df.iloc[:, 0] == label]
        if not match.empty and match.iloc[0, 2] != -1:
            index = int(match.iloc[0, 2])
            if index < len(pca_weights_df):
                trait_map[label] = pca_weights_df.iloc[index].values
            else:
                print(f"Warning: Index {index} is out of the range of vectors for label {label}.")
    return trait_map


# ---------------------------
# Ancestral state reconstruction
# ---------------------------
def reconstruct_ancestral_states(tree, trait_map):
    """
    Ancestor State Reconstruction, using Phylogenetic Independent Contrast methods:
    - Leaf nodes: Directly assign from trait_map.
    - Internal nodes: Use weighted average (based on the reciprocal of branch length).
    - If branch_length is missing or zero, it’s replaced with a tiny ε to avoid division by zero.
    All states are stored in `node.state`.
    """

    # tip-value assigning
    for tip in tree.get_terminals():
        if tip.name in trait_map:
            tip.state = trait_map[tip.name]
            tip.equiv_length = 0.0
        else:
            tip.state = None  # missing value
            tip.equiv_length = 0.0

    # make the default value smaller than the min value.
    all_bl = [node.branch_length for node in tree.find_clades()
              if node.branch_length and node.branch_length > 0]
    min_bl = min(all_bl) if all_bl else 1.0
    eps = min_bl * 1e-4

    # post-order traverse and calculate the weighted average
    for node in tree.get_nonterminals(order="postorder"):
        child_states = []
        weights = []
        for child in node.clades:
            if hasattr(child, "state") and child.state is not None:
                bl = child.branch_length if child.branch_length and child.branch_length > 0 else eps
                equiv_bl = child.equiv_length if child.equiv_length else 0
                weight = 1.0 / (bl + equiv_bl)
                weights.append(weight)
                child_states.append(child.state)
        if child_states:
            weights = np.array(weights)
            node.state = np.average(child_states, axis=0, weights=weights)
            node.equiv_length = 1 / weights.sum()
        else:
            node.state = None

    return tree


def assign_node_ages(tree):
    """calculate the node age for each node"""
    for node in tree.find_clades():
        node.age = tree.distance(node)
    return tree


# ---------------------------
# Calculate the disparity
# ---------------------------
def interpolate_state(p_state, c_state, p_age, c_age, t):
    """Calculate the state of a given time on a branch"""
    if c_age == p_age:
        return p_state
    frac = (t - p_age) / (c_age - p_age)
    return p_state + frac * (c_state - p_state)


def compute_variance(vectors):
    """
    Compute the variance of vectors based on Euclidean distance
    """
    if len(vectors) < 2:
        return 0.0
    arr = np.array(vectors)
    mean_vec = np.mean(arr, axis=0)
    sq_dists = np.sum((arr - mean_vec) ** 2, axis=1)
    return np.mean(sq_dists)


def compute_dtt_time_slices(tree, num_slices=None, interval=None):
    """
    Calculate the variance of vectors across different time slices of the tree, all active branches included.
    Interpolation for intermediate nodes, real data for leaf nodes.
    """

    max_time = max(tree.depths().values())

    slice_times = None

    if (interval is None) == (num_slices is None):
        raise ValueError("Exactly one of 'interval' or 'num_slices' must be set.")

    if interval:
        slice_times = np.arange(max_time, -interval, -interval).tolist()
        if slice_times[-1] > 0:
            slice_times.append(0.0)
        slice_times = sorted(slice_times)
    else:
        slice_times = np.linspace(0, max_time, num_slices).tolist()

    # active clades per slice, use indexes as keys to avoid float errors
    vectors_per_slice = {i: [] for i in range(len(slice_times))}

    # avoid floating-point error, in one of my cases, t of the last slice
    # was 112.4116612, while end_time was 112.41149999.
    # so we just use all extant species (terminal nodes).
    for leaf in tree.get_terminals():
        # if leaf.name in trait_map: # already aligned.
        vectors_per_slice[len(slice_times) - 1].append(leaf.state)

    # traverse every clade
    def collect_active_clades(p_node):
        for c_node in p_node.clades:
            start_time = p_node.age
            end_time = c_node.age

            i_start = bisect.bisect_left(slice_times, start_time)
            i_end = bisect.bisect_right(slice_times, end_time, lo=i_start)

            for i in range(i_start, i_end):
                t = slice_times[i]
                # get states of the given time
                if hasattr(p_node, 'state') and hasattr(c_node, 'state'):
                    p_state = p_node.state
                    c_state = c_node.state
                    if p_state is not None and c_state is not None:
                        interp_state = interpolate_state(p_state, c_state, start_time, end_time, t)
                        vectors_per_slice[i].append(interp_state)
            collect_active_clades(c_node)

    collect_active_clades(tree.root)

    results = []
    for i in range(len(slice_times)):
        t = slice_times[i]
        variance = compute_variance(vectors_per_slice[i])
        results.append((t, variance))

    return max_time, results


# ---------------------------
# Main function
# ---------------------------
def main(tree_file, load_file=None):
    base_filename = os.path.basename(tree_file)

    name_match_file = "birdtree_name_match.csv"
    pca_weights_file = "pca_weights.csv"

    # the relationship between labels in the tree and indexes of vectors
    name_match = read_csv(name_match_file)
    # read PCA reduced weights vectors
    pca_weights = read_csv(pca_weights_file)

    trait_mapping = None

    start = time.time()

    os.makedirs('output', exist_ok=True)
    out_path = f'output/disparity_through_time-{base_filename}-{start:.0f}.csv'
    progress = get_progress(load_file) if load_file is not None else 0

    with open(tree_file) as f:
        with open(out_path, 'a') as out_file:
            writer = csv.writer(out_file)

            for i, line in enumerate(f):
                if i < progress:
                    continue

                line = line.strip()
                tree_item = read_phylogenetic_trees(line)

                if trait_mapping is None:
                    trait_mapping = create_trait_mapping(tree_item, name_match, pca_weights)

                remove_missing_data_nodes(tree_item, name_match)

                # check whether there are more than one valid tips
                if len(list(tree_item.get_terminals())) < 2:
                    print(f"Tree {i + 1} has insufficient number of leaves")
                    continue

                # Do ASR and calculate the ages.
                reconstruct_ancestral_states(tree_item, trait_mapping)
                assign_node_ages(tree_item)

                tree_total_time, dtt_results = compute_dtt_time_slices(tree_item, interval=1.0)

                times, variances = zip(*dtt_results)

                print(f"[{base_filename}] It took {(time.time() - start):.2f} seconds to process {i} trees!")

                writer.writerow([tree_total_time, *variances])

                del line, tree_item, tree_total_time, dtt_results, times, variances
                gc.collect()

if __name__ == "__main__":
    processes = []

    for file in os.listdir('CombinedTrees/'):
        p = Process(target=main, args=(f'CombinedTrees/{file}',))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
