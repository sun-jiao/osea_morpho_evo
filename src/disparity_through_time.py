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


def pre_process(tree, trait_map):
    """
    Assign values to species, remove value-absent species, merge single-child branches
    :param tree: The phylogenetic tree to be processed
    :param trait_map: A mapping from species names to trait vectors
    :return: The processed phylogenetic tree
    """
    for node in tree.find_clades(order='postorder'):
        for index in range(len(node.clades) - 1, -1, -1):
            child = node.clades[index]
            if hasattr(child, 'remove') and child.remove:
                node.clades.remove(child)
            elif hasattr(child, 'merge') and child.merge:
                grandchild = child.clades[0]
                grandchild.branch_length = child.new_bl
                node.clades[index] = grandchild

        num_children = len(node.clades)
        if num_children == 0:
            if node.name and node.name != '' and node.name in trait_map.keys():
                # In cladistic taxonomy, "ancestor" is a hypothetical scientific model, not representing a real fossil
                # species.
                # Thus, a name indicates that it is an original leaf-node, rather than resulted by child-removing.
                node.state = trait_map[node.name]
            else:
                node.state = None
                # avoid traverse loop, because a node does not have references to its parent,
                # it is necessary to traverse the tree to find its parent.
                node.remove = True
        elif num_children == 1:
            child = node.clades[0]
            bl_parent = node.branch_length if node.branch_length is not None else 0.0
            bl_child = child.branch_length if child.branch_length is not None else 0.0
            # same as above
            node.merge = True
            node.new_bl = bl_parent + bl_child
        else:
            continue

    return tree


def create_trait_mapping(name_match_df, weights_df):
    """create label-trait vector mapping"""
    trait_map = {}
    for index, row in name_match_df.iterrows():
        index = int(row[2])
        label = str(row[0])
        if index != -1:
            if index < len(weights_df):
                trait_map[label] = weights_df.iloc[index].values
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
    tree.eps = eps

    contrast2s = []
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

            if len(child_states) == 2 and len(weights) == 2:
                #  As far as I know, birdtree trees are all strict binary trees
                contrast2 = (child_states[0] - child_states[1]) ** 2 / np.reciprocal(weights).sum()
            else:
                # Generalized contrast for polytomies: weighted sum of squares around ancestor
                weighted_squares = [weights[idx] * (state - node.state) ** 2 for idx, state in enumerate(child_states)]
                contrast2 = np.array(weighted_squares).sum(axis=0)

            # if sorted(contrast2.tolist(), reverse=True)[0] > 5:
            #     breakpoint()

            contrast2s.append(contrast2)

        else:
            node.state = None

    tree.sigma2 = np.sum(contrast2s, axis=0) / (len(trait_map) - 1)  # trait_map has the same size with leaves
    return tree


def assign_node_ages(tree):
    """calculate the node age for each node"""
    for node in tree.find_clades():
        node.age = tree.distance(node)
    return tree


def brownian_null_simulation(tree):
    if not (hasattr(tree.root, "state") and tree.root.state is not None
            and hasattr(tree, "sigma2") and tree.sigma2 is not None):
        return tree

    tree.root.null_state = tree.root.state.copy()
    for node in tree.get_nonterminals(order="preorder"):
        p_null = node.null_state
        for child in node.clades:
            t = child.branch_length if child.branch_length and child.branch_length > 0 else tree.eps
            sd = np.sqrt(tree.sigma2 * t)
            delta = np.random.normal(loc=0.0, scale=sd, size=p_null.shape)
            child.null_state = p_null + delta

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


def compute_dtt_time_slices(tree, num_slices=None, interval=None, null_test=False):
    """
    Calculate the variance of vectors across different time slices of the tree, all active branches included.
    Interpolation for intermediate nodes, real data for leaf nodes.
    """

    state_key = 'state'
    if null_test:
        state_key = 'null_state'

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
        vectors_per_slice[len(slice_times) - 1].append(getattr(leaf, state_key))

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
                if hasattr(p_node, state_key) and hasattr(c_node, state_key):
                    p_state = getattr(p_node, state_key)
                    c_state = getattr(c_node, state_key)
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
def main(tree_file, load_file=None, null_test=False, sample_ratio=100):
    # Sample_ratio only works when null_test is True,
    # because it will take a very long time to run null_test on all 10,000 trees

    base_filename = os.path.basename(tree_file)

    name_match_file = "birdtree_name_match.csv"
    pca_weights_file = "pca_weights.csv"

    # the relationship between labels in the tree and indexes of vectors
    name_match = read_csv(name_match_file)
    # read PCA reduced weights vectors
    pca_weights = read_csv(pca_weights_file)

    trait_mapping = create_trait_mapping(name_match, pca_weights)

    start = time.time()

    progress = get_progress(load_file) if load_file is not None else 0

    trees = open(tree_file, 'r')

    if null_test:
        os.makedirs('output_null', exist_ok=True)
        rand_num = 1  # fixed number for outlier SC test
        # rand_num = np.random.randint(0, sample_ratio)
    else:
        os.makedirs('output', exist_ok=True)
        out_path = f'output/disparity_through_time-{base_filename}-{start:.0f}.csv'
        out_file = open(out_path, 'a')
        writer = csv.writer(out_file)

    for i, line in enumerate(trees):
        if i < progress:
            continue

        if null_test:
            # choose only 1 / sample_ratio of all trees to process
            if (i + rand_num) % sample_ratio != 0:
                continue

            out_path = f'output_null/disparity_through_time-{base_filename}-tree{i}-{start:.0f}.csv'
            out_file = open(out_path, 'a')
            writer = csv.writer(out_file)

        line = line.strip()
        tree_item = read_phylogenetic_trees(line)

        pre_process(tree_item, trait_mapping)

        # check whether there are more than one valid tips
        if len(list(tree_item.get_terminals())) < 2:
            print(f"Tree {i + 1} has insufficient number of leaves")
            continue

        # Do ASR and calculate the ages.
        reconstruct_ancestral_states(tree_item, trait_mapping)
        assign_node_ages(tree_item)

        tree_total_time, dtt_results = compute_dtt_time_slices(tree_item, interval=1.0)
        times, variances = zip(*dtt_results)
        writer.writerow([tree_total_time, *variances])

        if null_test:
            for j in range(100):
                brownian_null_simulation(tree_item)
                tree_total_time, dtt_results = compute_dtt_time_slices(tree_item, interval=1.0, null_test=True)
                times, variances = zip(*dtt_results)
                writer.writerow([tree_total_time, *variances])
                print(f"[{base_filename}] It took {(time.time() - start):.2f} seconds to process {j + 1} times "
                      f"null test in the {i + 1}th tree!")

        print(f"[{base_filename}] It took {(time.time() - start):.2f} seconds to process {i + 1} trees!")

        if null_test:
            out_file.close()

        del line, tree_item, tree_total_time, dtt_results, times, variances
        gc.collect()

    trees.close()
    if not null_test:
        out_file.close()

if __name__ == "__main__":
    processes = []

    for file in os.listdir('CombinedTrees/'):
        p = Process(target=main, args=(f'CombinedTrees/{file}',))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
