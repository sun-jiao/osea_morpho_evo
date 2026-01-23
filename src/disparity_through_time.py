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
from Bio.Phylo.BaseTree import Tree


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


def get_species_by_group(csv_file, target_group, level='order'):
    """
    get species list of specific group from bird_info.csv (e.g. PASSERIFORMES)
    """
    target_species = set()

    if level == 'order':
        group_idx = 4
    elif level == 'family':
        group_idx = 5
    else:
        raise ValueError("Level must be 'order' or 'family'")

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > group_idx and row[group_idx] == target_group:
                name = row[2].replace(' ', '_')
                target_species.add(name)
    return target_species


def extract_subclade(tree, target_species_set):
    """
    extract subclade from bird_info.csv (e.g. PASSERIFORMES)
    """
    # find all existing leaf nodes that belongs to this group
    present_targets = [leaf for leaf in tree.get_terminals() if leaf.name in target_species_set]

    if len(present_targets) < 2:
        return None  # too little to conduct an analysis

    # find MRCA
    mrca = tree.common_ancestor(present_targets)

    new_tree = Tree(root=mrca)

    return new_tree


def prune_subclade(tree, target_species_set):
    """
    remove subclade from bird_info.csv (e.g. PASSERIFORMES)
    """
    # find all leaf nodes
    to_prune = [leaf for leaf in tree.get_terminals() if leaf.name in target_species_set]

    for leaf in to_prune:
        try:
            tree.prune(leaf)
        except ValueError:
            continue

    if len(list(tree.get_terminals())) < 2:
        return None

    return tree

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
            """
                Computes the ancestral state and independent contrasts for spherical data.

                Strategy:
                - N=2: Uses Exact Spherical Linear Interpolation (SLERP).
                - N>2: Uses Projected Arithmetic Mean with Geometric Correction.
                """
            weights = np.array(weights)
            child_states = np.array(child_states)

            # 1. Calculate Equivalent Length (Felsenstein, 1985)
            # This is topology-dependent, not geometry-dependent.
            node.equiv_length = 1.0 / weights.sum()

            # 2. Hybrid State Reconstruction & Contrast Calculation

            # CASE A: Bifurcation (N=2) -> Use Geometric Exact Solution (SLERP)
            if len(child_states) == 2:
                vec_a = child_states[0]
                vec_b = child_states[1]

                # Calculate angle theta (Geodesic distance)
                dot_prod = np.dot(vec_a, vec_b)
                dot_prod = np.clip(dot_prod, -1.0, 1.0)
                theta = np.arccos(dot_prod)

                # --- State Reconstruction: SLERP ---
                if theta < 1e-9:
                    # If vectors are identical, simple average avoids division by zero
                    node.state = vec_a
                else:
                    # Calculate interpolation factor t based on weights
                    # P = (sin((1-t)θ)/sinθ) * A + (sin(tθ)/sinθ) * B
                    # Weight w1 pulls towards B, so t = w1 / (w0 + w1)
                    t = weights[1] / weights.sum()

                    sin_theta = np.sin(theta)
                    coeff_a = np.sin((1 - t) * theta) / sin_theta
                    coeff_b = np.sin(t * theta) / sin_theta

                    node.state = coeff_a * vec_a + coeff_b * vec_b

                    # Re-normalize to ensure numerical stability on the manifold
                    # (SLERP theoretically stays on sphere, but float errors accumulate)
                    node.state = node.state / np.linalg.norm(node.state)

                # --- Contrast Calculation ---
                # Even with SLERP, we need a vector representing the contrast.
                # We use the Scaled Euclidean Difference to approximate the Tangent Vector.
                # Scale = Arc_Length / Chord_Length

                raw_diff = vec_a - vec_b
                euclidean_sq_dist = np.sum(raw_diff ** 2)

                # Base PIC variance (Euclidean)
                contrast_variance = (raw_diff ** 2) / np.reciprocal(weights).sum()

                # Correction Factor: Arc^2 / Chord^2
                arc_sq_dist = theta ** 2

                if euclidean_sq_dist > 1e-9:
                    correction_factor = arc_sq_dist / euclidean_sq_dist
                else:
                    correction_factor = 1.0

                contrast2 = contrast_variance * correction_factor

            # CASE B: Polytomy (N != 2) -> Use Projected Mean Approximation
            else:
                # --- State Reconstruction: Projected Mean ---
                # 1. Geometric Center in Euclidean Space
                raw_state = np.average(child_states, axis=0, weights=weights)

                # 2. Project back to Hypersphere
                current_norm = np.linalg.norm(raw_state)
                if current_norm > 1e-9:
                    node.state = raw_state / current_norm
                else:
                    node.state = raw_state

                # --- Contrast Calculation: Generalized & Corrected ---
                contrast2 = np.zeros_like(node.state)

                for idx, state in enumerate(child_states):
                    weight = weights[idx]

                    # Euclidean difference vector relative to the approximated ancestor
                    diff_vec = state - node.state
                    eu_dist_sq = np.sum(diff_vec ** 2)

                    # Geodesic distance (Arc length)
                    dp = np.dot(state, node.state)
                    dp = np.clip(dp, -1.0, 1.0)
                    arc_dist_sq = np.arccos(dp) ** 2

                    # Correction Factor
                    factor = 1.0
                    if eu_dist_sq > 1e-9:
                        factor = arc_dist_sq / eu_dist_sq

                    # Accumulate weighted, geometrically corrected variance
                    contrast2 += weight * (diff_vec ** 2) * factor

            # if sorted(contrast2.tolist(), reverse=True)[0] > 5:
            #     breakpoint()

            contrast2s.append(contrast2)

        else:
            node.state = None

    num_tips = len(list(tree.get_terminals()))
    tree.sigma2 = np.sum(contrast2s, axis=0) / (num_tips - 1)  # trait_map has the same size with leaves
    return tree


def assign_node_ages(tree):
    """calculate the node age for each node"""
    for node in tree.find_clades():
        node.age = tree.distance(node)
    return tree


def brownian_null_simulation(tree):
    """
    Riemannian Brownian Motion on the Unit Hypersphere, Simulating the Neutral Evolution

    Assumption:
        The input tree.root.state is already L2-normalised (Unit Vector).
    Mechanism:
        1. Projects Euclidean noise onto the tangent space (Gram-Schmidt).
        2. Use the Exponential Map (Geodesic flow) to move along the great circle.
    """
    # tree check
    if not (hasattr(tree.root, "state") and tree.root.state is not None
            and hasattr(tree, "sigma2") and tree.sigma2 is not None):
        return tree

    # initialise the root node
    tree.root.null_state = tree.root.state.copy()

    # preorder simulation
    for node in tree.get_nonterminals(order="preorder"):
        # get state of the current node
        parent_vect = node.null_state

        for child in node.clades:
            # branch length t
            t = child.branch_length if child.branch_length and child.branch_length > 0 else tree.eps

            # Generate the Gaussian noise and project it onto the tangent space
            # sigma2 = diffusion rate; t = time
            sd = np.sqrt(tree.sigma2 * t)
            euclidean_displacement = np.random.normal(loc=0.0, scale=sd, size=parent_vect.shape)

            # Project to Tangent Space
            # v_tan = v_raw - <v_raw, parent_vect> * parent_vect
            # Remove the radial component, keep the component perpendicular to parent_vect
            dot_product = np.dot(euclidean_displacement, parent_vect)
            tangent_vector = euclidean_displacement - dot_product * parent_vect

            # Exponential Map
            # calculate the length of tangent vector (theta), this is the length we're going to move on the hypersphere
            theta = np.linalg.norm(tangent_vector)

            # Avoid dividing by zero
            if theta < 1e-15:
                child.null_state = parent_vect.copy()
            else:
                tangent_direction = tangent_vector / theta

                # move following a great circle:
                # new_vector = old_vector * cos(theta) + tangent_direction * sin(theta)
                # although len(result) in mathematically inherently == 1,
                # re-normalisation is still necessary to avoid float error
                child.null_state = parent_vect * np.cos(theta) + tangent_direction * np.sin(theta)
                child.null_state /= np.linalg.norm(child.null_state)  # Re-normalise to correct drift

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


def compute_spherical_variance(vectors):
    """
    Compute Spherical Variance (1 - resultant_length).
    Input vectors will be normalised internally.
    Range: [0, 1]
    """
    if len(vectors) < 2:
        return 0.0

    arr = np.array(vectors)

    # L2 Normalise input vectors to unit length
    # This eliminates the "shrinkage" artefact completely
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Avoid division by zero for zero-vectors (though rare in ResNet)
    norms[norms < 1e-9] = 1.0
    normalized_arr = arr / norms

    # Calculate Mean Resultant Vector
    mean_vec = np.mean(normalized_arr, axis=0)

    # Resultant Length
    R = np.linalg.norm(mean_vec)

    # Spherical Variance
    return 1.0 - R


def compute_dtt_time_slices(tree, num_slices=None, interval=None):
    """
    Calculate the variance of vectors across different time slices of the tree, all active branches included.
    Interpolation for intermediate nodes, real data for leaf nodes.
    """

    state_key = 'state'

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
        variance = compute_spherical_variance(vectors_per_slice[i])
        results.append((t, variance))

    return max_time, results


# ---------------------------
# Main function
# ---------------------------
MODE_FULL = "full"
MODE_PASSERINES = "passerines"
MODE_NON_PASSERINES = "non_passerines"


def main(tree_file, load_file=None, null_test=False, sample_ratio=100, mode=MODE_FULL):
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

    passerine_species = set()
    if mode in [MODE_PASSERINES, MODE_NON_PASSERINES]:
        passerine_species = get_species_by_group("bird_info.csv", "PASSERIFORMES", level='order')

    output_dir = f'output_{mode}' if not null_test else f'output_{mode}_null'
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()

    progress = get_progress(load_file) if load_file is not None else 0

    with open(tree_file, 'r') as f:
        first_line = f.readline().strip().upper()
        if first_line.startswith("#NEXUS"):
            tree_format = "nexus"
        else:
            tree_format = "newick"

    if null_test:
        rand_num = 1  # fixed number for outlier SC test
        # rand_num = np.random.randint(0, sample_ratio)
    else:
        out_path = f'{output_dir}/disparity_through_time-{mode}_{base_filename}-{start:.0f}.csv'
        out_file = open(out_path, 'a')
        writer = csv.writer(out_file)

    if tree_format == "nexus":
        trees = [Phylo.read(tree_file, "nexus")]
    else:
        trees = open(tree_file, 'r')

    for i, tree_item in enumerate(trees):
        if i < progress:
            continue

        if null_test:
            # choose only 1 / sample_ratio of all trees to process
            if (i + rand_num) % sample_ratio != 0 and tree_format == "newick":
                continue

            out_path = f'{output_dir}/disparity_through_time-{base_filename}-tree{i}-{start:.0f}.csv'
            out_file = open(out_path, 'a')
            writer = csv.writer(out_file)

        if tree_format == "newick":
            tree_item = tree_item.strip()
            tree_item = read_phylogenetic_trees(tree_item)

        if mode == MODE_PASSERINES:
            tree_item = extract_subclade(tree_item, passerine_species)
            if tree_item is None:
                print(f"Tree {i} has insufficient Passerines.")
                continue
        elif mode == MODE_NON_PASSERINES:
            tree_item = prune_subclade(tree_item, passerine_species)
            if tree_item is None:
                print(f"Tree {i} has insufficient Non-Passerines.")
                continue

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

                # construct a trait_map, with only simulated leaf nodes
                simulated_trait_map = {}
                for leaf in tree_item.get_terminals():
                    if hasattr(leaf, 'null_state'):
                        simulated_trait_map[leaf.name] = leaf.null_state

                reconstruct_ancestral_states(tree_item, simulated_trait_map)
                tree_total_time, dtt_results = compute_dtt_time_slices(tree_item, interval=1.0)

                times, variances = zip(*dtt_results)
                writer.writerow([tree_total_time, *variances])
                print(f"[{base_filename}] It took {(time.time() - start):.2f} seconds to process {j + 1} times "
                      f"null test in the {i + 1}th tree!")

        print(f"[{base_filename} - {mode}] It took {(time.time() - start):.2f} seconds to process {i + 1} trees!")

        if null_test:
            out_file.close()

        del tree_item, tree_total_time, dtt_results, times, variances
        gc.collect()

    if tree_format == "newick":
        trees.close()

    if not null_test:
        out_file.close()


if __name__ == "__main__":
    # processes = []
    # for file in os.listdir('CombinedTrees/'):
    #     for mode in [MODE_FULL, MODE_PASSERINES, MODE_NON_PASSERINES]:
    #         p = Process(target=main, args=(f'CombinedTrees/{file}', None, False, 100, mode))
    #         p.start()
    #         processes.append(p)
    #
    # for p in processes:
    #     p.join()
    main("CombinedTrees/Avian-TimeTree.tre", null_test=True)