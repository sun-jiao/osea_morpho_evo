import pandas as pd
from ete3 import Tree, TextFace, TreeStyle

# Parameters (adjust as needed)
PURE_THRESHOLD = 0.85  # purity threshold to stop further check
MIN_SAMPLE_COUNT = 5  # minimum number of samples in a node to consider early-stopping
LEVEL = "family"


def load_classification_data(csv_file, level):
    # The data was provided by authors of the training dataset and I added the orders and families based on IOC 15.1
    # So the final file is in this order: Vascular name (zh), vascular name (en), scientific name, order, family
    # 非洲鸵鸟,Common Ostrich,Struthio camelus,STRUTHIONIFORMES,Struthionidae
    if level == 'order':
        index = 3
    elif level == 'family':
        index = 4
    else:
        raise ValueError('level must be either "order" or "family"')

    df = pd.read_csv(csv_file, header=None)
    classification_dict = {str(i): df.iloc[i, index] for i in range(len(df))}
    return classification_dict


def assign_classifications(tree, classification_dict):
    for leaf in tree.iter_leaves():
        # In case a leaf name is not found, assign None.
        leaf.add_feature("classification", classification_dict.get(leaf.name.split("_")[0], None))


def compute_purity(node):
    # Purity(node) = (number of leaves of the majority classification) / (total number of leaves)

    # Gather classifications for all leaves under this node
    labels = [leaf.classification for leaf in node.iter_leaves() if leaf.classification is not None]
    total = len(labels)
    if total == 0:
        return 0.0, None, []
    # Count frequencies
    freq = {}
    for lbl in labels:
        freq[lbl] = freq.get(lbl, 0) + 1
    # Determine the majority label
    majority_label = max(freq, key=freq.get)
    purity = freq[majority_label] / total
    # the third return value is a list of (leaf, label) pairs.
    return purity, majority_label, list(zip(node.iter_leaves(), labels))


def check_node(node):
    # Recursively check nodes in the tree from the root to the leaves.

    purity, majority_label, leaves_info = compute_purity(node)
    total_leaves = len(leaves_info)
    # Early stop if node is pure and has enough samples
    if total_leaves >= MIN_SAMPLE_COUNT and purity >= PURE_THRESHOLD:
        print(
            f"Node '{node.name}' (total leaves: {total_leaves}) is pure: purity={purity:.2f}, majority_class={majority_label}")
        # Optionally, you could also annotate the node if you plan to write out the tree.
        node.add_feature("status", "pure")
        node.add_feature("purity", purity)
        node.add_feature("major", majority_label)
        return  # stop further recursion on this branch
    else:
        # Annotate node as "mixed" if not already pure
        node.add_feature("status", "mixed")
        # If node is a leaf or has no children, nothing more to do.
        if node.is_leaf():
            return
        # Recursively check each child
        for child in node.get_children():
            check_node(child)


def mark_outliers(node):
    # identify outlier leaves within a parent node
    if node.status != "pure":
        return
    # Get majority label info
    _, majority_label, leaves_info = compute_purity(node)
    # Print out leaves that do not match the majority
    for leaf, label in leaves_info:
        if label != majority_label:
            # print the info in console and add a mark in its name
            print(f"Leaf '{leaf.name}' in node '{node.name}' is an outlier (label {label} vs majority {majority_label}).")
            if not leaf.name.startswith("🔺"):
                leaf.name = f"🔺{leaf.name}"


def main():
    # Input files: adjust file names as needed
    newick_file = "morphology_cluster.tre"
    bird_info_csv = "bird_info.csv"

    # Load classification data
    classification_dict = load_classification_data(bird_info_csv, LEVEL)

    # Load tree from Newick format
    tree = Tree(newick_file, format=1)  # adjust format if needed

    # Assign classifications to leaves
    assign_classifications(tree, classification_dict)

    # Start the check from the root
    print("Starting hierarchical purity check...")
    check_node(tree)

    # run outlier detection on nodes marked as pure.
    for node in tree.traverse():
        if hasattr(node, "status") and node.status == "pure":
            mark_outliers(node)

    # add labels and save the annotated tree
    for node in tree.traverse():
        if node.is_leaf():
            text_face = TextFace(node.name, fsize=24)
            node.add_face(text_face, column=0, position="branch-right")
        else:
            if hasattr(node, "status") and node.status == "pure":
                text_face = TextFace(f"{node.major}-{node.purity:.2f}", fsize=24)
                node.add_face(text_face, column=1, position="branch-top")

    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.scale = 200
    ts.branch_vertical_margin = 2

    n_leaves = len(tree.get_leaves())

    # height
    img_height = max(400, n_leaves * 15)

    tree.render(f"tree_output_analysed_{LEVEL}.svg", w=700, h=img_height, tree_style=ts)

    tree2 = tree.copy()

    for node in tree2.traverse():
        if not node.is_leaf():
            if hasattr(node, "status") and node.status == "pure":
                node.name = f"{node.major}-{node.purity:.2f}-{len(node.get_leaves())}"
                node.children = []
                text_face = TextFace(node.name, fsize=24)
                node.add_face(text_face, column=0, position="branch-right")
            else:
                text_face = TextFace("clade", fsize=24)
                node.add_face(text_face, column=1, position="branch-top")

    n_leaves = len(tree2.get_leaves())

    # height
    img_height = n_leaves * 15

    tree2.render(f"tree_output_analysed_{LEVEL}_collapsed.svg", w=400, h=img_height, tree_style=ts)

    # tree.write(outfile="annotated_tree.tre")


if __name__ == "__main__":
    main()
