from ete3 import Tree, TreeStyle, TextFace

tree = Tree("morphology_tree.tre")
n_leaves = len(tree.get_leaves())

# height
img_height = max(400, n_leaves * 15)

for leaf in tree.iter_leaves():
    name_face = TextFace(leaf.name, fsize=24)
    leaf.add_face(name_face, column=0, position="branch-right")

ts = TreeStyle()
ts.show_leaf_name = False
ts.show_scale = False
ts.scale = 200
ts.branch_vertical_margin = 2

tree.render("tree_output.svg", w=1000, h=img_height, tree_style=ts)
