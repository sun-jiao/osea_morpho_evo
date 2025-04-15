from ete3 import Tree

trees = []

with open("CombinedTrees/AllBirdsHackett1.tre") as f:
    for line in f:
        line = line.strip()
        if line:  # ignore empty line
            try:
                t = Tree(line, format=1)
                trees.append(t)
            except Exception as e:
                print("Tree parsing failed: ", e)
            finally:
                break

print([leaf.name for leaf in trees[0].get_leaves()])
