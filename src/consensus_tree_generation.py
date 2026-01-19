import os
import time

import dendropy

trees = dendropy.TreeList()

consensus_trees = dendropy.TreeList()

data_dir = 'CombinedTrees'

for file in os.listdir(data_dir):
    filepath = os.path.join(data_dir, file)
    with open(filepath) as f:
        for line in f:
            trees.append(dendropy.Tree.get(
                data=line,
                schema="newick"))

            print(f'{len(trees)} trees was loaded')

            if len(trees) == 100:
                consensus_tree = trees.consensus(min_freq=0.5)

                consensus_tree.write(
                    path=f"output_consensus/consensus_tree-{time.time()}.tre",
                    schema="newick",
                    suppress_edge_lengths=False,
                    suppress_internal_node_labels=False
                )

                consensus_trees.append(consensus_tree)

                print(f'{len(consensus_trees)} consensus trees were generated')

                trees.clear()

consensus_tree = consensus_trees.consensus(min_freq=0.5)

consensus_tree.write(
    path=f"output_consensus/consensus_tree-{time.time()}.tre",
    schema="newick",
    suppress_edge_lengths=False,
    suppress_internal_node_labels=False
)