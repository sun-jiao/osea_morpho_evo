library(ape)

tree_files <- sprintf(
  "./src/CombinedTrees/HackettStage1Full_%d.tre",
  1:10
)

# 读取所有树
tree_list <- lapply(tree_files, read.tree)

# 展平所有树
trees <- do.call(c, tree_list)
class(trees) <- "multiPhylo"

cat("Total trees:", length(trees), "\n")

cons_tree <- consensus(trees, p = 0.5)

write.tree(
  cons_tree,
  "./src/CombinedTrees/HackettStage1Full_consensus.tre"
)