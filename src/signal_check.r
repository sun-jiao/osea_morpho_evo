library(ape)
library(geomorph)
library(picante)
library(readr)
############################################################
# create_trait_mapping
############################################################

create_trait_mapping <- function(name_match_df, weights_df) {
  trait_map <- list()

  for (i in seq_len(nrow(name_match_df))) {

    label <- as.character(name_match_df[i, 1][[1]])
    index <- as.integer(name_match_df[i, 3][[1]])

    if (!is.na(index) && index != -1) {

      # Python dataframe.iloc 从0开始
      r_index <- index + 1

      if (r_index <= nrow(weights_df)) {

        trait_map[[label]] <- as.numeric(weights_df[r_index, ])

      } else {

        warning(
          sprintf(
            "Index %d is out of range for label %s",
            index,
            label
          )
        )

      }
    }
  }

  return(trait_map)
}

############################################################
# load tree
############################################################

load_tree <- function(tree_file) {
  first_line <- toupper(readLines(tree_file, n = 1))

  if (startsWith(first_line, "#NEXUS")) {
    tree <- read.nexus(tree_file)
  } else {
    tree <- read.tree(tree_file)
  }

  if (inherits(tree, "multiPhylo")) {
    tree <- tree[[1]]
  }

  if (any(duplicated(tree$tip.label))) {
    stop("Tree contains duplicated tip labels.")
  }

  tree <- multi2di(tree, random = TRUE)

  tree$edge.length[tree$edge.length == 0] <- 1e-6

  return(tree)
}

############################################################
# build trait matrix
############################################################

build_trait_matrix <- function(tree, trait_map) {

  species_in_tree <- tree$tip.label

  common_species <- intersect(
    species_in_tree,
    names(trait_map)
  )

  if (length(common_species) < 3) {
    stop(sprintf("Too few matched species: %d.", length(common_species)))
  }

  removed_species <- setdiff(species_in_tree, common_species)
  
  if (length(removed_species) > 0) {
    cat("Removed species list:\n")
    print(removed_species) # 如果物种太多，可以用 cat(paste(removed_species, collapse = ", "), "\n") 一行打印
  } else {
    cat("No species removed from the tree.\n")
  }

  tree_pruned <- keep.tip(tree, common_species)

  trait_matrix <- do.call(
    rbind,
    lapply(tree_pruned$tip.label,
           function(x) trait_map[[x]])
  )

  rownames(trait_matrix) <- tree_pruned$tip.label

  list(
    tree = tree_pruned,
    traits = trait_matrix
  )
}

############################################################
# run multivariate Blomberg's K
############################################################

run_multivariate_K <- function(tree_file,
                               name_match_file,
                               weights_file,
                               iter = 999,
                               manual_permutation_loop = FALSE,
                               sdr_suffix = NULL) {

  cat("Reading tree...\n")

  tree <- load_tree(tree_file)

  cat("Reading mapping table...\n")

  name_match <- read_csv(
    name_match_file,
    col_names = FALSE,
    show_col_types = FALSE
  )

  cat("Reading vectors...\n")

  weights <- read_csv(
    weights_file,
    col_names = FALSE,
    show_col_types = FALSE
  )

  trait_map <- create_trait_mapping(
    name_match,
    weights
  )

  dat <- build_trait_matrix(
    tree,
    trait_map
  )

  cat("Matched species:",
      nrow(dat$traits),
      "\n")

  cat("Trait dimensions:",
      ncol(dat$traits),
      "\n")

  cat("Running Multivariate Blomberg's K...\n")

  signal_result <- physignal(
    A = dat$traits,
    phy = dat$tree,
    iter = iter,
    seed = 42,
    print.progress = TRUE
  )

  if (is.null(sdr_suffix)) {
    sdr_suffix <- format(Sys.time(), "%Y%m%d%H%M%S")
  }

  saveRDS(
    signal_result,
    file = sprintf("src/physignal_result-%s.rds", sdr_suffix)
  )
  
  str(signal_result)
  
  cat("\n")
  cat("=====================================\n")
  cat("Multivariate Blomberg's K\n")
  cat("=====================================\n")
  cat("K_mult =", signal_result$phy.signal, "\n")
  cat("P-value =", signal_result$pvalue, "\n")
  cat("K.by.p =", signal_result$K.by.p, "\n")
  cat("=====================================\n")

  ############################################################
  # Plot K values for each PCA axis
  ############################################################

  k_by_axis <- signal_result$K.by.p

  k_df <- data.frame(
    Axis = seq_along(k_by_axis),
    K = as.numeric(k_by_axis)
  )
  
  pdf("src/Blomberg_K_by_PACA_axis_numbers.pdf", width = 8, height = 5)
  
  plot(
    k_df$Axis,
    k_df$K,
    type = "l",
    lwd = 2,
    xlab = "PACA Axis",
    ylab = "Blomberg's K",
    main = "Phylogenetic Signal per PACA Axis (K by axis)"
  )

  points(k_df$Axis, k_df$K, pch = 16, cex = 0.5)
  abline(h = 1, col = "red", lty = 2)
  abline(h = 0, col = "grey", lty = 3)

  dev.off()

  if (manual_permutation_loop) {

    valid_indices_empirical <- which(signal_result$K.by.p > 1)
    
    if (length(valid_indices_empirical) == 0) {
      stop("Empirical K.by.p never exceeds 1. Cannot define target_idx.")
    }
    # 
    target_idx <- max(valid_indices_empirical)

    empirical_K_at_target <- signal_result$K.by.p[target_idx]

    empirical_K_paca_1 <- signal_result$K.by.p[1]

    cat("=====================================\n")
    cat("Manual permutation loop Multivariate Blomberg's K\n")
    cat("=====================================\n")
    cat(sprintf("Empirical K.by.p[%d] = %.4f\n", target_idx, empirical_K_at_target))

    random_K_values   <- numeric(iter)   # overall K_mult per permutation
    K_at_target_values <- numeric(iter)  # K.by.p[target_idx] per permutation
    K_paca_1_values <- numeric(iter)  # K.by.p[1] per permutation

    set.seed(42)  # For reproducibility

    for (i in 1:iter) {

      if (i %% 100 == 0) {
        cat(sprintf("Permutation %d/%d\n", i, iter))
      }

      # Break tip-trait association while keeping tree structure and trait
      # matrix itself fixed — random relabelling of which trait vector belongs
      # to which tip.
      permuted_traits <- dat$traits
      rownames(permuted_traits) <- sample(rownames(dat$traits))

      permuted_result <- physignal(
        A = permuted_traits,
        phy = dat$tree,
        iter = 0
      )

      random_K_values[i] <- permuted_result$phy.signal

      K_paca_1_values[i] <- permuted_result$K.by.p[1]
      K_at_target_values[i] <- permuted_result$K.by.p[target_idx]
    }

    # One-sided permutation p-value: how often does the null produce a
    # cumulative-K-at-dimension-16 at least as extreme (large) as observed?
    # NAs are excluded from the comparison but reported, rather than silently
    # folded into na.rm without comment.
    prop_as_extreme <- (sum(K_at_target_values >= empirical_K_at_target) + 1 ) / (length(K_at_target_values) + 1)
    prop_as_extreme_paca_1 <- (sum(K_paca_1_values >= empirical_K_paca_1) + 1 ) / (length(K_paca_1_values) + 1)

    cat("\n")
    cat("=====================================\n")
    cat("Summary of manual permutation loop\n")
    cat("=====================================\n")
    cat(sprintf("target_idx      = %d\n", target_idx))
    cat(sprintf("Empirical K.by.p[%d]       = %.4f\n", target_idx, empirical_K_at_target))
    cat(sprintf("Empirical K.by.p[1]       = %.4f\n", empirical_K_paca_1))
    cat(sprintf("Empirical overall K_mult         = %.4f\n", signal_result$phy.signal))
    cat(sprintf("Mean permuted K.by.p[%d]   = %.4f\n", target_idx, mean(K_at_target_values, na.rm = TRUE)))
    cat(sprintf("Mean permuted K.by.p[1]   = %.4f\n", mean(K_paca_1_values, na.rm = TRUE)))
    cat(sprintf("Mean permuted overall K_mult        = %.4f\n", mean(random_K_values)))
    cat(sprintf("Max permuted K.by.p[PACA_1:%d]    = %.4f\n", target_idx, max(K_at_target_values, na.rm = TRUE)))
    cat(sprintf("Max permuted K.by.p[1]    = %.4f\n", max(K_paca_1_values, na.rm = TRUE)))
    cat(sprintf("Max permuted overall K_mult         = %.4f\n", max(random_K_values, na.rm = TRUE)))
    cat(sprintf("p-value (%d) = %.5f \n", target_idx, prop_as_extreme))
    cat(sprintf("p-value (PACA_1) = %.5f \n", prop_as_extreme_paca_1))
    cat("=====================================\n")

    invisible(list(
      target_idx = target_idx,
      empirical_K_at_target = empirical_K_at_target,
      random_K_values = random_K_values,
      K_at_target_values = K_at_target_values,
      pvalue = prop_as_extreme
    ))
  }
}

############################################################
# run univariate Blomberg's K for each axis
############################################################

run_univariate_K <- function(tree_file,
                                  name_match_file,
                                  weights_file,
                                  iter = 999) {
  
  cat("Reading tree...\n")
  tree <- load_tree(tree_file)
  
  cat("Reading mapping table...\n")
  name_match <- read_csv(name_match_file, show_col_types = FALSE)
  
  cat("Reading PCA vectors...\n")
  weights <- read_csv(weights_file, show_col_types = FALSE)
  
  trait_map <- create_trait_mapping(name_match, weights)
  dat <- build_trait_matrix(tree, trait_map)
  
  n_species <- nrow(dat$traits)
  n_axes <- ncol(dat$traits)
  
  cat("Matched species:", n_species, "\n")
  cat("Trait dimensions (PCA axes):", n_axes, "\n")
  cat("Running Univariate Blomberg's K for each axis...\n")
  
  k_values <- numeric(n_axes)
  p_values <- numeric(n_axes)
  
  for (i in seq_len(n_axes)) {
    single_trait <- as.numeric(dat$traits[, i])
    names(single_trait) <- rownames(dat$traits)
    
    res <- phylosignal(x = single_trait, phy = dat$tree, reps = iter)
    
    k_values[i] <- res$K
    p_values[i] <- res$PIC.variance.P
    
    if (i %% 20 == 0 || i == n_axes) {
      cat(sprintf("Progress: [%d/%d] axes calculated...\n", i, n_axes))
    }
  }
  
  K_table <- data.frame(
    Axis = paste0("PC", seq_len(n_axes)),
    K = k_values,
    P_value = p_values
  )
  
  cat("Univariate analysis completed.\n")

  ############################################################
  # plot 1: Blomberg's K 
  ############################################################
  pdf("src/Blomberg_K_by_PCA_axis.pdf", width = 8, height = 5)
  
  x_coords <- seq_along(K_table$Axis)
  
  plot(
    x_coords,             # 使用数字作为 X 轴坐标
    K_table$K,
    type = "l",
    lwd = 2,
    xaxt = "n",           # 先隐藏默认的 X 轴刻度标签，后面手动加
    xlab = "PCA Axis",
    ylab = "Blomberg's K",
    main = "Phylogenetic Signal per PCA Axis (K by axis)"
  )
  
  # 手动添加 X 轴标签（因为有 157 个轴，全部显示会挤在一起，这里每隔 20 个轴显示一个标签）
  axis_ticks <- seq(1, length(x_coords), by = 20)
  axis(1, at = axis_ticks, labels = K_table$Axis[axis_ticks])

  points(x_coords, K_table$K, pch = 16, cex = 0.5)
  abline(h = 1, col = "red", lty = 2)
  abline(h = 0, col = "grey", lty = 3)

  dev.off()

  ############################################################
  # 绘图 2: P 值
  ############################################################
  pdf("src/Blomberg_K_pvalues_by_PCA_axis.pdf", width = 8, height = 5)
  
  plot(
    x_coords,             # 使用数字作为 X 轴坐标
    K_table$P_value,
    type = "l",
    lwd = 2,
    xaxt = "n",           # 同样隐藏默认 X 轴
    xlab = "PCA Axis",
    ylab = "p value",
    main = "P-values of Phylogenetic Signal per PCA Axis"
  )
  
  # 手动添加 X 轴标签
  axis(1, at = axis_ticks, labels = K_table$Axis[axis_ticks])

  points(x_coords, K_table$P_value, pch = 16, cex = 0.5)
  
  # 对于 P 值图，建议把红虚线加在 0.05 的位置，更具生物学统计意义
  abline(h = 0.05, col = "red", lty = 2) 
  abline(h = 0, col = "grey", lty = 3)

  dev.off()
  
  return(K_table)
}

# K_table <- run_univariate_K(
#   tree_file = "src/CombinedTrees/Avian-TimeTree.tre",
#   name_match_file = "src/avian_timetree_name_match.csv",
#   weights_file = "src/weights.csv"
# )

# write.csv(
#   K_table,
#   "src/PCA_axis_phylogenetic_signal.csv",
#   row.names = FALSE
# )

# print(K_table)


# result <- run_multivariate_K(
#   tree_file = "src/CombinedTrees/Avian-TimeTree.tre",
#   name_match_file = "src/avian_timetree_name_match.csv",
#   weights_file = "src/all_weights.csv",
#   iter = 999,
#   manual_permutation_loop = TRUE,
#   sdr_suffix = "Stiller_all_weights"
# )

# result <- run_multivariate_K(
#   tree_file = "src/CombinedTrees/Avian-TimeTree.tre",
#   name_match_file = "src/avian_timetree_name_match.csv",
#   weights_file = "src/pca_weights.csv",
#   iter = 999,
#   manual_permutation_loop = TRUE,
#   sdr_suffix = "Stiller_weights"
# )

result <- run_multivariate_K(
  tree_file = "src/CombinedTrees/Avian-TimeTree.tre",
  name_match_file = "src/avian_timetree_name_match_trimmed.csv",
  weights_file = "src/avian_timetree_one_fifth_species_pca_weights.csv",
  iter = 999,
  manual_permutation_loop = TRUE,
  sdr_suffix = "Stiller_pca_40_dims"
)

result <- run_multivariate_K(
  tree_file = "src/CombinedTrees/Avian-TimeTree.tre",
  name_match_file = "src/avian_timetree_name_match_trimmed.csv",
  weights_file = "src/avian_timetree_pca_weights.csv",
  iter = 999,
  manual_permutation_loop = TRUE,
  sdr_suffix = "Stiller_pca_80_percent"
)

result <- run_multivariate_K(
  tree_file = "src/CombinedTrees/Avian-TimeTree.tre",
  name_match_file = "src/avian_timetree_name_match_trimmed.csv",
  weights_file = "src/avian_timetree_pca_weights_95.csv",
  iter = 999,
  manual_permutation_loop = TRUE,
  sdr_suffix = "Stiller_pca_95_percent"
)

result <- run_multivariate_K(
  tree_file = "src/CombinedTrees/Avian-TimeTree.tre",
  name_match_file = "src/avian_timetree_name_match_trimmed.csv",
  weights_file = "src/avian_timetree_pca_weights_100.csv",
  iter = 999,
  manual_permutation_loop = TRUE,
  sdr_suffix = "Stiller_pca_100_percent"
)

# result <- run_multivariate_K(
#   tree_file = "src/CombinedTrees/HackettStage1Full_5.tre",
#   name_match_file = "src/birdtree_name_match.csv",
#   weights_file = "src/all_weights.csv",
#   iter = 1,
#   manual_permutation_loop = FALSE,
#   sdr_suffix = "Hackett_all_weights"
# )
