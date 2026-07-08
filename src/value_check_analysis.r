library(geomorph)
library(arrow)
library(tibble)

signal_result <- readRDS(
  "src/physignal_result-Stiller_pca_100_percent.rds"
)

str(signal_result)

df <- as.data.frame(signal_result$PACA$rot)
write_feather(df, "src/PACA_rot.feather")

df <- as.data.frame(signal_result$PACA$center)
write_feather(df, "src/PACA_center.feather")


paca_df <- as.data.frame(signal_result$PACA$x)
paca_df <- rownames_to_column(paca_df, var = "Species")

write_feather(paca_df, "src/PACA_x.feather")