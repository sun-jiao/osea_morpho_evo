library(geomorph)
library(arrow)

signal_result <- readRDS(
  "src/physignal_result-Stiller_pca_100_percent.rds"
)

str(signal_result)

str(signal_result$PACA$rot)
write_feather(signal_result$PACA$rot, "PACA_rot.feather")
