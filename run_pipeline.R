# Otto Product Classification - Main pipeline (stacking ensemble)
# Run from project root: source("run_pipeline.R")
# Writes output/submission.csv

# Load required packages
library(data.table)
library(randomForest)
library(xgboost)
library(nnet)
library(caret)
library(e1071)

# Source pipeline modules (order matters)
source("R/config.R")
source("R/utils.R")
source("R/data.R")
source("R/features.R")
source("R/submission.R")
source("R/models.R")
source("R/ensemble.R")

cat("Otto Product Classification - Stacking Pipeline\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

result <- run_stacking(verbose = TRUE)

cat("\nGenerating submission file...\n")
write_submission(
  normalize_preds(result$test_preds),
  result$test_id,
  file.path(OUTPUT_DIR, "submission.csv")
)
cat(sprintf("Submission saved to: %s\n", file.path(OUTPUT_DIR, "submission.csv")))
cat("Done.\n")
