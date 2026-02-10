# Configuration for Otto Product Classification pipeline
# Source from project root: source("R/config.R")

DATA_DIR <- "data"
TRAIN_FILE <- file.path(DATA_DIR, "train.csv")
TEST_FILE <- file.path(DATA_DIR, "test.csv")
SAMPLE_SUB_FILE <- file.path(DATA_DIR, "sampleSubmission.csv")
OUTPUT_DIR <- "output"

# Cross-validation
N_FOLDS <- 5
SEED <- 42

# Class names
CLASSES <- paste0("Class_", 1:9)
N_CLASSES <- 9

# Ensure output directory exists
dir.create(OUTPUT_DIR, showWarnings = FALSE)
