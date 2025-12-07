# Configuration file for Otto Product Classification

# Data paths
DATA_DIR <- "data"
TRAIN_FILE <- file.path(DATA_DIR, "train.csv")
TEST_FILE <- file.path(DATA_DIR, "test.csv")
SAMPLE_SUB_FILE <- file.path(DATA_DIR, "sampleSubmission.csv")

# Output paths
OUTPUT_DIR <- "output"
MODELS_DIR <- file.path(OUTPUT_DIR, "models")
PREDICTIONS_DIR <- file.path(OUTPUT_DIR, "predictions")
FEATURES_DIR <- file.path(OUTPUT_DIR, "features")

# Create directories if they don't exist
dir.create(OUTPUT_DIR, showWarnings = FALSE)
dir.create(MODELS_DIR, showWarnings = FALSE)
dir.create(PREDICTIONS_DIR, showWarnings = FALSE)
dir.create(FEATURES_DIR, showWarnings = FALSE)

# Cross-validation settings
N_FOLDS_L1 <- 5  # Level 1 cross-validation folds
N_FOLDS_L2 <- 4  # Level 2 cross-validation folds
SEED <- 42

# Feature engineering settings
TSNE_DIMS <- 3
KMEANS_CLUSTERS <- c(5, 10, 15, 20, 25, 30, 50)  # Different cluster counts
KNN_NEIGHBORS <- c(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

# Model settings
N_TREES_RF <- 500
N_TREES_XGB <- 300
N_TREES_ET <- 500
MAX_DEPTH <- 6
LEARNING_RATE <- 0.1

# Neural network settings
NN_HIDDEN_LAYERS <- c(100, 50)
NN_EPOCHS <- 50
NN_BATCH_SIZE <- 128
NN_DROPOUT <- 0.3

# Ensemble settings
N_BAGS_XGB <- 30
N_BAGS_NN <- 10
N_BAGS_ADA <- 10

# Class names
CLASSES <- paste0("Class_", 1:9)
N_CLASSES <- length(CLASSES)

