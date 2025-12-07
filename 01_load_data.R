# Load and prepare data for Otto Product Classification

source("config.R")

# Load required libraries
if (!require("data.table")) install.packages("data.table")
if (!require("dplyr")) install.packages("dplyr")
library(data.table)
library(dplyr)

# Load training data
cat("Loading training data...\n")
train <- fread(TRAIN_FILE, stringsAsFactors = FALSE)
cat(sprintf("Training data: %d rows, %d columns\n", nrow(train), ncol(train)))

# Load test data
cat("Loading test data...\n")
test <- fread(TEST_FILE, stringsAsFactors = FALSE)
cat(sprintf("Test data: %d rows, %d columns\n", nrow(test), ncol(test)))

# Extract feature columns
feature_cols <- grep("^feat_", colnames(train), value = TRUE)
cat(sprintf("Number of features: %d\n", length(feature_cols)))

# Prepare training data
train_features <- as.matrix(train[, ..feature_cols])
train_target <- train$target
train_id <- train$id

# Prepare test data
test_features <- as.matrix(test[, ..feature_cols])
test_id <- test$id

# Convert target to factor with all 9 classes
train_target_factor <- factor(train_target, levels = CLASSES)

# Create target matrix for multi-class classification
train_target_matrix <- model.matrix(~ 0 + train_target_factor)
colnames(train_target_matrix) <- CLASSES

# Display class distribution
cat("\nClass distribution:\n")
print(table(train_target))

# Display feature statistics
cat("\nFeature statistics (first 5 features):\n")
print(summary(train_features[, 1:5]))

# Save prepared data
save(train_features, train_target, train_target_factor, train_target_matrix, train_id,
     test_features, test_id, feature_cols,
     file = file.path(OUTPUT_DIR, "data_prepared.RData"))

cat("\nData preparation complete!\n")
cat(sprintf("Saved to: %s\n", file.path(OUTPUT_DIR, "data_prepared.RData")))

