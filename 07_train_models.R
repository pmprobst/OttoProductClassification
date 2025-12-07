# Main Training Script

source("config.R")
source("01_load_data.R")
source("02_feature_engineering.R")
source("04_level1_models.R")
source("05_level2_models.R")
source("06_level3_ensemble.R")

cat(paste(rep("=", 50), collapse = ""), "\n")
cat("Otto Product Classification - Training Pipeline\n")
cat(paste(rep("=", 50), collapse = ""), "\n\n")

# Step 1: Load data
cat("Step 1: Loading data...\n")
source("01_load_data.R")
load(file.path(OUTPUT_DIR, "data_prepared.RData"))
cat("Data loaded successfully!\n\n")

# Step 2: Feature Engineering
cat("Step 2: Feature engineering...\n")
features <- engineer_features(
  train_features, 
  train_target, 
  test_features,
  include_tsne = TRUE,  # Set to FALSE for faster initial testing
  include_kmeans = TRUE,
  include_knn_dist = TRUE,
  seed = SEED
)
save(features, file = file.path(FEATURES_DIR, "engineered_features.RData"))
cat("Feature engineering complete!\n\n")

# Step 3: Train Level 1 Models
cat("Step 3: Training Level 1 models...\n")
level1_predictions <- train_level1_models(
  train_features,
  train_target,
  features = features,
  n_folds = N_FOLDS_L1,
  seed = SEED
)
save(level1_predictions, file = file.path(PREDICTIONS_DIR, "level1_predictions.RData"))
cat("Level 1 models trained!\n\n")

# Step 4: Prepare additional features for Level 2
cat("Step 4: Preparing additional features for Level 2...\n")
additional_features <- list()

# Add row statistics
additional_features$row_stats <- features$row_stats_train

# Add t-SNE features if available
if (!is.null(features$tsne_train)) {
  additional_features$tsne <- features$tsne_train
}

# Add K-means features if available
if (!is.null(features$kmeans)) {
  for (kmeans_name in names(features$kmeans)) {
    if (is.factor(features$kmeans[[kmeans_name]])) {
      # Convert factor to one-hot encoding
      kmeans_matrix <- model.matrix(~ 0 + features$kmeans[[kmeans_name]])
      additional_features[[kmeans_name]] <- kmeans_matrix
    }
  }
}

# Add KNN distance features if available
if (!is.null(features$knn_dist)) {
  for (knn_name in names(features$knn_dist)) {
    if (!grepl("_test$", knn_name)) {
      additional_features[[knn_name]] <- features$knn_dist[[knn_name]]
    }
  }
}

# Step 5: Train Level 2 Models
cat("Step 5: Training Level 2 models...\n")
level2_predictions <- train_level2_models(
  level1_predictions,
  train_target,
  additional_features = additional_features,
  n_folds = N_FOLDS_L2,
  seed = SEED
)
save(level2_predictions, file = file.path(PREDICTIONS_DIR, "level2_predictions.RData"))
cat("Level 2 models trained!\n\n")

# Step 6: Optimize and create final ensemble
cat("Step 6: Creating final ensemble...\n")
ensemble_weights <- optimize_ensemble_weights(level2_predictions, train_target, 
                                               method = "combined")
final_ensemble_pred <- create_final_ensemble(level2_predictions, 
                                           weights = ensemble_weights,
                                           method = "combined")

# Calculate final CV score
final_score <- multiclass_logloss(train_target, final_ensemble_pred)
cat(sprintf("\nFinal Ensemble CV Score: %.6f\n", final_score))

# Save ensemble predictions
save(final_ensemble_pred, ensemble_weights, final_score,
     file = file.path(PREDICTIONS_DIR, "final_ensemble.RData"))

cat("\nTraining complete!\n")
cat(sprintf("Final CV Log Loss: %.6f\n", final_score))

