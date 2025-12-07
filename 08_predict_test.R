# Generate Test Predictions and Submission

source("config.R")
source("01_load_data.R")
source("02_feature_engineering.R")
source("04_level1_models.R")
source("05_level2_models.R")
source("06_level3_ensemble.R")

cat(paste(rep("=", 50), collapse = ""), "\n")
cat("Otto Product Classification - Test Prediction\n")
cat(paste(rep("=", 50), collapse = ""), "\n\n")

# Load data
cat("Loading data...\n")
load(file.path(OUTPUT_DIR, "data_prepared.RData"))
load(file.path(FEATURES_DIR, "engineered_features.RData"))
cat("Data loaded!\n\n")

# Load ensemble weights
load(file.path(PREDICTIONS_DIR, "final_ensemble.RData"))

# Train final models on full training set
cat("Training final models on full training set...\n")

# Level 1: Train all models on full training set
cat("Training Level 1 models on full training...\n")
level1_models <- list()

# Random Forest
cat("Training Random Forest...\n")
level1_models$rf <- train_rf(train_features, train_target)

# Logistic Regression
cat("Training Logistic Regression...\n")
level1_models$logreg <- train_logreg(features$log_X_train, train_target)

# Extra Trees
cat("Training Extra Trees...\n")
level1_models$et <- train_et(features$log_X_train, train_target)

# KNN (we'll use k=5 for final prediction)
cat("Training KNN...\n")
level1_models$knn <- train_knn(features$scaled_log_X_train, train_target, k = 5)

# Naive Bayes
cat("Training Naive Bayes...\n")
level1_models$nb <- train_nb(features$log_X_train, train_target)

# XGBoost
cat("Training XGBoost...\n")
level1_models$xgb <- train_xgb(train_features, train_target)

# Neural Network
cat("Training Neural Network...\n")
level1_models$nnet <- train_nnet(features$scaled_log_X_train, train_target)

# Generate Level 1 test predictions
cat("Generating Level 1 test predictions...\n")
level1_test_preds <- list()

level1_test_preds$rf_raw <- predict_rf(level1_models$rf, test_features)
level1_test_preds$logreg_log <- predict_logreg(level1_models$logreg, features$log_X_test)
level1_test_preds$et_log <- predict_et(level1_models$et, features$log_X_test)
level1_test_preds$knn_scaled_log <- predict_knn(level1_models$knn, features$scaled_log_X_test)
level1_test_preds$nb_log <- predict_nb(level1_models$nb, features$log_X_test)
level1_test_preds$xgb_raw <- predict_xgb(level1_models$xgb, test_features)
level1_test_preds$nnet_scaled_log <- predict_nnet(level1_models$nnet, features$scaled_log_X_test)

# Additional KNN models
for (k in c(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)) {
  cat(sprintf("Training KNN k=%d...\n", k))
  knn_model <- train_knn(features$scaled_log_X_train, train_target, k = k)
  level1_test_preds[[paste0("knn_k", k)]] <- predict_knn(knn_model, features$scaled_log_X_test)
}

# Prepare additional features for Level 2
cat("Preparing additional features for Level 2...\n")
additional_test_features <- list()

additional_test_features$row_stats <- features$row_stats_test

if (!is.null(features$tsne_test)) {
  additional_test_features$tsne <- features$tsne_test
}

if (!is.null(features$kmeans)) {
  for (kmeans_name in names(features$kmeans)) {
    if (is.list(features$kmeans[[kmeans_name]])) {
      kmeans_test <- features$kmeans[[kmeans_name]]$test
      kmeans_matrix <- model.matrix(~ 0 + kmeans_test)
      additional_test_features[[kmeans_name]] <- kmeans_matrix
    }
  }
}

if (!is.null(features$knn_dist)) {
  for (knn_name in names(features$knn_dist)) {
    if (grepl("_test$", knn_name)) {
      additional_test_features[[gsub("_test$", "", knn_name)]] <- features$knn_dist[[knn_name]]
    }
  }
}

# Create meta-features for Level 2
X_meta_test <- create_meta_features(level1_test_preds, additional_test_features)

# Train Level 2 models on full training set
cat("Training Level 2 models on full training...\n")

# Create meta-features for training
load(file.path(PREDICTIONS_DIR, "level1_predictions.RData"))
# Load additional features from training
load(file.path(FEATURES_DIR, "engineered_features.RData"))
additional_features <- list()
additional_features$row_stats <- features$row_stats_train
if (!is.null(features$tsne_train)) {
  additional_features$tsne <- features$tsne_train
}
if (!is.null(features$kmeans)) {
  for (kmeans_name in names(features$kmeans)) {
    if (is.factor(features$kmeans[[kmeans_name]])) {
      kmeans_matrix <- model.matrix(~ 0 + features$kmeans[[kmeans_name]])
      additional_features[[kmeans_name]] <- kmeans_matrix
    }
  }
}
if (!is.null(features$knn_dist)) {
  for (knn_name in names(features$knn_dist)) {
    if (!grepl("_test$", knn_name)) {
      additional_features[[knn_name]] <- features$knn_dist[[knn_name]]
    }
  }
}

X_meta_train <- create_meta_features(level1_predictions, additional_features)

# Train Level 2 models
level2_models <- list()
cat("Training Level 2 XGBoost...\n")
level2_models$xgb <- train_xgb_l2(X_meta_train, train_target)

cat("Training Level 2 Neural Network...\n")
level2_models$nnet <- train_nnet_l2(X_meta_train, train_target)

cat("Training Level 2 AdaBoost ET...\n")
level2_models$adaboost_et <- train_adaboost_et(X_meta_train, train_target)

# Generate Level 2 test predictions
cat("Generating Level 2 test predictions...\n")
level2_test_preds <- list()

level2_test_preds$xgb <- predict_xgb_l2(level2_models$xgb, X_meta_test)
level2_test_preds$nnet <- predict_nnet_l2(level2_models$nnet, X_meta_test)
level2_test_preds$adaboost_et <- predict_adaboost_et(level2_models$adaboost_et, X_meta_test)

# Create final ensemble prediction
cat("Creating final ensemble prediction...\n")
final_test_pred <- create_final_ensemble(level2_test_preds, 
                                        weights = ensemble_weights,
                                        method = "combined")

# Create submission file
cat("Creating submission file...\n")
submission <- data.frame(id = test_id)
for (class in CLASSES) {
  submission[[class]] <- final_test_pred[, class]
}

# Write submission
submission_file <- file.path(OUTPUT_DIR, "submission.csv")
write.csv(submission, submission_file, row.names = FALSE)
cat(sprintf("Submission file saved to: %s\n", submission_file))

cat("\nTest prediction complete!\n")

