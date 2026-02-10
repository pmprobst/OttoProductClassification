# Stacking ensemble: Level 1 base models + Level 2 meta-model
# Depends: R/config.R, R/utils.R, R/data.R, R/features.R, R/models.R
# Call after sourcing config, utils, data, features, models and loading packages.

#' Run full stacking pipeline. Uses global config (N_FOLDS, SEED, CLASSES, etc.).
#' Expects: d = load_otto_data(), f = build_features(d$X_train, d$X_test, "full"), folds already created.
#' @return list with test_preds (matrix), cv_score (numeric), level1_scores (named numeric)
run_stacking <- function(verbose = TRUE) {
  if (verbose) {
    cat("Loading data...\n")
  }
  d <- load_otto_data()
  set.seed(SEED)
  folds <- caret::createFolds(d$y_train, k = N_FOLDS, list = TRUE)
  if (verbose) {
    cat(sprintf("Training: %d samples, %d features\n", nrow(d$X_train), ncol(d$X_train)))
    cat(sprintf("Test: %d samples\n\n", nrow(d$X_test)))
    cat("Creating feature transformations...\n")
  }
  f <- build_features(d$X_train, d$X_test, "full")
  if (verbose) cat("Feature engineering complete.\n\n")

  # Level 1: base models
  level1 <- list()
  if (verbose) cat(paste(rep("=", 60), collapse = ""), "\nLEVEL 1: Training Base Models\n", paste(rep("=", 60), collapse = ""), "\n\n")

  if (verbose) cat("Model 1: Random Forest (raw)...\n")
  level1$rf <- train_rf_cv(d$X_train, d$y_factor, d$X_test, folds)
  if (verbose) cat(sprintf("  RF Log Loss: %.4f\n\n", logloss(d$y_train, level1$rf$oof)))

  if (verbose) cat("Model 2: XGBoost (raw, 300 rounds)...\n")
  level1$xgb_raw <- train_xgb_raw_cv(d$X_train, d$y_numeric, d$X_test, folds)
  if (verbose) cat(sprintf("  XGB Raw Log Loss: %.4f\n\n", logloss(d$y_train, level1$xgb_raw$oof)))

  if (verbose) cat("Model 3: XGBoost (log, 300 rounds)...\n")
  level1$xgb_log <- train_xgb_log_cv(f$log_X_train, d$y_numeric, f$log_X_test, folds)
  if (verbose) cat(sprintf("  XGB Log Log Loss: %.4f\n\n", logloss(d$y_train, level1$xgb_log$oof)))

  if (verbose) cat("Model 4: Neural Network (scaled log)...\n")
  level1$nn <- train_nn_cv(f$scaled_log_X_train, d$y_factor, f$scaled_log_X_test, folds)
  if (verbose) cat(sprintf("  NN Log Loss: %.4f\n\n", logloss(d$y_train, level1$nn$oof)))

  if (verbose) cat("Model 5: Naive Bayes (log)...\n")
  level1$nb <- train_nb_cv(f$log_X_train, d$y_factor, f$log_X_test, folds)
  if (verbose) cat(sprintf("  NB Log Loss: %.4f\n\n", logloss(d$y_train, level1$nb$oof)))

  # Average Level 1
  n_l1 <- length(level1)
  L1_train <- matrix(0, nrow = nrow(d$X_train), ncol = N_CLASSES)
  L1_test <- matrix(0, nrow = nrow(d$X_test), ncol = N_CLASSES)
  colnames(L1_train) <- CLASSES
  colnames(L1_test) <- CLASSES
  for (m in level1) {
    L1_train <- L1_train + m$oof
    L1_test <- L1_test + m$test_preds
  }
  L1_train <- normalize_preds(L1_train / n_l1)
  L1_test <- normalize_preds(L1_test / n_l1)
  if (verbose) cat(sprintf("Level 1 average Log Loss: %.4f\n\n", logloss(d$y_train, L1_train)))

  # Level 2: meta-model (L1 predictions + row stats)
  if (verbose) cat(paste(rep("=", 60), collapse = ""), "\nLEVEL 2: Training Meta-Model\n", paste(rep("=", 60), collapse = ""), "\n\n")
  meta_train <- cbind(L1_train, f$row_stats_train)
  meta_test <- cbind(L1_test, f$row_stats_test)
  level2 <- train_meta_cv(meta_train, d$y_numeric, meta_test, folds)
  cv_score <- logloss(d$y_train, level2$oof)
  if (verbose) cat(sprintf("Final stacked model Log Loss: %.4f\n", cv_score))

  level1_scores <- vapply(level1, function(m) logloss(d$y_train, m$oof), numeric(1))
  list(
    test_preds = level2$test_preds,
    cv_score = cv_score,
    level1_scores = level1_scores,
    test_id = d$test_id
  )
}
