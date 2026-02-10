# Otto Product Classification - Single models (no stacking)
# Run from project root: source("run_single_models.R")
# Trains RF, Multinomial Logistic, Naive Bayes, XGBoost (early stopping) and writes
# output/submission_<model>.csv for each.

library(data.table)
library(randomForest)
library(xgboost)
library(nnet)
library(caret)
library(e1071)

source("R/config.R")
source("R/utils.R")
source("R/data.R")
source("R/features.R")
source("R/submission.R")
source("R/models.R")

cat("Otto Product Classification - Single Models\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

d <- load_otto_data()
f_simple <- build_features(d$X_train, d$X_test, "simple")
set.seed(SEED)
folds <- caret::createFolds(d$y_train, k = N_FOLDS, list = TRUE)

cat(sprintf("Training: %d samples, %d features. Test: %d samples.\n\n",
  nrow(d$X_train), ncol(d$X_train), nrow(d$X_test)))

# Random Forest (raw)
cat("Random Forest...\n")
rf <- train_rf_cv(d$X_train, d$y_factor, d$X_test, folds)
cat(sprintf("  Log Loss: %.4f\n", logloss(d$y_train, rf$oof)))
write_submission(rf$test_preds, d$test_id, file.path(OUTPUT_DIR, "submission_random_forest.csv"))
cat(sprintf("  Saved: %s\n\n", file.path(OUTPUT_DIR, "submission_random_forest.csv")))

# Multinomial Logistic (log features)
cat("Multinomial Logistic Regression...\n")
mlr <- train_mlr_cv(f_simple$log_X_train, d$y_factor, f_simple$log_X_test, folds)
cat(sprintf("  Log Loss: %.4f\n", logloss(d$y_train, mlr$oof)))
write_submission(mlr$test_preds, d$test_id, file.path(OUTPUT_DIR, "submission_multinomial_logistic.csv"))
cat(sprintf("  Saved: %s\n\n", file.path(OUTPUT_DIR, "submission_multinomial_logistic.csv")))

# Naive Bayes (log features)
cat("Naive Bayes...\n")
nb <- train_nb_cv(f_simple$log_X_train, d$y_factor, f_simple$log_X_test, folds)
cat(sprintf("  Log Loss: %.4f\n", logloss(d$y_train, nb$oof)))
write_submission(nb$test_preds, d$test_id, file.path(OUTPUT_DIR, "submission_naive_bayes.csv"))
cat(sprintf("  Saved: %s\n\n", file.path(OUTPUT_DIR, "submission_naive_bayes.csv")))

# XGBoost with early stopping (raw)
cat("XGBoost (early stopping)...\n")
xgb <- train_xgb_early_cv(d$X_train, d$y_numeric, d$X_test, folds)
cat(sprintf("  Log Loss: %.4f\n", logloss(d$y_train, xgb$oof)))
write_submission(xgb$test_preds, d$test_id, file.path(OUTPUT_DIR, "submission_xgboost_early_stopping.csv"))
cat(sprintf("  Saved: %s\n\n", file.path(OUTPUT_DIR, "submission_xgboost_early_stopping.csv")))

cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Summary:\n")
cat(sprintf("  Random Forest:              %.4f\n", logloss(d$y_train, rf$oof)))
cat(sprintf("  Multinomial Logistic:       %.4f\n", logloss(d$y_train, mlr$oof)))
cat(sprintf("  Naive Bayes:               %.4f\n", logloss(d$y_train, nb$oof)))
cat(sprintf("  XGBoost (early stop):       %.4f\n", logloss(d$y_train, xgb$oof)))
cat("Done.\n")
