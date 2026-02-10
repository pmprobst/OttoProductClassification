# Data loading for Otto Product Classification
# Depends: R/config.R

#' Load train and test data from DATA_DIR.
#' Assumes config has been sourced (TRAIN_FILE, TEST_FILE, CLASSES).
#' @return list with X_train, X_test, y_train, y_factor, y_numeric, test_id, feature_cols
load_otto_data <- function() {
  train <- data.table::fread(TRAIN_FILE)
  test <- data.table::fread(TEST_FILE)
  feature_cols <- grep("^feat_", colnames(train), value = TRUE)
  X_train <- as.matrix(train[, ..feature_cols])
  X_test <- as.matrix(test[, ..feature_cols])
  y_train <- train$target
  test_id <- test$id
  y_factor <- factor(y_train, levels = CLASSES)
  y_numeric <- as.numeric(y_factor) - 1L  # 0-indexed for XGBoost
  list(
    X_train = X_train,
    X_test = X_test,
    y_train = y_train,
    y_factor = y_factor,
    y_numeric = y_numeric,
    test_id = test_id,
    feature_cols = feature_cols
  )
}
