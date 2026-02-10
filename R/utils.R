# Shared utilities: evaluation and prediction normalization
# Depends: R/config.R (for CLASSES)

#' Multi-class log loss
#' @param y_true character or factor, true labels (e.g. Class_1, ...)
#' @param y_pred matrix, predicted class probabilities (rows = samples, cols = classes)
logloss <- function(y_true, y_pred) {
  y_pred <- y_pred / rowSums(y_pred)
  y_pred <- pmax(pmin(y_pred, 1 - 1e-15), 1e-15)
  y_true_num <- as.numeric(factor(y_true, levels = CLASSES))
  -mean(log(y_pred[cbind(seq_len(nrow(y_pred)), y_true_num)]))
}

#' Normalize predictions to sum to 1 per row (valid probability distribution)
normalize_preds <- function(preds) {
  preds <- pmax(preds, 0)
  preds / rowSums(preds)
}
