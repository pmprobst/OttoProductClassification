# Feature engineering for Otto Product Classification
# Depends: none (pure function of matrices)

#' Build feature transformations from raw feature matrices.
#' @param X_train matrix, training features
#' @param X_test matrix, test features
#' @param type "full" = log, sqrt, scaled, scaled_log, row_stats (for stacking);
#'             "simple" = log only (for single models that only need log)
#' @return list with requested matrices (e.g. log_X_train, log_X_test, row_stats_train, ...)
build_features <- function(X_train, X_test, type = c("full", "simple")) {
  type <- match.arg(type)
  out <- list()
  # Log (used by both)
  out$log_X_train <- log(X_train + 1)
  out$log_X_test <- log(X_test + 1)
  if (type == "simple") return(out)
  # Sqrt
  out$sqrt_X_train <- sqrt(X_train + 3/8)
  out$sqrt_X_test <- sqrt(X_test + 3/8)
  # Scaled
  out$scaled_X_train <- scale(X_train)
  out$scaled_X_test <- scale(X_test,
    center = attr(out$scaled_X_train, "scaled:center"),
    scale = attr(out$scaled_X_train, "scaled:scale"))
  # Scaled log (apply train center/scale to test)
  out$scaled_log_X_train <- scale(out$log_X_train)
  out$scaled_log_X_test <- scale(out$log_X_test,
    center = attr(out$scaled_log_X_train, "scaled:center"),
    scale = attr(out$scaled_log_X_train, "scaled:scale"))
  # Row statistics
  out$row_stats_train <- cbind(
    rowSums(X_train),
    rowSums(X_train > 0),
    rowSums(X_train == 0),
    rowMeans(X_train),
    apply(X_train, 1, max),
    apply(X_train, 1, min)
  )
  out$row_stats_test <- cbind(
    rowSums(X_test),
    rowSums(X_test > 0),
    rowSums(X_test == 0),
    rowMeans(X_test),
    apply(X_test, 1, max),
    apply(X_test, 1, min)
  )
  out
}
