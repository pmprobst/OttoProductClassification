# Level 3 - Final Ensemble

source("config.R")
source("03_cross_validation.R")

# Geometric mean function
geometric_mean <- function(predictions, weights = NULL) {
  if (is.null(weights)) {
    weights <- rep(1 / length(predictions), length(predictions))
  }
  
  # Ensure weights sum to 1
  weights <- weights / sum(weights)
  
  # Calculate weighted geometric mean
  n_samples <- nrow(predictions[[1]])
  n_classes <- ncol(predictions[[1]])
  
  result <- matrix(1, nrow = n_samples, ncol = n_classes)
  colnames(result) <- CLASSES
  
  for (i in 1:length(predictions)) {
    # Clip predictions to avoid log(0)
    pred <- pmax(pmin(predictions[[i]], 1 - 1e-15), 1e-15)
    result <- result * (pred ^ weights[i])
  }
  
  # Normalize to sum to 1
  result <- result / rowSums(result)
  
  return(result)
}

# Arithmetic mean function
arithmetic_mean <- function(predictions, weights = NULL) {
  if (is.null(weights)) {
    weights <- rep(1 / length(predictions), length(predictions))
  }
  
  # Ensure weights sum to 1
  weights <- weights / sum(weights)
  
  # Calculate weighted arithmetic mean
  result <- matrix(0, nrow = nrow(predictions[[1]]), ncol = ncol(predictions[[1]]))
  colnames(result) <- CLASSES
  
  for (i in 1:length(predictions)) {
    result <- result + weights[i] * predictions[[i]]
  }
  
  return(result)
}

# Combined ensemble (geometric mean of XGB and NN, then arithmetic with ET)
combined_ensemble <- function(xgb_pred, nn_pred, et_pred, 
                              xgb_weight = 0.65, nn_weight = 0.35,
                              geo_weight = 0.85, et_weight = 0.15) {
  # Geometric mean of XGBoost and Neural Network
  geo_pred <- geometric_mean(list(xgb_pred, nn_pred), 
                            weights = c(xgb_weight, nn_weight))
  
  # Arithmetic mean with Extra Trees
  final_pred <- geo_weight * geo_pred + et_weight * et_pred
  
  # Normalize
  final_pred <- final_pred / rowSums(final_pred)
  
  return(final_pred)
}

# Optimize ensemble weights
optimize_ensemble_weights <- function(level2_predictions, y_true, 
                                     method = "combined") {
  cat("Optimizing ensemble weights...\n")
  
  if (method == "combined") {
    # Try different weight combinations
    best_score <- Inf
    best_weights <- NULL
    
    # Grid search over weight combinations
    xgb_weights <- seq(0.5, 0.8, by = 0.05)
    geo_weights <- seq(0.7, 0.9, by = 0.05)
    
    for (xgb_w in xgb_weights) {
      nn_w <- 1 - xgb_w
      for (geo_w in geo_weights) {
        et_w <- 1 - geo_w
        
        ensemble_pred <- combined_ensemble(
          level2_predictions$xgb,
          level2_predictions$nnet,
          level2_predictions$adaboost_et,
          xgb_weight = xgb_w,
          nn_weight = nn_w,
          geo_weight = geo_w,
          et_weight = et_w
        )
        
        score <- multiclass_logloss(y_true, ensemble_pred)
        
        if (score < best_score) {
          best_score <- score
          best_weights <- list(xgb_weight = xgb_w, nn_weight = nn_w,
                              geo_weight = geo_w, et_weight = et_w)
        }
      }
    }
    
    cat(sprintf("Best ensemble score: %.6f\n", best_score))
    cat(sprintf("Best weights: XGB=%.2f, NN=%.2f, Geo=%.2f, ET=%.2f\n",
                best_weights$xgb_weight, best_weights$nn_weight,
                best_weights$geo_weight, best_weights$et_weight))
    
    return(best_weights)
  } else {
    # Simple arithmetic mean optimization
    weights <- rep(1 / length(level2_predictions), length(level2_predictions))
    # Could implement more sophisticated optimization here
    return(weights)
  }
}

# Create final ensemble prediction
create_final_ensemble <- function(level2_predictions, weights = NULL,
                                 method = "combined") {
  if (method == "combined") {
    if (is.null(weights)) {
      # Default weights from competition description
      weights <- list(xgb_weight = 0.65, nn_weight = 0.35,
                     geo_weight = 0.85, et_weight = 0.15)
    }
    
    final_pred <- combined_ensemble(
      level2_predictions$xgb,
      level2_predictions$nnet,
      level2_predictions$adaboost_et,
      xgb_weight = weights$xgb_weight,
      nn_weight = weights$nn_weight,
      geo_weight = weights$geo_weight,
      et_weight = weights$et_weight
    )
  } else {
    # Simple arithmetic mean
    final_pred <- arithmetic_mean(level2_predictions, weights = weights)
  }
  
  return(final_pred)
}

