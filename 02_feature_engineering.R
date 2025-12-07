# Feature Engineering Functions

source("config.R")

# Load required libraries
if (!require("cluster")) install.packages("cluster")
library(cluster)

# Check if Rtsne is available (optional package)
HAS_RTSNE <- require("Rtsne", quietly = TRUE)
if (!HAS_RTSNE) {
  cat("Warning: Rtsne package not available. t-SNE features will be skipped.\n")
  cat("Install with: install.packages('Rtsne')\n")
  cat("Or set include_tsne = FALSE in feature engineering calls.\n")
}

# Function to create log-transformed features
create_log_features <- function(X) {
  log(X + 1)
}

# Function to create sqrt-transformed features
create_sqrt_features <- function(X) {
  sqrt(X + 3/8)
}

# Function to scale features (standardize)
scale_features <- function(X, center = TRUE, scale = TRUE) {
  scale(X, center = center, scale = scale)
}

# Function to create row statistics
create_row_stats <- function(X) {
  row_sums <- rowSums(X)
  row_nonzero <- rowSums(X > 0)
  row_zeros <- rowSums(X == 0)
  row_means <- rowMeans(X)
  row_max <- apply(X, 1, max)
  row_min <- apply(X, 1, min)
  
  cbind(row_sums, row_nonzero, row_zeros, row_means, row_max, row_min)
}

# Function to create t-SNE features
create_tsne_features <- function(X_train, X_test = NULL, dims = TSNE_DIMS, 
                                  perplexity = 30, max_iter = 1000, seed = SEED) {
  if (!HAS_RTSNE) {
    warning("Rtsne package not available. Skipping t-SNE feature creation.")
    return(NULL)
  }
  
  set.seed(seed)
  
  if (is.null(X_test)) {
    # Only training data
    tsne_result <- Rtsne::Rtsne(X_train, dims = dims, perplexity = perplexity, 
                         max_iter = max_iter, verbose = FALSE)
    return(tsne_result$Y)
  } else {
    # Combined approach: fit on train, transform test
    # Note: t-SNE doesn't have a direct transform, so we'll fit on combined data
    # For production, we'd need a different approach or use the train fit
    combined <- rbind(X_train, X_test)
    tsne_result <- Rtsne::Rtsne(combined, dims = dims, perplexity = perplexity,
                         max_iter = max_iter, verbose = FALSE)
    
    train_tsne <- tsne_result$Y[1:nrow(X_train), ]
    test_tsne <- tsne_result$Y[(nrow(X_train)+1):nrow(combined), ]
    
    return(list(train = train_tsne, test = test_tsne))
  }
}

# Function to create K-means clustering features
create_kmeans_features <- function(X_train, X_test = NULL, n_clusters = KMEANS_CLUSTERS, 
                                   seed = SEED) {
  set.seed(seed)
  
  features_list <- list()
  
  for (k in n_clusters) {
    kmeans_result <- kmeans(X_train, centers = k, nstart = 10, iter.max = 100)
    
    if (is.null(X_test)) {
      features_list[[paste0("kmeans_", k)]] <- as.factor(kmeans_result$cluster)
    } else {
      # Predict cluster for test data
      # Using nearest cluster center
      test_clusters <- apply(X_test, 1, function(x) {
        distances <- colSums((t(kmeans_result$centers) - x)^2)
        which.min(distances)
      })
      
      train_cluster <- as.factor(kmeans_result$cluster)
      test_cluster <- as.factor(test_clusters)
      
      features_list[[paste0("kmeans_", k)]] <- list(
        train = train_cluster,
        test = test_cluster
      )
    }
  }
  
  return(features_list)
}

# Function to create KNN distance features to each class
create_knn_distance_features <- function(X_train, y_train, X_test = NULL, 
                                        k_neighbors = c(1, 2, 4), seed = SEED) {
  set.seed(seed)
  
  if (!require("FNN")) install.packages("FNN")
  library(FNN)
  
  # Get unique classes
  classes <- unique(y_train)
  n_classes <- length(classes)
  
  distance_features <- list()
  
  for (k in k_neighbors) {
    # For each class, find distances to k nearest neighbors
    class_distances <- matrix(0, nrow = nrow(X_train), ncol = n_classes)
    colnames(class_distances) <- classes
    
    for (class in classes) {
      class_mask <- y_train == class
      class_data <- X_train[class_mask, , drop = FALSE]
      
      if (nrow(class_data) > 0) {
        # Find k nearest neighbors within the class
        knn_result <- get.knnx(class_data, X_train, k = min(k, nrow(class_data)))
        # Use mean distance to k nearest neighbors
        class_distances[, class] <- rowMeans(knn_result$nn.dist)
      }
    }
    
    distance_features[[paste0("knn_dist_k", k)]] <- class_distances
    
    if (!is.null(X_test)) {
      test_class_distances <- matrix(0, nrow = nrow(X_test), ncol = n_classes)
      colnames(test_class_distances) <- classes
      
      for (class in classes) {
        class_mask <- y_train == class
        class_data <- X_train[class_mask, , drop = FALSE]
        
        if (nrow(class_data) > 0) {
          knn_result <- get.knnx(class_data, X_test, k = min(k, nrow(class_data)))
          test_class_distances[, class] <- rowMeans(knn_result$nn.dist)
        }
      }
      
      distance_features[[paste0("knn_dist_k", k, "_test")]] <- test_class_distances
    }
  }
  
  return(distance_features)
}

# Main feature engineering function
engineer_features <- function(X_train, y_train = NULL, X_test = NULL, 
                              include_tsne = TRUE, include_kmeans = TRUE,
                              include_knn_dist = TRUE, seed = SEED) {
  set.seed(seed)
  
  cat("Starting feature engineering...\n")
  
  features <- list()
  
  # Basic transformations
  cat("Creating log features...\n")
  features$log_X_train <- create_log_features(X_train)
  if (!is.null(X_test)) {
    features$log_X_test <- create_log_features(X_test)
  }
  
  cat("Creating sqrt features...\n")
  features$sqrt_X_train <- create_sqrt_features(X_train)
  if (!is.null(X_test)) {
    features$sqrt_X_test <- create_sqrt_features(X_test)
  }
  
  cat("Creating scaled features...\n")
  features$scaled_X_train <- scale_features(X_train)
  if (!is.null(X_test)) {
    # Scale test using training statistics
    train_mean <- attr(features$scaled_X_train, "scaled:center")
    train_sd <- attr(features$scaled_X_train, "scaled:scale")
    features$scaled_X_test <- scale(X_test, center = train_mean, scale = train_sd)
  }
  
  cat("Creating scaled log features...\n")
  log_X_train <- features$log_X_train
  features$scaled_log_X_train <- scale_features(log_X_train)
  if (!is.null(X_test)) {
    log_X_test <- features$log_X_test
    train_mean <- attr(features$scaled_log_X_train, "scaled:center")
    train_sd <- attr(features$scaled_log_X_train, "scaled:scale")
    features$scaled_log_X_test <- scale(log_X_test, center = train_mean, scale = train_sd)
  }
  
  cat("Creating row statistics...\n")
  features$row_stats_train <- create_row_stats(X_train)
  if (!is.null(X_test)) {
    features$row_stats_test <- create_row_stats(X_test)
  }
  
  # Advanced features
  if (include_tsne && !is.null(y_train) && HAS_RTSNE) {
    cat("Creating t-SNE features (this may take a while)...\n")
    tsne_result <- create_tsne_features(X_train, X_test, dims = TSNE_DIMS, seed = seed)
    if (!is.null(tsne_result)) {
      if (is.list(tsne_result)) {
        features$tsne_train <- tsne_result$train
        features$tsne_test <- tsne_result$test
      } else {
        features$tsne_train <- tsne_result
      }
    }
  } else if (include_tsne && !HAS_RTSNE) {
    cat("Skipping t-SNE features (Rtsne package not available)...\n")
  }
  
  if (include_kmeans && !is.null(y_train)) {
    cat("Creating K-means features...\n")
    kmeans_features <- create_kmeans_features(X_train, X_test, seed = seed)
    features$kmeans <- kmeans_features
  }
  
  if (include_knn_dist && !is.null(y_train)) {
    cat("Creating KNN distance features...\n")
    knn_dist_features <- create_knn_distance_features(X_train, y_train, X_test, 
                                                      k_neighbors = c(1, 2, 4), seed = seed)
    features$knn_dist <- knn_dist_features
  }
  
  cat("Feature engineering complete!\n")
  return(features)
}

