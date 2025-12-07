# Install required R packages for Otto Product Classification

cat("Installing required R packages...\n\n")

required_packages <- c(
  "data.table",    # Fast data loading
  "dplyr",         # Data manipulation
  "caret",         # Cross-validation
  "randomForest",  # Random Forest
  "xgboost",       # XGBoost
  "extraTrees",    # Extra Trees
  "e1071",         # Naive Bayes, SVM
  "nnet",          # Neural Networks
  "class",         # KNN
  "cluster",       # K-means
  "FNN"            # Fast KNN
)

# Check and install missing packages
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    tryCatch({
      install.packages(pkg, dependencies = TRUE, repos = "https://cran.rstudio.com/")
    }, error = function(e) {
      cat(sprintf("Warning: Failed to install %s: %s\n", pkg, e$message))
    })
  } else {
    cat(sprintf("%s is already installed.\n", pkg))
  }
}

# Try to install Rtsne (optional, may require system dependencies)
cat("\nAttempting to install Rtsne (optional, for t-SNE features)...\n")
if (!require("Rtsne", quietly = TRUE)) {
  tryCatch({
    install.packages("Rtsne", dependencies = TRUE, repos = "https://cran.rstudio.com/")
    if (require("Rtsne", quietly = TRUE)) {
      cat("Rtsne installed successfully.\n")
    } else {
      cat("Warning: Rtsne installation may have failed. t-SNE features will be disabled.\n")
      cat("You can install it manually later or skip t-SNE features.\n")
    }
  }, error = function(e) {
    cat("Warning: Could not install Rtsne. t-SNE features will be disabled.\n")
    cat("Error:", e$message, "\n")
    cat("Note: Rtsne may require system dependencies. You can:\n")
    cat("  1. Install XQuartz (https://www.xquartz.org/) on macOS\n")
    cat("  2. Or set include_tsne = FALSE in feature engineering\n")
  })
} else {
  cat("Rtsne is already installed.\n")
}

cat("\nPackage installation complete!\n")
cat("Note: If Rtsne failed to install, you can still run the pipeline\n")
cat("      by setting include_tsne = FALSE in 07_train_models.R\n")

