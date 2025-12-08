# Reformat DataRobot predictions to Kaggle submission format
# Converts: target_Class_X_PREDICTION columns -> Class_X columns
# Ensures proper format: id, Class_1, Class_2, ..., Class_9

library(data.table)

# File paths
DATA_DIR <- "data"
INPUT_FILE <- "datarobotresultslightgradientbosted.csv"
OUTPUT_DIR <- "output"
OUTPUT_FILE <- file.path(OUTPUT_DIR, "datarobot_submission.csv")
SAMPLE_SUB_FILE <- file.path(DATA_DIR, "sampleSubmission.csv")

# Create output directory
dir.create(OUTPUT_DIR, showWarnings = FALSE)

cat("Loading DataRobot predictions...\n")
datarobot <- fread(INPUT_FILE)

cat(sprintf("Loaded %d rows, %d columns\n", nrow(datarobot), ncol(datarobot)))

# Extract prediction columns (Class_1 through Class_9)
pred_cols <- paste0("target_Class_", 1:9, "_PREDICTION")
cat("Prediction columns:\n")
print(pred_cols)

# Check if all prediction columns exist
missing_cols <- setdiff(pred_cols, colnames(datarobot))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing columns: %s\n", paste(missing_cols, collapse = ", ")))
}

# Extract id column
if (!"id" %in% colnames(datarobot)) {
  stop("Error: 'id' column not found in DataRobot file\n")
}

# Create submission data frame
cat("\nCreating submission format...\n")
submission <- data.frame(
  id = datarobot$id,
  Class_1 = as.numeric(datarobot[[pred_cols[1]]]),
  Class_2 = as.numeric(datarobot[[pred_cols[2]]]),
  Class_3 = as.numeric(datarobot[[pred_cols[3]]]),
  Class_4 = as.numeric(datarobot[[pred_cols[4]]]),
  Class_5 = as.numeric(datarobot[[pred_cols[5]]]),
  Class_6 = as.numeric(datarobot[[pred_cols[6]]]),
  Class_7 = as.numeric(datarobot[[pred_cols[7]]]),
  Class_8 = as.numeric(datarobot[[pred_cols[8]]]),
  Class_9 = as.numeric(datarobot[[pred_cols[9]]])
)

# Normalize predictions to sum to 1 (in case they don't)
cat("Normalizing predictions...\n")
pred_matrix <- as.matrix(submission[, 2:10])
row_sums <- rowSums(pred_matrix)
pred_matrix <- pred_matrix / row_sums
submission[, 2:10] <- pred_matrix

# Clip predictions to [1e-15, 1-1e-15] to avoid log(0) issues
cat("Clipping predictions to [1e-15, 1-1e-15]...\n")
submission[, 2:10] <- pmax(pmin(as.matrix(submission[, 2:10]), 1 - 1e-15), 1e-15)

# Renormalize after clipping
pred_matrix <- as.matrix(submission[, 2:10])
pred_matrix <- pred_matrix / rowSums(pred_matrix)
submission[, 2:10] <- pred_matrix

# Load sample submission to verify ID order
cat("\nVerifying ID order with sample submission...\n")
sample_sub <- fread(SAMPLE_SUB_FILE)

if (nrow(submission) != nrow(sample_sub)) {
  cat(sprintf("WARNING: Row count mismatch - submission: %d, sample: %d\n", 
              nrow(submission), nrow(sample_sub)))
}

# Check if IDs match
if (all(submission$id == sample_sub$id)) {
  cat("✓ IDs match sample submission order\n")
} else {
  cat("WARNING: IDs don't match sample submission order\n")
  cat("  Merging to ensure correct order...\n")
  
  # Merge to get correct order
  submission <- merge(sample_sub[, "id", with = FALSE], 
                     submission, 
                     by = "id", 
                     all.x = TRUE, 
                     sort = FALSE)
  
  # Reorder to match sample submission
  submission <- submission[match(sample_sub$id, submission$id), ]
  
  # Check for any missing predictions
  if (any(is.na(submission[, 2:10]))) {
    cat("ERROR: Some predictions are missing after merge!\n")
    stop("Cannot proceed with missing predictions")
  }
}

# Final validation
cat("\nValidating submission...\n")
row_sums <- rowSums(submission[, 2:10])
if (any(row_sums < 0.99) || any(row_sums > 1.01)) {
  cat("WARNING: Some predictions don't sum to 1!\n")
  cat(sprintf("  Min sum: %.10f, Max sum: %.10f\n", min(row_sums), max(row_sums)))
} else {
  cat("✓ All predictions sum to 1\n")
}

if (any(submission[, 2:10] < 0) || any(submission[, 2:10] > 1)) {
  cat("WARNING: Some predictions outside [0, 1] range!\n")
  cat(sprintf("  Min: %.10f, Max: %.10f\n", 
              min(as.matrix(submission[, 2:10])), 
              max(as.matrix(submission[, 2:10]))))
} else {
  cat("✓ All predictions in [0, 1] range\n")
}

# Display summary statistics
cat("\nPrediction summary:\n")
cat(sprintf("  Number of rows: %d\n", nrow(submission)))
cat(sprintf("  Number of columns: %d\n", ncol(submission)))
cat(sprintf("  ID range: %d to %d\n", min(submission$id), max(submission$id)))
cat("\nClass probability ranges:\n")
for (i in 1:9) {
  col_name <- paste0("Class_", i)
  probs <- submission[[col_name]]
  cat(sprintf("  %s: [%.6f, %.6f], mean=%.6f\n", 
              col_name, min(probs), max(probs), mean(probs)))
}

# Save submission
cat(sprintf("\nSaving submission to: %s\n", OUTPUT_FILE))
write.csv(submission, OUTPUT_FILE, row.names = FALSE)

cat("\n✓ Submission file created successfully!\n")
cat("  Ready for Kaggle upload\n")

