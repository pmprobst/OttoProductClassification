# Submission file writing for Otto Product Classification
# Depends: R/config.R (DATA_DIR, OUTPUT_DIR, CLASSES)

#' Write submission CSV in Kaggle format (id, Class_1, ..., Class_9).
#' Aligns test_id with sample submission if needed.
#' @param predictions matrix with columns Class_1 ... Class_9
#' @param test_id vector of test IDs
#' @param path output file path (e.g. file.path(OUTPUT_DIR, "submission.csv"))
write_submission <- function(predictions, test_id, path) {
  sample_sub <- data.table::fread(SAMPLE_SUB_FILE)
  if (length(test_id) != nrow(sample_sub) || !all(test_id == sample_sub$id)) {
    test_id <- sample_sub$id
  }
  submission <- data.frame(
    id = test_id,
    Class_1 = as.numeric(predictions[, "Class_1"]),
    Class_2 = as.numeric(predictions[, "Class_2"]),
    Class_3 = as.numeric(predictions[, "Class_3"]),
    Class_4 = as.numeric(predictions[, "Class_4"]),
    Class_5 = as.numeric(predictions[, "Class_5"]),
    Class_6 = as.numeric(predictions[, "Class_6"]),
    Class_7 = as.numeric(predictions[, "Class_7"]),
    Class_8 = as.numeric(predictions[, "Class_8"]),
    Class_9 = as.numeric(predictions[, "Class_9"])
  )
  row_sums <- rowSums(submission[, 2:10])
  if (any(row_sums < 0.99) || any(row_sums > 1.01)) {
    warning("Some predictions do not sum to 1")
  }
  write.csv(submission, path, row.names = FALSE)
  invisible(submission)
}
