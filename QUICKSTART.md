# Quick Start Guide

## Step 1: Install Packages

Open R and run:
```r
source("install_packages.R")
```

This will install all required packages.

## Step 2: Verify Data

Make sure your data files are in the `data/` directory:
- `train.csv`
- `test.csv`
- `sampleSubmission.csv`

## Step 3: Run Training

Train all models (this may take several hours):
```r
source("07_train_models.R")
```

**Note:** For faster initial testing, you can modify `07_train_models.R` and set:
- `include_tsne = FALSE` (line 26) - skips slow t-SNE computation
- Reduce `N_FOLDS_L1` and `N_FOLDS_L2` in `config.R` for faster CV

## Step 4: Generate Submission

After training completes, generate test predictions:
```r
source("08_predict_test.R")
```

The submission file will be saved to `output/submission.csv`.

## Troubleshooting

### Package Installation Issues
If some packages fail to install:
- **Rtsne** (t-SNE): This package may require system dependencies
  - On macOS: Install XQuartz from https://www.xquartz.org/ (if you get X11 errors)
  - Alternatively: You can skip t-SNE features by setting `include_tsne = FALSE` in `07_train_models.R`
  - The code will work without Rtsne, just without t-SNE features
- **xgboost**: May need to be installed from source on some systems
- Try installing packages individually: `install.packages("package_name")`

### X11/XQuartz Warnings
If you see X11 library warnings (especially on macOS):
- These are usually harmless and won't affect functionality
- If you want to use t-SNE features, install XQuartz
- Otherwise, just set `include_tsne = FALSE` to skip t-SNE

### Memory Issues
If you run out of memory:
- Reduce the number of KNN models (edit `KNN_NEIGHBORS` in `config.R`)
- Set `include_kmeans = FALSE` in feature engineering
- Reduce `N_TREES_RF`, `N_TREES_XGB`, etc. in `config.R`

### Slow Performance
- Disable t-SNE: `include_tsne = FALSE`
- Reduce CV folds: `N_FOLDS_L1 = 3`, `N_FOLDS_L2 = 3`
- Reduce number of KNN models
- Use fewer trees in tree-based models

## Expected Runtime

- Feature engineering: 10-30 minutes (depending on t-SNE)
- Level 1 models: 1-3 hours (depending on number of models)
- Level 2 models: 30-60 minutes
- Test prediction: 30-60 minutes

**Total: 2-5 hours** (depending on hardware and configuration)

