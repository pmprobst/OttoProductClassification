# Otto Product Classification

This project implements a 3-layer stacking ensemble approach for the Otto Product Classification competition.

## Project Structure

```
.
├── config.R                    # Configuration settings
├── 01_load_data.R              # Data loading and preparation
├── 02_feature_engineering.R    # Feature engineering functions
├── 03_cross_validation.R       # CV framework and metrics
├── 04_level1_models.R         # Level 1 base models
├── 05_level2_models.R          # Level 2 meta-learners
├── 06_level3_ensemble.R        # Level 3 final ensemble
├── 07_train_models.R           # Main training script
├── 08_predict_test.R           # Test prediction and submission
├── install_packages.R          # Package installation script
├── data/                       # Data directory
│   ├── train.csv
│   ├── test.csv
│   └── sampleSubmission.csv
└── output/                     # Output directory (created automatically)
    ├── models/
    ├── predictions/
    └── features/
```

## Setup

1. **Install R packages:**
   ```r
   source("install_packages.R")
   ```

2. **Ensure data files are in the `data/` directory:**
   - `train.csv`
   - `test.csv`
   - `sampleSubmission.csv`

## Usage

### Training Models

Run the main training script to train all models and generate cross-validation predictions:

```r
source("07_train_models.R")
```

This will:
1. Load and prepare the data
2. Engineer features (log, sqrt, scale, t-SNE, K-means, KNN distances)
3. Train Level 1 models with 5-fold CV
4. Train Level 2 models with 4-fold CV
5. Optimize ensemble weights
6. Calculate final CV score

### Generating Test Predictions

After training, generate test predictions and submission file:

```r
source("08_predict_test.R")
```

This will create `output/submission.csv` in the required format.

## Model Architecture

### Level 1: Base Learners
- Random Forest (raw features)
- Logistic Regression (log features)
- Extra Trees (log features)
- K-Nearest Neighbors (multiple k values, scaled log features)
- Naive Bayes (log features)
- XGBoost (raw features)
- Neural Network (scaled log features)

### Level 2: Meta Learners
- XGBoost
- Neural Network
- AdaBoost with Extra Trees

### Level 3: Final Ensemble
Weighted combination using geometric mean of XGBoost and Neural Network, then arithmetic mean with AdaBoost Extra Trees.

## Feature Engineering

- **Basic transformations:** log(X+1), sqrt(X+3/8), scaled features
- **Row statistics:** sum, non-zero count, mean, max, min
- **Dimensionality reduction:** t-SNE (3 dimensions)
- **Clustering:** K-means with multiple cluster counts
- **Distance features:** KNN distances to each class

## Configuration

Edit `config.R` to adjust:
- Cross-validation folds
- Model hyperparameters
- Feature engineering settings
- Output directories

## Notes

- t-SNE computation can be slow for large datasets. Set `include_tsne = FALSE` in `07_train_models.R` for faster initial testing.
- The full pipeline may take several hours to run depending on your hardware.
- All models use stratified cross-validation to maintain class distribution.

## Performance

The ensemble approach aims to achieve competitive performance similar to the winning solutions described in the competition, targeting a log loss around 0.38-0.40.

