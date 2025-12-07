# Simplified Otto Classification - 2-Level Stacking

This is a **simplified version** that demonstrates the core stacking principle with minimal complexity.

## Core Principle

**Stacking**: Train diverse models (Level 1), use their predictions as features for a meta-model (Level 2).

## What's Different from the Full Version?

### Simplified:
- ✅ **2 levels** instead of 3
- ✅ **3 base models** instead of 10+
- ✅ **Basic features only** (no t-SNE, K-means, etc.)
- ✅ **Simple ensemble** (just averaging)
- ✅ **Single script** instead of modular files
- ✅ **Faster to run** (~30-60 minutes vs 2-5 hours)

### Still Includes:
- ✅ Cross-validation for unbiased predictions
- ✅ Multiple diverse models (RF, XGBoost, Neural Network)
- ✅ Stacking architecture (Level 1 → Level 2)
- ✅ Proper evaluation metric (multi-class log loss)

## Quick Start

1. **Install packages:**
   ```r
   source("install_packages.R")
   ```

2. **Run training:**
   ```r
   source("simple_train.R")
   ```

That's it! The script will:
- Load data
- Train 3 Level 1 models with 5-fold CV
- Train 1 Level 2 meta-model
- Generate submission file

## Model Architecture

### Level 1 (Base Models):
1. **Random Forest** - Tree-based, raw features
2. **XGBoost** - Gradient boosting, raw features  
3. **Neural Network** - Neural network, log-transformed features

### Level 2 (Meta-Model):
- **XGBoost** trained on Level 1 predictions

## Expected Performance

- **Simplified version**: ~0.45-0.50 log loss (good baseline)
- **Full version**: ~0.38-0.40 log loss (competitive)

## When to Use Which?

- **Simple version**: Learning, prototyping, quick results
- **Full version**: Competition submission, maximum performance

## Customization

Easy to modify `simple_train.R`:
- Add more Level 1 models (copy a model block)
- Change hyperparameters (ntree, nrounds, etc.)
- Add feature engineering (log, sqrt, etc.)
- Try different Level 2 models

## Key Concepts Demonstrated

1. **Out-of-fold predictions**: Never use training data to predict itself
2. **Model diversity**: Different algorithms capture different patterns
3. **Stacking**: Meta-model learns to combine base models optimally
4. **Cross-validation**: Proper evaluation without overfitting

