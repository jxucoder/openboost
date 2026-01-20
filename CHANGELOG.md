# Changelog

All notable changes to OpenBoost will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0rc1] - 2026-01-20

### Added

#### Core Models
- `GradientBoosting` - Standard gradient boosting for regression/classification
- `MultiClassGradientBoosting` - Multi-class classification with softmax
- `DART` - Dropout regularized trees
- `OpenBoostGAM` - GPU-accelerated interpretable GAM

#### Distributional Models (NaturalBoost)
- `NaturalBoostNormal` - Gaussian distribution
- `NaturalBoostLogNormal` - Log-normal for positive values
- `NaturalBoostGamma` - Gamma distribution
- `NaturalBoostPoisson` - Count data
- `NaturalBoostStudentT` - Heavy tails
- `NaturalBoostTweedie` - Insurance claims (Kaggle favorite)
- `NaturalBoostNegBin` - Sales forecasting (Kaggle favorite)

#### Advanced Features
- `LinearLeafGBDT` - Linear models in tree leaves
- GPU acceleration via Numba CUDA
- Multi-GPU support via Ray
- GOSS sampling (LightGBM-style)
- Mini-batch training for out-of-core datasets
- Memory-mapped array support

#### sklearn Integration
- `OpenBoostRegressor` - sklearn-compatible regressor
- `OpenBoostClassifier` - sklearn-compatible classifier
- `OpenBoostDistributionalRegressor` - Distributional regressor
- `OpenBoostLinearLeafRegressor` - Linear leaf regressor

#### Callbacks
- `EarlyStopping` - Stop training when validation metric stops improving
- `Logger` - Print training progress
- `ModelCheckpoint` - Save best models during training
- `LearningRateScheduler` - Dynamic learning rate

#### Loss Functions
- MSE, MAE, Huber, Quantile (regression)
- LogLoss, Softmax (classification)
- Poisson, Gamma, Tweedie (count/positive data)
- Custom loss function support

#### Growth Strategies
- Level-wise (XGBoost-style)
- Leaf-wise (LightGBM-style)
- Symmetric/Oblivious (CatBoost-style)

#### Utilities
- `compute_feature_importances()` - Gain-based importance
- `suggest_params()` - Automatic parameter suggestions
- `cross_val_predict()` - Out-of-fold predictions
- `evaluate_coverage()` - Prediction interval validation

### Performance
- NaturalBoost: 1.3-2x faster than NGBoost
- OpenBoostGAM: 10-40x faster than InterpretML EBM on GPU
- Standard GBDT: Comparable to XGBoost (within 5% RMSE)

### Known Limitations (1.0.0rc1)
- `sample_weight` is not yet fully supported on GPU backend (works on CPU)
- `MultiClassGradientBoosting` does not support callbacks (early stopping, logging)
- Multi-GPU training requires Ray and raw numpy arrays (not pre-binned data)
- JAX backend for custom distributions is optional (falls back to numerical gradients)

### Documentation
- Comprehensive README with examples
- Quickstart guide
- Uncertainty quantification tutorial
- Custom loss function tutorial
- XGBoost migration guide
- 13 runnable examples

## Development Phases

This release represents 22 phases of development:
- Phase 1-7: Core implementation
- Phase 8-9: Growth strategies and loss functions
- Phase 10-11: Feature importance and custom loss
- Phase 12-14: Callbacks, sklearn integration, regularization
- Phase 15-16: Distributional GBDT (NaturalBoost)
- Phase 17-18: Large-scale training, multi-GPU
- Phase 19-21: Integration testing, CUDA verification
- Phase 22: Pre-launch QA
