# Changelog

All notable changes to OpenBoost will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `FormulaBoost`: boost parameters of a user formula `y = f(θ, x)` with
  finite-difference Jacobian and damped GGN (`precond='full'|'diag'|'plain'`).
- `WeibullAFT`: right-censored Weibull AFT that boosts both scale `λ(z)`
  and shape `k(z)` using expected-Fisher natural gradient.
- Unified `fit_boosting` trainer and `Objective` protocol shared by
  NaturalBoost, FormulaBoost, and WeibullAFT.
- Capability benchmarks `benchmarks/bench_formula.py` and
  `benchmarks/bench_survival.py`; probabilistic speed/quality suite
  `benchmarks/bench_probabilistic.py`.

### Changed

- Documentation and package description framed as distributional
  regression (GAMLSS / NGBoost) and varying-coefficient models: README,
  docs home, quickstart, GPU setup, NaturalBoost guide, XGBoost
  migration, and a new benchmarks page. Headline numbers: 1229× vs
  NGBoost on A100 at 90K (NLL tied); FormulaBoost ~21× better
  extrapolation than black-box GBDT; WeibullAFT recovers `k(z)`.
- Performance documentation now quotes only results backed by committed
  benchmark runs (Modal A100 speed/capability, CPU UCI quality).
- Made multi-round `fit_trees_batch` recompute gradients and hessians from targets.
- Consolidated batch configuration and training state into one canonical module.
- Added manually triggered CUDA verification on Modal GPUs.
- Custom distributions now fall back to numerical differentiation when JAX
  cannot trace a user NLL, instead of returning placeholder gradients.
- Unsupported `sample_weight` inputs now raise on CUDA, distributed, and
  multi-GPU training paths instead of being ignored.

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
GPU-accelerates GBDT variants that were previously slow:
- NaturalBoost and OpenBoostGAM include CUDA tree-building paths.
- Exact speedups depend on the workload and environment; current documentation
  quotes only benchmark results whose raw artifacts are committed.

For standard GBDT, XGBoost/LightGBM are faster. OpenBoost's value is in the variants and customizability.

### Known Limitations (1.0.0rc1)
- `sample_weight` is CPU-only; CUDA, distributed, and multi-GPU paths raise
  `NotImplementedError` rather than silently ignoring weights
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

### Roadmap
- GPU sample_weight support
- Callbacks for multi-class models
- **Train-many optimization**: Native support for training many models efficiently (hyperparameter tuning, CV, per-segment models) - batching and GPU parallelization across models

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
