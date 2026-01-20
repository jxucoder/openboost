# OpenBoost

<p align="center" style="font-size: 1.2em;">
  <strong>The GPU-native, all-Python platform for tree-based machine learning.</strong>
</p>

<p align="center">
  <a href="getting-started/quickstart/">Quickstart</a> •
  <a href="user-guide/models/gradient-boosting/">Models</a> •
  <a href="user-guide/naturalboost/overview/">NaturalBoost</a> •
  <a href="api/openboost/">API Reference</a>
</p>

---

## Why OpenBoost?

**XGBoost and LightGBM are products. OpenBoost is a platform.**

| | XGBoost / LightGBM | OpenBoost |
|---|---|---|
| **Code** | 200K+ lines of C++ | ~6K lines of Python |
| **GPU** | Added later (bolt-on) | Native (designed for it) |
| **Customize** | Write C++, recompile | Modify Python, reload |
| **New algorithms** | Wait for maintainers | Build it yourself |

## Quick Example

```python
import openboost as ob

# Standard gradient boosting
model = ob.GradientBoosting(n_trees=100, max_depth=6)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Probabilistic predictions with uncertainty
prob_model = ob.NaturalBoostNormal(n_trees=100)
prob_model.fit(X_train, y_train)
mean = prob_model.predict(X_test)
lower, upper = prob_model.predict_interval(X_test, alpha=0.1)  # 90% interval
```

## Features

### :rocket: GPU Accelerated

Numba CUDA kernels for histogram building and tree construction. 10-40x faster GAM training compared to CPU alternatives.

### :brain: Probabilistic Predictions

NaturalBoost provides full probability distributions with uncertainty quantification. 8 built-in distributions including Normal, Gamma, Tweedie, and Negative Binomial.

### :snake: All Python

~6K lines of readable, hackable code. No C++ compilation needed. Understand and modify the algorithms.

### :gear: sklearn Compatible

Drop-in replacement for scikit-learn pipelines. Works with GridSearchCV, cross_val_score, and Pipeline.

## Installation

```bash
pip install openboost

# With GPU support
pip install "openboost[cuda]"
```

## What's Included

| Category | Models |
|----------|--------|
| **Standard GBDT** | GradientBoosting, MultiClassGradientBoosting, DART |
| **Interpretable** | OpenBoostGAM, LinearLeafGBDT |
| **Probabilistic** | NaturalBoostNormal, LogNormal, Gamma, Poisson, StudentT, Tweedie, NegBin |

## Performance

| Benchmark | Comparison | Result |
|-----------|------------|--------|
| NaturalBoost vs NGBoost | 20K samples | **11x faster** |
| OpenBoostGAM vs InterpretML EBM | 100K samples | **43x faster** |
| GradientBoosting vs XGBoost | Model quality | Within 5% RMSE |

## Who Is OpenBoost For?

- **Kaggle Competitors** - Probabilistic predictions that XGBoost can't do
- **ML Researchers** - Prototype new algorithms in Python
- **Startups** - Ship interpretable models fast
- **Students** - Actually understand how gradient boosting works

## License

Apache 2.0
