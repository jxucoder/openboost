# OpenBoost

<p align="center" style="font-size: 1.2em;">
  <strong>A GPU-native, all-Python platform for tree-based machine learning.</strong>
</p>

<p align="center">
  <a href="getting-started/quickstart/">Quickstart</a> •
  <a href="user-guide/models/gradient-boosting/">Models</a> •
  <a href="user-guide/naturalboost/overview/">NaturalBoost</a> •
  <a href="api/openboost/">API Reference</a>
</p>

---

## Why OpenBoost?

For standard GBDT, use XGBoost/LightGBM—they're highly optimized C++.

For GBDT **variants** (probabilistic predictions, interpretable GAMs, custom algorithms), OpenBoost brings GPU acceleration to methods that were previously CPU-only and slow:

- **NaturalBoost**: 1.3-2x faster than NGBoost
- **OpenBoostGAM**: 10-40x faster than InterpretML EBM

Plus: ~20K lines of readable Python. Modify, extend, and build on—no C++ required.

| | XGBoost / LightGBM | OpenBoost |
|---|---|---|
| **Code** | 200K+ lines of C++ | ~20K lines of Python |
| **GPU** | Added later | Native from day one |
| **Customize** | Modify C++, recompile | Modify Python, reload |

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

~20K lines of readable, hackable code. No C++ compilation needed. Understand and modify the algorithms.

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

OpenBoost GPU-accelerates GBDT variants that were previously slow:

| Benchmark | Result |
|-----------|--------|
| NaturalBoost vs NGBoost | **1.3-2x faster** |
| OpenBoostGAM vs InterpretML EBM | **10-40x faster** |

For standard GBDT, XGBoost/LightGBM are faster. OpenBoost's value is in the variants.

## Who Is OpenBoost For?

- **Kaggle Competitors** - Probabilistic predictions that XGBoost can't do
- **ML Researchers** - Prototype new algorithms in Python
- **Startups** - Ship interpretable models fast
- **Students** - Actually understand how gradient boosting works

## Roadmap

**Train-many optimization**: Industry workloads often train many models (hyperparameter tuning, CV, per-segment models). XGBoost optimizes for one model fast. OpenBoost plans to enable native optimization for training many models efficiently.

## License

Apache 2.0
