# OpenBoost

<p align="center" style="font-size: 1.2em;">
  <strong>The hackable gradient boosting platform — probabilistic predictions, interpretable GAMs, and custom algorithms in readable Python, with CPU and CUDA tree backends.</strong>
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

For GBDT **variants** (probabilistic predictions, interpretable GAMs, custom algorithms), OpenBoost provides reusable Python primitives and a CUDA tree-building path:

- **NaturalBoost**: full-distribution prediction; comparable to NGBoost on the committed CPU benchmark (0.8-1.3x wall-clock, quality within ~1%)
- **OpenBoostGAM**: interpretable main effects with an optional GPU path; benchmark it on your own workload with the included harness

Plus: ~20K lines of readable Python. Modify, extend, and build on—no C++ required.

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

Numba CUDA kernels accelerate histogram building and tree construction. Some
features and model stages remain CPU-only or deliberately fall back to CPU;
the model guides document those boundaries.

### :brain: Probabilistic Predictions

NaturalBoost provides full probability distributions with uncertainty quantification. 8 built-in distributions including Normal, Gamma, Tweedie, and Negative Binomial.

### :snake: All Python

~20K lines of readable, hackable code. No C++ compilation needed. Understand and modify the algorithms.

### :gear: sklearn Compatible

Drop-in replacement for scikit-learn pipelines. Works with GridSearchCV, cross_val_score, and Pipeline.

## Installation

```bash
pip install --pre openboost

# With GPU support
pip install --pre "openboost[cuda]"
```

Without `--pre`, pip installs the older stable release rather than the current
1.0 release candidate.

## What's Included

| Category | Models |
|----------|--------|
| **Standard GBDT** | GradientBoosting, MultiClassGradientBoosting, DART |
| **Interpretable** | OpenBoostGAM, LinearLeafGBDT |
| **Probabilistic** | NaturalBoostNormal, LogNormal, Gamma, Poisson, StudentT, Tweedie, NegBin |

## Performance

The repository currently includes one auditable third-party comparison:
`benchmarks/results/ngboost_comparison_20260720.json`.

| Benchmark | Result |
|-----------|--------|
| NaturalBoost vs NGBoost (CPU) | ~parity: 0.8-1.3x, NLL/CRPS/RMSE within ~1% (`ngboost_comparison_20260720.json`) |

GPU benchmark harnesses are included, but exact third-party speedup claims are
not published until the corresponding raw result artifact and environment
metadata are committed. For standard GBDT, use XGBoost/LightGBM; OpenBoost's
value is in research-friendly variants and extensibility.

## Who Is OpenBoost For?

- **Kaggle Competitors** - Probabilistic predictions that XGBoost can't do
- **ML Researchers** - Prototype new algorithms in Python
- **Product teams** - Prototype interpretable or probabilistic models before production hardening
- **Students** - Actually understand how gradient boosting works

## Roadmap

**Train-many optimization**: Industry workloads often train many models (hyperparameter tuning, CV, per-segment models). XGBoost optimizes for one model fast. OpenBoost plans to enable native optimization for training many models efficiently.

## License

Apache 2.0
