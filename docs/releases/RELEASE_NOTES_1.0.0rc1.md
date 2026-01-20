# OpenBoost 1.0.0rc1 Release Notes

**The GPU-native, all-Python platform for tree-based machine learning.**

## Why OpenBoost?

XGBoost and LightGBM are 200K+ lines of C++. OpenBoost is ~20K lines of Python.

For standard GBDT, use XGBoost. For GBDT **variants** (probabilistic predictions, interpretable GAMs, custom algorithms), OpenBoost brings GPU acceleration to methods that were previously CPU-only and slow.

| | XGBoost / LightGBM | OpenBoost |
|---|---|---|
| **Code** | 200K+ lines of C++ | ~20K lines of Python |
| **GPU** | Added later | Native from day one |
| **Customize** | Modify C++, recompile | Modify Python, reload |

## What's New in 1.0.0rc1

### Core Models
- `GradientBoosting` - Standard gradient boosting for regression/classification
- `MultiClassGradientBoosting` - Multi-class classification with softmax
- `DART` - Dropout regularized trees
- `OpenBoostGAM` - GPU-accelerated interpretable GAM (10-40x faster than InterpretML EBM)

### Probabilistic Predictions (NaturalBoost)
Predict full probability distributions, not just point estimates:
- Normal, LogNormal, Gamma, Poisson, StudentT
- **Tweedie** - Insurance claims (Kaggle favorite)
- **NegativeBinomial** - Sales forecasting (Kaggle favorite)

```python
model = ob.NaturalBoostNormal(n_trees=100)
model.fit(X_train, y_train)
lower, upper = model.predict_interval(X_test, alpha=0.1)  # 90% prediction interval
```

### Advanced Features
- Linear models in tree leaves (`LinearLeafGBDT`)
- GPU acceleration via Numba CUDA
- Multi-GPU support via Ray
- GOSS sampling (LightGBM-style)
- Full sklearn compatibility (`OpenBoostRegressor`, `OpenBoostClassifier`)

### Performance
OpenBoost GPU-accelerates GBDT variants that were previously slow:
- **NaturalBoost**: 1.3-2x faster than NGBoost
- **OpenBoostGAM**: 10-40x faster than InterpretML EBM

For standard GBDT, XGBoost/LightGBM are faster. OpenBoost's value is in the variants.

## Installation

```bash
pip install openboost

# With GPU support
pip install openboost[cuda]
```

## Quick Start

```python
import openboost as ob

model = ob.GradientBoosting(n_trees=100, max_depth=6)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Known Limitations (rc1)
- `sample_weight` not yet fully supported on GPU (works on CPU)
- `MultiClassGradientBoosting` does not support callbacks yet
- Multi-GPU requires Ray and raw numpy arrays

## Links
- **Documentation**: https://jxucoder.github.io/openboost
- **GitHub**: https://github.com/jxucoder/openboost
- **Examples**: https://github.com/jxucoder/openboost/tree/main/examples

## What's Next for 1.0.0
- GPU sample_weight support
- Callbacks for multi-class models
- Additional distributions

## Roadmap: Train-Many Optimization

Industry workloads often require training **many models** at once:
- Hyperparameter tuning (100s of configs)
- Cross-validation (k models)
- Per-segment models (one per store/product/customer)

XGBoost/LightGBM optimize for training **one model fast**. OpenBoost's Python architecture plans to enable native optimization for training **many models efficiently** - batching, GPU parallelization across models, and amortized overhead.

This is a planned advantage for future releases.

---

*Built for the agentic era: every algorithm readable, modifiable, debuggable in Python.*
