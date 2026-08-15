# OpenBoost

**The hackable gradient boosting platform — probabilistic predictions, interpretable GAMs, and custom algorithms in readable Python, with CPU and CUDA tree backends.**

> **Note:** OpenBoost is in active development. APIs may change between releases. Use at your own risk.

## Why OpenBoost?

For standard GBDT, use XGBoost/LightGBM — they're highly optimized C++.

For GBDT **variants** (probabilistic predictions, interpretable GAMs, custom algorithms), OpenBoost provides reusable Python primitives and a CUDA tree-building path:

- **NaturalBoost**: full-distribution prediction with a GPU tree path. On the committed CPU comparison, OpenBoost and NGBoost are comparable (0.8-1.3x wall-clock, quality within ~1%) — see [Benchmarks](#benchmarks)
- **OpenBoostGAM**: interpretable main effects with an optional GPU training path; use the included harness to measure speed and accuracy on your workload
- **Your own algorithms**: custom losses, distributions, and tree-growth strategies are registration APIs (`register_loss`, `register_distribution`, `register_growth_strategy`), not C++ forks — see the [cookbook](https://jxucoder.github.io/openboost/cookbook/custom-loss/)

Plus: ~20K lines of readable Python. Modify, extend, and build on — no C++ required.

| | XGBoost / LightGBM | OpenBoost |
|---|---|---|
| **Code** | 200K+ lines of C++ | ~20K lines of Python |
| **GPU** | Added later | Native from day one |
| **Customize** | Modify C++, recompile | Modify Python, reload |

## What You Can Build

OpenBoost provides primitives (histograms, binning, tree fitting) that you combine into algorithms:

- **Standard GBDT** — drop-in gradient boosting with selectable growth strategies (`growth='levelwise' | 'leafwise' | 'symmetric'`), early stopping, and callbacks
- **Distributional GBDT** — predict full probability distributions with [NGBoost](https://arxiv.org/abs/1910.03225)-style natural gradient boosting
- **Interpretable GAMs** — explainable feature effects inspired by [EBM](https://arxiv.org/abs/1909.09223)
- **DART** — [dropout regularization](https://arxiv.org/abs/1505.01866) for reduced overfitting
- **Linear-leaf models** — linear models in tree leaves for better extrapolation
- **Your own algorithms** — custom losses, distributions, or entirely new methods

The core tree-building paths support CPU and CUDA backends. Some features and
model stages remain CPU-only or deliberately fall back to CPU; see the model
guides for those boundaries. All models support `save()`/`load()` persistence,
and most support callbacks and early stopping.

## Quick Start

**High-level API:**

```python
import openboost as ob

model = ob.GradientBoosting(n_trees=100, max_depth=6, random_state=42)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[ob.EarlyStopping(patience=10)])
predictions = model.predict(X_test)
```

**sklearn-compatible:**

```python
from openboost import OpenBoostRegressor
from sklearn.model_selection import GridSearchCV

# Works with GridSearchCV, Pipeline, cross_val_score, etc.
model = OpenBoostRegressor(n_estimators=100, random_state=42)
search = GridSearchCV(model, {"max_depth": [4, 6, 8]}, cv=5)
search.fit(X_train, y_train)

# Also available: OpenBoostClassifier, OpenBoostDARTRegressor,
# OpenBoostGAMRegressor, OpenBoostDistributionalRegressor
```

**Hyperparameter suggestions:**

```python
# Auto-suggest params based on dataset characteristics
params = ob.suggest_params(X_train, y_train, task='regression', style='core')
model = ob.GradientBoosting(**params)
```

**Low-level API** (full control over the training loop):

```python
import openboost as ob

X_binned = ob.array(X_train)
pred = np.zeros(len(y_train), dtype=np.float32)

for round in range(100):
    grad = 2 * (pred - y_train)  # your gradients
    hess = np.ones_like(grad) * 2
    tree = ob.fit_tree(X_binned, grad, hess, max_depth=6)
    pred += 0.1 * tree(X_binned)
```

## Installation

```bash
# Current release candidate (recommended while 1.0 is in prerelease)
pip install --pre openboost

# With GPU support
pip install --pre "openboost[cuda]"

# With sklearn integration
pip install --pre "openboost[sklearn]"
```

`pip install openboost` without `--pre` installs the older stable release.

## Documentation

Full docs, tutorials, and API reference: **[jxucoder.github.io/openboost](https://jxucoder.github.io/openboost)**

- [Getting Started](https://jxucoder.github.io/openboost/getting-started/installation/)
- [User Guide](https://jxucoder.github.io/openboost/user-guide/models/gradient-boosting/)
- [API Reference](https://jxucoder.github.io/openboost/api/openboost/)
- [Examples](./examples/)

## Benchmarks

### Committed comparison: NaturalBoost vs NGBoost on CPU

The repository includes one current, auditable third-party comparison:
`benchmarks/results/ngboost_comparison_20260720.json`. It uses fixed seeds,
identical boosting budgets, and the same train/test splits.

| Dataset | OpenBoost / NGBoost fit time | Result |
|---|---|---|
| Synthetic heteroscedastic, 10K | 16.4s / 18.8s (1.15x) | OpenBoost slightly better NLL/CRPS/RMSE |
| Synthetic heteroscedastic, 50K | 74.1s / 95.3s (1.29x) | NGBoost slightly better NLL/CRPS/RMSE |
| California Housing, 20.6K | 30.6s / 25.0s (0.82x) | OpenBoost slightly better NLL/CRPS/RMSE |

The honest read is CPU parity: neither implementation wins every dataset, and
quality is within roughly 1% in this run.

Reproduce it with:

```bash
OPENBOOST_BACKEND=cpu uv run --with ngboost python benchmarks/bench_ngboost_comparison.py
```

### GPU benchmark harnesses

GPU comparisons are available in `benchmarks/bench_gpu.py` and
`benchmarks/compare_gpu.py`. Third-party GPU speedups are intentionally not
quoted here until the exact raw result artifact and environment metadata are
committed alongside the claim.

```bash
# Local CUDA GPU
uv run python benchmarks/bench_gpu.py --task all --scale medium

# Modal A100
uv run modal run benchmarks/bench_gpu.py --task all --scale medium
```

NaturalBoost's CUDA acceleration applies to histogram-based tree building;
distribution gradients and Fisher/natural-gradient calculations still run on
CPU. Benchmark end-to-end fit time, accuracy, and calibration on the workload
you actually care about.

## Roadmap

**Train-many optimization**: OpenBoost now has a correctness-first API that shares
binned data across hyperparameter configurations. The next milestone is fusing
histogram and split work across configurations on GPU, with the sequential path
serving as the behavioral reference.

## References

OpenBoost implements and builds on ideas from these papers:

- **Gradient Boosting**: Friedman, J. H. (2001). [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/euclid.aos/1013203451). *Annals of Statistics*.
- **XGBoost**: Chen, T., & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754). *KDD*.
- **LightGBM**: Ke, G., et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree). *NeurIPS*.
- **CatBoost**: Prokhorenkova, L., et al. (2018). [CatBoost: Unbiased Boosting with Categorical Features](https://arxiv.org/abs/1706.09516). *NeurIPS*.
- **NGBoost**: Duan, T., et al. (2020). [NGBoost: Natural Gradient Boosting for Probabilistic Prediction](https://arxiv.org/abs/1910.03225). *ICML*.
- **EBM**: Nori, H., et al. (2019). [InterpretML: A Unified Framework for Machine Learning Interpretability](https://arxiv.org/abs/1909.09223).
- **DART**: Rashmi, K. V., & Gilad-Bachrach, R. (2015). [DART: Dropouts meet Multiple Additive Regression Trees](https://arxiv.org/abs/1505.01866). *AISTATS*.

## License

Apache 2.0
