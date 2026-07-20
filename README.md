# OpenBoost

**A GPU-native, all-Python platform for tree-based machine learning.**

> **Note:** OpenBoost is in active development. APIs may change between releases. Use at your own risk.

## Why OpenBoost?

For standard GBDT, use XGBoost/LightGBM — they're highly optimized C++.

For GBDT **variants** (probabilistic predictions, interpretable GAMs, custom algorithms), OpenBoost brings GPU acceleration to methods that were previously CPU-only and slow:

- **NaturalBoost**: 1.6-11x faster than NGBoost on GPU (tree build only; gradient/Fisher math stays on CPU). On CPU the two are comparable (0.8-1.3x, quality within ~1%) — see Benchmarks
- **OpenBoostGAM**: much faster than InterpretML EBM on our committed run (56x), with an accuracy tradeoff — see [Benchmarks](#benchmarks) for the honest numbers

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

All run on GPU with the same Python code. All models support `save()`/`load()` persistence, and most support callbacks and early stopping.

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
pip install openboost

# With GPU support
pip install openboost[cuda]

# With sklearn integration
pip install openboost[sklearn]
```

## Documentation

Full docs, tutorials, and API reference: **[jxucoder.github.io/openboost](https://jxucoder.github.io/openboost)**

- [Getting Started](https://jxucoder.github.io/openboost/getting-started/installation/)
- [User Guide](https://jxucoder.github.io/openboost/user-guide/models/gradient-boosting/)
- [API Reference](https://jxucoder.github.io/openboost/api/openboost/)
- [Examples](./examples/)

## Benchmarks

### GPU: OpenBoost vs XGBoost

On standard GBDT, OpenBoost's GPU-native tree builder is **3-4x faster** than XGBoost's GPU histogram method on an A100, with comparable accuracy:

| Task | Data | Trees | OpenBoost | XGBoost | Speedup |
|---|---|---|---|---|---|
| Regression | 2M x 80 | 300 | 10.0s | 45.5s | **4.6x** |
| Binary | 2M x 80 | 300 | 11.8s | 40.9s | **3.5x** |

<details>
<summary>Benchmark details</summary>

- **Hardware**: NVIDIA A100 (Modal)
- **Fairness controls**: both receive raw numpy arrays (no pre-built DMatrix), `cuda.synchronize()` after OpenBoost `fit()`, both at default threading, XGBoost `max_bin=256` to match OpenBoost, JIT/GPU warmup before timing
- **Metric**: median of 3 trials, timing `fit()` only
- **XGBoost config**: `tree_method="hist"`, `device="cuda"`

Reproduce with:
```bash
# Local (requires CUDA GPU)
uv run python benchmarks/bench_gpu.py --task all --scale medium

# On Modal A100
uv run modal run benchmarks/bench_gpu.py --task all --scale medium
```

Available scales: `small` (500K), `medium` (2M), `large` (5M), `xlarge` (10M).

</details>

### Variant models

Where OpenBoost really shines is on GBDT variants that don't exist in XGBoost/LightGBM. From the committed benchmark run (`benchmarks/results/gpu_benchmark_20260322_153105.json`, Modal A100):

| Model | vs. | Speedup | Accuracy |
|---|---|---|---|
| NaturalBoost (GPU) | NGBoost | 1.6x (California housing), 11.5x (synthetic 50K) | NLL slightly behind NGBoost on both datasets |
| NaturalBoost (CPU) | NGBoost | ~parity: 0.8x–1.3x (`ngboost_comparison_20260720.json`) | NLL/CRPS/RMSE within ~1% of each other; NGBoost wins some |
| OpenBoostGAM (GPU) | InterpretML EBM | 56x (synthetic 50K) | **Lower**: R² 0.66 vs 0.74 |

Caveats to read before quoting these numbers:

- The EBM comparison disabled EBM's interactions and bagging (`interactions=0`, `outer_bags=1`, `inner_bags=0`) to isolate main-effect training; OpenBoostGAM is main-effects-only. On this run OpenBoostGAM was much faster but less accurate.
- NaturalBoost's GPU acceleration applies to the histogram-based tree build. The per-round distribution gradient and Fisher/natural-gradient computations run on CPU (numpy), so distributional training is not GPU-accelerated end-to-end.
- On CPU the two libraries are comparable: the committed CPU-vs-CPU run (`benchmarks/results/ngboost_comparison_20260720.json`, fixed seed, identical budgets) shows 0.8x–1.3x wall-clock and quality metrics within ~1%, with NGBoost ahead on some. The GPU speedups above are where NaturalBoost's advantage actually lives.

> **Note:** Benchmarks reflect the current state of development and may change as both OpenBoost and comparison libraries evolve.

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
