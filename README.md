# OpenBoost

**A GPU-native, all-Python platform for tree-based machine learning.**

> **Note:** OpenBoost is in active development. APIs may change between releases. Use at your own risk.

## Why OpenBoost?

For standard GBDT, use XGBoost/LightGBM — they're highly optimized C++.

For GBDT **variants** (probabilistic predictions, interpretable GAMs, custom algorithms), OpenBoost brings GPU acceleration to methods that were previously CPU-only and slow:

- **NaturalBoost**: 1.3-2x faster than NGBoost
- **OpenBoostGAM**: 10-40x faster than InterpretML EBM

Plus: ~20K lines of readable Python. Modify, extend, and build on — no C++ required.

| | XGBoost / LightGBM | OpenBoost |
|---|---|---|
| **Code** | 200K+ lines of C++ | ~20K lines of Python |
| **GPU** | Added later | Native from day one |
| **Customize** | Modify C++, recompile | Modify Python, reload |

## What You Can Build

OpenBoost provides primitives (histograms, binning, tree fitting) that you combine into algorithms:

- **Standard GBDT** — drop-in gradient boosting with multiple growth strategies
- **Distributional GBDT** — predict full probability distributions with [NGBoost](https://arxiv.org/abs/1910.03225)-style natural gradient boosting
- **Interpretable GAMs** — explainable feature effects inspired by [EBM](https://arxiv.org/abs/1909.09223)
- **DART** — [dropout regularization](https://arxiv.org/abs/1505.01866) for reduced overfitting
- **Linear-leaf models** — linear models in tree leaves for better extrapolation
- **Your own algorithms** — custom losses, distributions, or entirely new methods

All run on GPU with the same Python code.

## Quick Start

**High-level API:**

```python
import openboost as ob

model = ob.GradientBoosting(n_trees=100, max_depth=6)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
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

Where OpenBoost really shines is on GBDT variants that don't exist in XGBoost/LightGBM:

| Model | vs. | Speedup |
|---|---|---|
| NaturalBoost (GPU) | NGBoost | 1.3-2x |
| OpenBoostGAM (GPU) | InterpretML EBM | 10-40x |

> **Note:** Benchmarks reflect the current state of development and may change as both OpenBoost and comparison libraries evolve.

## Roadmap

**Train-many optimization**: Industry workloads often train many models (hyperparameter tuning, CV, per-segment models). XGBoost optimizes for one model fast. OpenBoost plans to enable native optimization for training many models efficiently.

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
