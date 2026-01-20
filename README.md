# OpenBoost

**A GPU-native, all-Python platform for tree-based machine learning.**

## Why OpenBoost?

Tree-based ML libraries are monolithic C++ codebases—optimized for performance, but hard to customize, extend, or learn from.

OpenBoost takes a different approach: GPU-accelerated building blocks in pure Python. Readable code you can understand, modify, and build on.

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
