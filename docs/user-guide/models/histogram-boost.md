# HistogramBoost

`HistogramBoost` predicts a flexible probability histogram instead of assuming
a Normal, Gamma, or other parametric family. Each bin represents uniform
density, and the model trains the complete CDF with that histogram's exact
continuous ranked probability score (CRPS).

Each boosting round learns one tree structure. Every leaf stores a vector of
updates for the ordered histogram logits. A softmax converts those logits into
non-negative probabilities that sum to one, so predicted CDFs are monotone and
quantiles cannot cross.

```python
import openboost as ob

model = ob.HistogramBoost(
    n_distribution_bins=50,
    n_trees=100,
    learning_rate=0.05,
    max_depth=6,
)
model.fit(X_train, y_train)

distribution = model.predict_distribution(X_test)
mean = distribution.mean()
lower, upper = distribution.interval(alpha=0.1)
samples = distribution.sample(n_samples=100, seed=42)
```

## When to use it

Use `HistogramBoost` when a two-parameter distribution is too restrictive—for
example when conditional outcomes may be skewed or have more than one mode—and
CRPS is the primary quality target. Use NaturalBoost when a named distribution,
analytic likelihood, exposure offset, or distribution-specific interpretation
is important.

## Current boundary

This is a CPU-only development model. It currently supports numeric features,
including missing values, and sample weights. CUDA, categorical splits,
callbacks, evaluation sets, early stopping, and row/column sampling are not yet
implemented.

The target grid is learned from the training target range. Predictions cannot
put mass outside that finite range plus its half-bin padding, so held-out
extremes may be clipped. Always evaluate CRPS together with coverage, interval
score, RMSE, and failure rate.

The default configuration is frozen for the repository's preregistered
ScoringBench development experiment. It is not yet an overall leaderboard or
state-of-the-art claim.

## Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `n_distribution_bins` | 50 | Number of ordered target bins |
| `n_trees` | 100 | Shared-vector boosting rounds |
| `learning_rate` | 0.05 | Shrinkage applied to each tree |
| `max_depth` | 6 | Maximum routing depth |
| `n_feature_bins` | 254 | Numeric feature histogram bins |
| `curvature_scale` | 1.0 | Scale of the PSD Gauss–Newton diagonal |
| `reg_lambda` | 1.0 | L2 regularization used to select split structures |
| `leaf_reg_lambda` | `None` | L2 regularization for vector leaves; `None` reuses `reg_lambda` |
| `base_smoothing` | 1.0 | Total Dirichlet prior weight, spread evenly across target bins |
| `reg_alpha` | 0.0 | L1 regularization for vector leaf values |

`predict_distribution()` returns `HistogramDistributionOutput`, which provides
`mean()`, `variance()`, `std()`, exact `crps()`, `tempered()`, density-preserving
`subdivide()`, `quantile()`, `interval()`, and `sample()`.
