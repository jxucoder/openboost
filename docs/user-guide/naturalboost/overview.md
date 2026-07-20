# NaturalBoost Overview

NaturalBoost predicts full probability distributions instead of just point estimates, giving you uncertainty bounds on your predictions.

## Why Uncertainty Matters

Traditional gradient boosting gives you a single number: "the price will be $100". But in reality, you might want to know:

- How confident is the model?
- What's the range of likely values?
- What's the probability of exceeding a threshold?

NaturalBoost answers these questions by predicting distribution parameters (e.g., mean and variance for a Normal distribution).

## Quick Start

```python
import numpy as np
import openboost as ob

# Train probabilistic model
model = ob.NaturalBoostNormal(n_trees=100, max_depth=4)
model.fit(X_train, y_train)

# Point prediction (mean)
mean = model.predict(X_test)

# 90% prediction interval
lower, upper = model.predict_interval(X_test, alpha=0.1)

# Check coverage
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"Coverage: {coverage:.1%}")  # Should be ~90%
```

## Available Models

| Model | Distribution | Use Case |
|-------|--------------|----------|
| `NaturalBoostNormal` | Gaussian | General uncertainty |
| `NaturalBoostLogNormal` | Log-Normal | Positive skewed (prices) |
| `NaturalBoostGamma` | Gamma | Positive continuous |
| `NaturalBoostPoisson` | Poisson | Count data |
| `NaturalBoostStudentT` | Student-t | Heavy tails, outliers |
| `NaturalBoostTweedie` | Tweedie | Insurance claims (Kaggle!) |
| `NaturalBoostNegBin` | Negative Binomial | Sales forecasting (Kaggle!) |

## Distribution Output

For full control, use `predict_distribution()`:

```python
output = model.predict_distribution(X_test)

# Access distribution parameters
mean = output.mean()
std = output.std()
variance = output.variance()

# Prediction intervals
lower, upper = output.interval(alpha=0.1)  # 90% interval

# Negative log-likelihood
nll = output.nll(y_test)
print(f"Mean NLL: {np.mean(nll):.4f}")
```

## Monte Carlo Sampling

Sample from the predicted distribution for downstream analysis:

```python
# Draw samples
samples = model.sample(X_test, n_samples=1000)  # Shape: (1000, n_test)

# Risk analysis
threshold = 10.0
prob_exceed = np.mean(samples > threshold, axis=0)  # P(Y > 10)

# Quantile estimation
q90 = np.percentile(samples, 90, axis=0)
```

## Performance vs NGBoost

Measured head-to-head against NGBoost 0.5.11 (both with a Normal distribution,
natural gradient, and an identical budget: 500 boosting rounds, learning rate 0.03,
depth-3 trees, seed 42, same train/test splits). This is a CPU-vs-CPU comparison —
NGBoost is CPU-only, and OpenBoost's GPU tree path is deliberately **not** measured
here. NLL and CRPS use the same closed-form Gaussian formulas for both models.
Full configs, metrics, and library versions are committed in
`benchmarks/results/ngboost_comparison_20260720.json`. Reproduce with:

```bash
OPENBOOST_BACKEND=cpu uv run --with ngboost python benchmarks/bench_ngboost_comparison.py
```

| Dataset | Fit time OB / NGB | Test NLL OB / NGB | CRPS OB / NGB | RMSE OB / NGB |
|---------|-------------------|-------------------|---------------|---------------|
| Synthetic heteroscedastic, 10K | **16.4s** / 18.8s (1.15x) | **2.124** / 2.134 | **1.169** / 1.174 | **2.145** / 2.149 |
| Synthetic heteroscedastic, 50K | **74.1s** / 95.3s (1.29x) | 2.122 / **2.116** | 1.184 / **1.177** | 2.165 / **2.152** |
| California Housing, 20.6K | 30.6s / **25.0s** (0.82x) | **0.572** / 0.575 | **0.255** / 0.256 | **0.518** / 0.521 |

The honest read: on CPU the two libraries are comparable. NaturalBoost's
histogram-based trees are modestly faster on the larger synthetic dataset (1.29x at
50K samples), while NGBoost was faster on California Housing (0.82x). Prediction
quality is essentially tied — NGBoost slightly wins NLL/CRPS/RMSE on the 50K
synthetic dataset; NaturalBoost slightly wins on the other two. NaturalBoost's main
differentiators are the GPU tree path and the wider distribution/custom-distribution
support, not raw CPU speed.

## Best Practices

1. **Use shallower trees** (`max_depth=3-4`) - better for uncertainty estimation
2. **Train longer** - NaturalBoost learns 2+ parameters per sample
3. **Evaluate with NLL** - not just RMSE

```python
model = ob.NaturalBoostNormal(
    n_trees=500,      # More trees
    max_depth=4,      # Shallower
    learning_rate=0.05,  # Lower LR
)
```
