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

| Samples | NGBoost | NaturalBoost (GPU) | Speedup |
|---------|---------|-------------------|---------|
| 5,000 | 5.2s | 1.9s | **2.8x** |
| 10,000 | 10.6s | 1.9s | **5.6x** |
| 20,000 | 22.2s | 2.0s | **11.3x** |

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
