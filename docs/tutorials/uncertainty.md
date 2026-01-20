# Uncertainty Quantification with NaturalBoost

NaturalBoost predicts full probability distributions instead of just point estimates, giving you uncertainty bounds on your predictions.

## Why Uncertainty Matters

Traditional gradient boosting gives you a single number: "the price will be $100". But in reality, you might want to know:

- How confident is the model?
- What's the range of likely values?
- What's the probability of exceeding a threshold?

NaturalBoost answers these questions by predicting distribution parameters (e.g., mean and variance for a Normal distribution).

## Basic Usage

```python
import numpy as np
import openboost as ob

# Generate heteroscedastic data (variance depends on X)
np.random.seed(42)
X = np.random.randn(1000, 10).astype(np.float32)
noise_std = 0.5 + np.abs(X[:, 0])  # Noise increases with |X[:, 0]|
y = (X[:, 0] * 2 + X[:, 1] + noise_std * np.random.randn(1000)).astype(np.float32)

X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Train NaturalBoost with Normal distribution
model = ob.NaturalBoostNormal(
    n_trees=100,
    max_depth=4,
    learning_rate=0.1,
)
model.fit(X_train, y_train)

# Point prediction (mean)
mean = model.predict(X_test)

# Prediction interval (90% confidence)
lower, upper = model.predict_interval(X_test, alpha=0.1)

print(f"Coverage: {np.mean((y_test >= lower) & (y_test <= upper)):.1%}")
# Should be close to 90%
```

## Distribution Output

For full control, use `predict_distribution()`:

```python
# Get full distribution output
output = model.predict_distribution(X_test)

# Access distribution parameters
print(f"Mean (loc):  {output.params['loc'][:5]}")
print(f"Std (scale): {output.params['scale'][:5]}")

# Convenience methods
mean = output.mean()
std = output.std()
variance = output.variance()

# Prediction intervals
lower, upper = output.interval(alpha=0.1)  # 90% interval
lower, upper = output.interval(alpha=0.05)  # 95% interval

# Negative log-likelihood (useful for evaluation)
nll = output.nll(y_test)
print(f"Mean NLL: {np.mean(nll):.4f}")
```

## Monte Carlo Sampling

Sample from the predicted distribution for downstream analysis:

```python
# Draw samples from predicted distributions
samples = model.sample(X_test, n_samples=1000)  # Shape: (1000, n_test)

# Use samples for risk analysis
threshold = 10.0
prob_exceed = np.mean(samples > threshold, axis=0)  # P(Y > 10) for each sample

# Quantile estimation
q90 = np.percentile(samples, 90, axis=0)  # 90th percentile
```

## Available Distributions

### Normal (Gaussian)

Best for: General regression with symmetric errors.

```python
model = ob.NaturalBoostNormal(n_trees=100)
model.fit(X_train, y_train)
```

### Log-Normal

Best for: Positive data with right skew (prices, incomes, durations).

```python
model = ob.NaturalBoostLogNormal(n_trees=100)
model.fit(X_train, y_train)  # y must be positive

# Predictions are in original scale
mean = model.predict(X_test)  # E[Y], not E[log(Y)]
```

### Gamma

Best for: Positive continuous data (insurance claims, rainfall).

```python
model = ob.NaturalBoostGamma(n_trees=100)
model.fit(X_train, y_train)  # y must be positive
```

### Poisson

Best for: Count data (number of events, visitors, transactions).

```python
model = ob.NaturalBoostPoisson(n_trees=100)
model.fit(X_train, y_train)  # y: non-negative integers
```

### Student-t

Best for: Data with heavy tails and outliers.

```python
model = ob.NaturalBoostStudentT(n_trees=100)
model.fit(X_train, y_train)
```

### Tweedie

Best for: Zero-inflated positive data (insurance claims, Kaggle competitions).

```python
model = ob.NaturalBoostTweedie(n_trees=100)
model.fit(X_train, y_train)  # y: non-negative, many zeros allowed

# Default power=1.5 (compound Poisson-Gamma)
# Use power closer to 1 for more zeros
# Use power closer to 2 for Gamma-like behavior
```

### Negative Binomial

Best for: Overdispersed count data (sales forecasting, Rossmann competition).

```python
model = ob.NaturalBoostNegBin(n_trees=100)
model.fit(X_train, y_train)  # y: non-negative integers

# Better than Poisson when variance > mean
```

## Comparing Distributions

To choose the right distribution, compare negative log-likelihood:

```python
import openboost as ob

distributions = [
    ('Normal', ob.NaturalBoostNormal),
    ('LogNormal', ob.NaturalBoostLogNormal),
    ('Gamma', ob.NaturalBoostGamma),
]

results = []
for name, ModelClass in distributions:
    model = ModelClass(n_trees=100, max_depth=4)
    model.fit(X_train, y_train)
    nll = model.score(X_test, y_test)  # Mean NLL
    results.append((name, nll))

# Lower NLL is better
for name, nll in sorted(results, key=lambda x: x[1]):
    print(f"{name}: NLL = {nll:.4f}")
```

## Custom Distributions

Define your own distribution:

```python
import openboost as ob
import numpy as np

# Method 1: Use create_custom_distribution (with autodiff)
MyDist = ob.create_custom_distribution(
    name='MyDist',
    param_names=['loc', 'scale'],
    nll_fn=lambda params, y: (
        0.5 * np.log(2 * np.pi * params['scale']**2) +
        0.5 * ((y - params['loc']) / params['scale'])**2
    ),
    mean_fn=lambda params: params['loc'],
    param_transforms={'scale': 'softplus'},  # Ensure scale > 0
)

# Use with NaturalBoost
model = ob.NaturalBoost(distribution=MyDist(), n_trees=100)
model.fit(X_train, y_train)
```

## Calibration

Check if your prediction intervals are well-calibrated:

```python
def check_calibration(model, X_test, y_test, alphas=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """Check if prediction intervals have correct coverage."""
    print("Alpha | Expected | Observed")
    print("-" * 30)
    
    for alpha in alphas:
        lower, upper = model.predict_interval(X_test, alpha=alpha)
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        expected = 1 - alpha
        print(f"{alpha:.2f}  | {expected:.0%}       | {coverage:.1%}")

check_calibration(model, X_test, y_test)
```

## Best Practices

### 1. Choose the Right Distribution

| Data Type | Recommended Distribution |
|-----------|-------------------------|
| General continuous | Normal |
| Positive values | LogNormal, Gamma |
| Count data | Poisson, Negative Binomial |
| Heavy tails | Student-t |
| Zero-inflated positive | Tweedie |
| Overdispersed counts | Negative Binomial |

### 2. Use Appropriate Tree Depth

Shallower trees (max_depth=3-4) often work better for uncertainty estimation than deep trees.

### 3. Train Longer

NaturalBoost learns two (or more) parameters per sample, so it may need more trees than standard GBDT.

```python
model = ob.NaturalBoostNormal(
    n_trees=500,      # More trees
    max_depth=4,      # Shallower trees
    learning_rate=0.05,  # Lower LR for stability
)
```

### 4. Evaluate with NLL

Use negative log-likelihood for model comparison, not just RMSE:

```python
nll = model.score(X_test, y_test)  # Lower is better
```

## Example: Insurance Claims Prediction

```python
import openboost as ob

# Tweedie distribution for insurance claims (many zeros, heavy tail)
model = ob.NaturalBoostTweedie(
    n_trees=300,
    max_depth=4,
    learning_rate=0.05,
)
model.fit(X_train, claims)

# Predict expected claim amount
expected_claim = model.predict(X_new)

# Probability of large claim (risk assessment)
output = model.predict_distribution(X_new)
samples = model.sample(X_new, n_samples=10000)
prob_large_claim = np.mean(samples > 10000, axis=0)

print(f"Expected claim: ${expected_claim[0]:.2f}")
print(f"P(claim > $10k): {prob_large_claim[0]:.1%}")
```

## Next Steps

- [Custom Loss Functions](custom-loss.md) - Define your own objectives
- [Migration from XGBoost](../migration/from-xgboost.md) - Switching from XGBoost
