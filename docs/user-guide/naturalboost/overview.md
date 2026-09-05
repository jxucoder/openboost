# NaturalBoost

NaturalBoost is **distributional regression** via boosting: each
parameter of `F(y | x)` (`loc`, `scale`, …) is its own ensemble, stepped
with natural gradient. That is the
[GAMLSS](https://doi.org/10.1111/j.1467-9876.2005.00510.x) / NGBoost
model class, with a GPU histogram-tree path.

For a structural formula that is not a distribution, use
[FormulaBoost](../formulaboost.md) (varying-coefficient). For
right-censored Weibull survival, use
[WeibullAFT](../survival.md). Shared engine:
[How it works](../how-it-works.md).

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

## Comparison with NGBoost

Compare held-out NLL, CRPS and calibration at matched training budgets, then
measure end-to-end fit/prediction on explicitly recorded hardware. A GPU/CPU
comparison must state the different resources and include transfer/compilation
costs. See [available evidence and reproduction](../../benchmarks.md).

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

## Related

- [How it works](../how-it-works.md): the shared engine
- [FormulaBoost](../formulaboost.md): when `f` is a formula, not a distribution
- [Weibull AFT](../survival.md): censored survival
- [Custom distributions](custom-distributions.md)
- [Benchmarks](../../benchmarks.md)

### Execution and reproducibility boundary

`NaturalBoost` and `DistributionalGBDT` accept `random_state` for CPU row and
column sampling. Repeating a fit with the same integer seed uses the same
sampling stream without changing NumPy's global RNG. `FormulaBoost` and
`WeibullAFT` share this trainer behavior. With `random_state=None`, each fit
creates its own unseeded generator. Legacy trainers have separate RNG paths.

The unified CUDA trainer currently requires `subsample=1` and
`colsample_bytree=1`; other sampling ratios fail before binning or updates.
Only exact built-in Normal and Poisson distributions use device objective
kernels. Custom distributions, subclasses and exposure offsets use host
objective math with a visible warning. A native-tree fallback also warns;
these mixed execution paths must be distinguished in benchmark provenance.
Kernel compilation or execution errors propagate as failed fits.

Sample weights multiply both gradient and Hessian. Weighted fits disable the
unit-Hessian bandwidth hint, including uniform weights. The foundation P2
real-device weighted parity gate is pending; this is not a new GPU quality or
performance claim. Initialization still uses the existing unweighted
`distribution.init_params(y)` estimate.
