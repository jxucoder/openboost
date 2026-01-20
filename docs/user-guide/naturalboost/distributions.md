# Distributions

NaturalBoost supports multiple probability distributions for different data types.

## Choosing a Distribution

| Data Type | Recommended Distribution |
|-----------|-------------------------|
| General continuous | Normal |
| Positive values | LogNormal, Gamma |
| Count data | Poisson, Negative Binomial |
| Heavy tails | Student-t |
| Zero-inflated positive | Tweedie |
| Overdispersed counts | Negative Binomial |

## Normal (Gaussian)

Best for: General regression with symmetric errors.

```python
model = ob.NaturalBoostNormal(n_trees=100)
model.fit(X_train, y_train)

mean = model.predict(X_test)
lower, upper = model.predict_interval(X_test, alpha=0.1)
```

## Log-Normal

Best for: Positive data with right skew (prices, incomes, durations).

```python
model = ob.NaturalBoostLogNormal(n_trees=100)
model.fit(X_train, y_train)  # y must be positive

# Predictions are in original scale
mean = model.predict(X_test)  # E[Y], not E[log(Y)]
```

## Gamma

Best for: Positive continuous data (insurance claims, rainfall).

```python
model = ob.NaturalBoostGamma(n_trees=100)
model.fit(X_train, y_train)  # y must be positive
```

## Poisson

Best for: Count data (number of events, visitors, transactions).

```python
model = ob.NaturalBoostPoisson(n_trees=100)
model.fit(X_train, y_train)  # y: non-negative integers
```

## Student-t

Best for: Data with heavy tails and outliers.

```python
model = ob.NaturalBoostStudentT(n_trees=100)
model.fit(X_train, y_train)
```

## Tweedie

Best for: Zero-inflated positive data (insurance claims, Kaggle competitions).

```python
model = ob.NaturalBoostTweedie(n_trees=100)
model.fit(X_train, y_train)  # y: non-negative, many zeros allowed

# Default power=1.5 (compound Poisson-Gamma)
# Use power closer to 1 for more zeros
# Use power closer to 2 for Gamma-like behavior
```

## Negative Binomial

Best for: Overdispersed count data (sales forecasting, Rossmann competition).

```python
model = ob.NaturalBoostNegBin(n_trees=100)
model.fit(X_train, y_train)  # y: non-negative integers

# Better than Poisson when variance > mean
```

## Comparing Distributions

Use negative log-likelihood to choose the best distribution:

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
