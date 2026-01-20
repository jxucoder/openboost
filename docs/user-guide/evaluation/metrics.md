# Evaluation Metrics

OpenBoost includes thin wrappers around sklearn metrics with full sample weight support.

## Regression Metrics

```python
import openboost as ob

mse = ob.mse_score(y_true, y_pred)
mae = ob.mae_score(y_true, y_pred)
rmse = ob.rmse_score(y_true, y_pred)
r2 = ob.r2_score(y_true, y_pred)

# With sample weights
weighted_mse = ob.mse_score(y_true, y_pred, sample_weight=weights)
```

| Metric | Function | Description |
|--------|----------|-------------|
| MSE | `mse_score` | Mean squared error |
| MAE | `mae_score` | Mean absolute error |
| RMSE | `rmse_score` | Root mean squared error |
| R² | `r2_score` | Coefficient of determination |

## Classification Metrics

```python
import openboost as ob

acc = ob.accuracy_score(y_true, y_pred)
auc = ob.roc_auc_score(y_true, y_proba)
logloss = ob.log_loss_score(y_true, y_proba)
f1 = ob.f1_score(y_true, y_pred)
precision = ob.precision_score(y_true, y_pred)
recall = ob.recall_score(y_true, y_pred)

# With sample weights
weighted_auc = ob.roc_auc_score(y_true, y_proba, sample_weight=weights)
```

| Metric | Function | Description |
|--------|----------|-------------|
| Accuracy | `accuracy_score` | Classification accuracy |
| ROC AUC | `roc_auc_score` | Area under ROC curve |
| Log Loss | `log_loss_score` | Cross-entropy loss |
| F1 | `f1_score` | Harmonic mean of precision/recall |
| Precision | `precision_score` | True positives / predicted positives |
| Recall | `recall_score` | True positives / actual positives |

## Probabilistic Metrics

For NaturalBoost and distributional models:

```python
import openboost as ob
import numpy as np

# Train probabilistic model
model = ob.NaturalBoostNormal(n_trees=100)
model.fit(X_train, y_train)

# Get predictions
output = model.predict_distribution(X_test)
mean = output.mean()
std = np.sqrt(output.variance())

# CRPS - gold standard for probabilistic forecasting
crps = ob.crps_gaussian(y_test, mean, std)

# For non-Gaussian, use Monte Carlo samples
samples = model.sample(X_test, n_samples=1000)
crps_mc = ob.crps_empirical(y_test, samples)

# Negative log-likelihood
nll = ob.negative_log_likelihood(y_test, mean, std)

# Prediction interval evaluation
lower, upper = model.predict_interval(X_test, alpha=0.1)
score = ob.interval_score(y_test, lower, upper, alpha=0.1)

# Pinball loss for quantiles
q90 = np.percentile(samples, 90, axis=0)
loss = ob.pinball_loss(y_test, q90, quantile=0.9)
```

| Metric | Function | Use Case |
|--------|----------|----------|
| CRPS (Gaussian) | `crps_gaussian` | Probabilistic regression |
| CRPS (Empirical) | `crps_empirical` | Any distribution |
| Brier Score | `brier_score` | Probabilistic classification |
| Pinball Loss | `pinball_loss` | Quantile regression |
| Interval Score | `interval_score` | Prediction intervals |
| NLL | `negative_log_likelihood` | Likelihood evaluation |

## Calibration Metrics

```python
import openboost as ob

# For classifiers
classifier = ob.OpenBoostClassifier(n_estimators=100)
classifier.fit(X_train, y_train)
y_proba = classifier.predict_proba(X_test)[:, 1]

# Brier score
brier = ob.brier_score(y_test, y_proba)

# Expected Calibration Error
ece = ob.expected_calibration_error(y_test, y_proba, n_bins=10)

# Reliability diagram data
frac_pos, mean_pred, counts = ob.calibration_curve(y_test, y_proba, n_bins=10)
```

| Metric | Function | Description |
|--------|----------|-------------|
| Brier Score | `brier_score` | Proper scoring rule for probabilities |
| ECE | `expected_calibration_error` | Calibration quality |
| Calibration Curve | `calibration_curve` | Data for reliability diagrams |
