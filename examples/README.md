> **Historical implementation:** this page describes the retired pre-rebuild API.
> Reproduce at Git revision `50acfc6`; see the repository README and `v1-sprints/` for current v1 status.

# OpenBoost Examples

Runnable scripts. Distributional regression first; mean-regression GBDT after.

```bash
uv run python examples/uncertainty_quantification.py
```

## Overview

| Example | Description | Key APIs |
|---------|-------------|----------|
| [uncertainty_quantification.py](uncertainty_quantification.py) | Full distribution, intervals, CRPS | `NaturalBoostNormal` |
| [kaggle_insurance.py](kaggle_insurance.py) | Zero-inflated claims | `NaturalBoostTweedie` |
| [kaggle_sales.py](kaggle_sales.py) | Overdispersed counts | `NaturalBoostNegBin` |
| [basic_regression.py](basic_regression.py) | Point-estimate regression | `GradientBoosting` |
| [binary_classification.py](binary_classification.py) | Binary classifier | `OpenBoostClassifier` |
| [multiclass_classification.py](multiclass_classification.py) | Softmax multi-class | `MultiClassGradientBoosting` |
| [custom_loss.py](custom_loss.py) | Quantile / Huber / asymmetric | custom `loss=` |
| [gpu_training.py](gpu_training.py) | Backend selection | `set_backend` |
| [gam_explainability.py](gam_explainability.py) | Interpretable main effects | `OpenBoostGAM` |
| [sklearn_pipeline.py](sklearn_pipeline.py) | `Pipeline` / `GridSearchCV` | `OpenBoostRegressor` |
| [model_persistence.py](model_persistence.py) | `save` / `load` | `PersistenceMixin` |

FormulaBoost and WeibullAFT do not have example scripts yet; copy from the
docs:

- [FormulaBoost](https://jxucoder.github.io/openboost/user-guide/formulaboost/)
- [Weibull AFT](https://jxucoder.github.io/openboost/user-guide/survival/)
- [Quickstart](https://jxucoder.github.io/openboost/getting-started/quickstart/)

## Uncertainty (`uncertainty_quantification.py`)

NaturalBoost: intervals, quantiles, sampling, CRPS / NLL, heteroscedastic
noise.

## Insurance (`kaggle_insurance.py`)

`NaturalBoostTweedie` for zero-inflated positive claims. Risk
segmentation, P(large claim), vs a plain MSE GBDT.

## Sales (`kaggle_sales.py`)

`NaturalBoostNegBin` for overdispersed counts. Service levels, vs Poisson.

## Point-estimate GBDT

`basic_regression.py`, `binary_classification.py`,
`multiclass_classification.py` cover `GradientBoosting` / sklearn
wrappers, callbacks, and feature importance. Use these when you want a
number, not `F(y | x)`. For production MSE/logloss at scale,
XGBoost / LightGBM are still faster C++.

## Custom loss (`custom_loss.py`)

Quantile, Huber, asymmetric, log-cosh. Each returns `(grad, hess)`. For a
**formula** with coupled parameters, this is the wrong tool; use
FormulaBoost (full GGN), not a diagonal custom objective.

## GPU (`gpu_training.py`)

Backend detection, pinning, and a small wall-clock check. Headline A100
numbers live in [benchmarks](https://jxucoder.github.io/openboost/benchmarks/).

## GAM (`gam_explainability.py`)

Main-effect shape functions, per-feature contributions.

## Requirements

```bash
pip install --pre openboost
pip install scikit-learn matplotlib   # sklearn + plots
pip install --pre "openboost[cuda]"   # GPU example
```

## Tips

1. Start with `uncertainty_quantification.py` if you care about intervals.
2. Confirm CUDA with `gpu_training.py` (`ob.is_cuda()`).
3. Insurance / sales examples are the distribution-family templates.
4. `custom_loss.py` is for point-estimate objectives; FormulaBoost is for
   `y = f(θ, x)`.

## Troubleshooting

**Import errors.** `pip install --pre openboost` (without `--pre` you get
the older stable release).

**GPU not detected.** `nvidia-smi`, then
`python -c "from numba import cuda; print(list(cuda.gpus))"`.

**Headless plots.** Examples write figures to files when no display is
available.
