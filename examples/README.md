# OpenBoost Examples

This directory contains runnable examples demonstrating OpenBoost's capabilities.

## Quick Start

```bash
# Run any example
uv run python examples/basic_regression.py

# Or with standard Python
python examples/basic_regression.py
```

## Examples Overview

| Example | Description | Key Features |
|---------|-------------|--------------|
| [basic_regression.py](basic_regression.py) | Standard gradient boosting for regression | `GradientBoosting`, callbacks, feature importance |
| [binary_classification.py](binary_classification.py) | Binary classification with probability outputs | `OpenBoostClassifier`, ROC AUC, calibration |
| [multiclass_classification.py](multiclass_classification.py) | Multi-class classification with softmax | `MultiClassGradientBoosting`, confusion matrix |
| [uncertainty_quantification.py](uncertainty_quantification.py) | Probabilistic predictions with uncertainty | `NaturalBoostNormal`, prediction intervals, CRPS |
| [kaggle_insurance.py](kaggle_insurance.py) | Insurance claims with Tweedie distribution | `NaturalBoostTweedie`, zero-inflated data |
| [kaggle_sales.py](kaggle_sales.py) | Sales forecasting with Negative Binomial | `NaturalBoostNegBin`, overdispersed counts |
| [custom_loss.py](custom_loss.py) | Custom loss functions | Quantile, Huber, asymmetric losses |
| [gpu_training.py](gpu_training.py) | GPU acceleration guide | Backend selection, benchmarking |
| [gam_explainability.py](gam_explainability.py) | Interpretable GAM models | `OpenBoostGAM`, shape functions |
| [sklearn_pipeline.py](sklearn_pipeline.py) | sklearn Pipeline integration | `Pipeline`, `GridSearchCV`, preprocessing |
| [model_persistence.py](model_persistence.py) | Saving and loading models | `save()`, `load()`, checkpointing |

## Detailed Descriptions

### Basic Regression (`basic_regression.py`)

Learn the fundamentals of OpenBoost with a standard regression task.

**Topics covered:**
- Training `GradientBoosting` with various hyperparameters
- Using callbacks (`EarlyStopping`, `Logger`)
- Computing feature importances
- sklearn-compatible API with `OpenBoostRegressor`
- Cross-validation utilities

### Binary Classification (`binary_classification.py`)

Train a binary classifier with probability calibration analysis.

**Topics covered:**
- Binary classification with `logloss` objective
- `OpenBoostClassifier` sklearn wrapper
- ROC AUC, precision, recall, F1 metrics
- Calibration analysis (Brier score, ECE)
- Out-of-fold probability predictions

### Uncertainty Quantification (`uncertainty_quantification.py`)

The power of NaturalBoost: full probability distributions, not just point estimates!

**Topics covered:**
- Training `NaturalBoostNormal` for probabilistic predictions
- Prediction intervals (90%, 80%, 50%)
- Quantile predictions
- Sampling from predicted distributions
- Proper scoring rules (CRPS, NLL)
- Heteroscedastic uncertainty

### Kaggle Insurance (`kaggle_insurance.py`)

Tweedie distribution for insurance claim prediction (like Porto Seguro, Allstate).

**Topics covered:**
- `NaturalBoostTweedie` for zero-inflated positive continuous data
- Risk segmentation analysis
- Probability of large claims
- Individual risk assessment
- Comparison with simple MSE model

### Kaggle Sales (`kaggle_sales.py`)

Negative Binomial for sales/demand forecasting (like Rossmann, Bike Sharing).

**Topics covered:**
- `NaturalBoostNegBin` for overdispersed count data
- Inventory planning (service levels)
- Day-of-week and promotional effects
- Probability of high demand
- Comparison with Poisson model

### Custom Loss Functions (`custom_loss.py`)

Build any loss function you need!

**Topics covered:**
- Quantile regression for different percentiles
- Huber loss for outlier robustness
- Asymmetric loss for business costs
- Log-cosh smooth approximation
- How to write custom loss functions

### GPU Training (`gpu_training.py`)

Get the most out of GPU acceleration.

**Topics covered:**
- Automatic GPU detection
- Manual backend selection
- Performance benchmarking
- Best practices for GPU training
- Multi-GPU training overview

### GAM Explainability (`gam_explainability.py`)

Interpretable machine learning with `OpenBoostGAM`.

**Topics covered:**
- Training interpretable GAM models
- Visualizing shape functions
- Per-feature contribution analysis
- Explaining individual predictions
- Trade-offs vs black-box models

## Requirements

All examples work with the base OpenBoost installation:

```bash
pip install openboost
```

Some examples benefit from optional dependencies:

```bash
# For sklearn integration examples
pip install scikit-learn

# For visualization
pip install matplotlib

# For GPU examples
pip install numba  # CUDA support included
```

## Running Examples

### Local Development

```bash
# From the repository root
cd openboost

# Run with uv
uv run python examples/basic_regression.py

# Or standard Python
python examples/basic_regression.py
```

### In a Notebook

```python
# Copy-paste code from examples into Jupyter/Colab cells
import openboost as ob

model = ob.GradientBoosting(n_trees=100)
model.fit(X_train, y_train)
```

### On Cloud (Modal, etc.)

```python
# Examples work on cloud GPU instances
import modal

app = modal.App()

@app.function(gpu="A100")
def train_model():
    import openboost as ob
    # ... example code ...
```

## Tips

1. **Start simple**: Begin with `basic_regression.py` to understand the API
2. **Check GPU**: Run `gpu_training.py` to verify GPU setup
3. **Explore uncertainty**: `uncertainty_quantification.py` shows NaturalBoost's unique value
4. **For Kaggle**: `kaggle_insurance.py` and `kaggle_sales.py` are ready-to-adapt templates
5. **Custom needs**: `custom_loss.py` shows how to extend OpenBoost

## Troubleshooting

**Example won't run?**
- Ensure OpenBoost is installed: `pip install openboost`
- For sklearn examples: `pip install scikit-learn`

**GPU not detected?**
- Check CUDA installation: `nvidia-smi`
- Ensure numba is installed: `pip install numba`
- See `gpu_training.py` for debugging tips

**Plots not showing?**
- Install matplotlib: `pip install matplotlib`
- In headless environments, plots save to files

## Contributing

Have a cool example to share? PRs welcome!

Guidelines:
- Self-contained (generates synthetic data or uses sklearn datasets)
- Well-commented
- Demonstrates a clear use case
- Follows existing style
