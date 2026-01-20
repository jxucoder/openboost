# openboost

Core module - main entry point for OpenBoost.

## Quick Reference

```python
import openboost as ob

# Check version and backend
print(ob.__version__)
print(ob.get_backend())  # "cuda" or "cpu"

# Data binning
X_binned = ob.array(X, n_bins=256)

# Models
model = ob.GradientBoosting(n_trees=100)
model = ob.NaturalBoostNormal(n_trees=100)
model = ob.OpenBoostGAM(n_rounds=500)
```

## Data Layer

::: openboost.array
    options:
      show_root_heading: true

::: openboost.BinnedArray
    options:
      show_root_heading: true

## Backend Control

::: openboost.get_backend
    options:
      show_root_heading: true

::: openboost.set_backend
    options:
      show_root_heading: true

::: openboost.is_cuda
    options:
      show_root_heading: true

## Low-Level Tree Building

::: openboost.fit_tree
    options:
      show_root_heading: true

::: openboost.predict_tree
    options:
      show_root_heading: true

::: openboost.predict_ensemble
    options:
      show_root_heading: true
