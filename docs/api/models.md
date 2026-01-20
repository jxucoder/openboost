# Models

All OpenBoost model classes.

## Standard GBDT

### GradientBoosting

::: openboost.GradientBoosting
    options:
      show_root_heading: true
      members:
        - __init__
        - fit
        - predict
        - save
        - load

### MultiClassGradientBoosting

::: openboost.MultiClassGradientBoosting
    options:
      show_root_heading: true
      members:
        - __init__
        - fit
        - predict
        - predict_proba

### DART

::: openboost.DART
    options:
      show_root_heading: true
      members:
        - __init__
        - fit
        - predict

## Interpretable Models

### OpenBoostGAM

::: openboost.OpenBoostGAM
    options:
      show_root_heading: true
      members:
        - __init__
        - fit
        - predict
        - get_feature_importance
        - plot_shape_function

### LinearLeafGBDT

::: openboost.LinearLeafGBDT
    options:
      show_root_heading: true
      members:
        - __init__
        - fit
        - predict

## Probabilistic Models (NaturalBoost)

### NaturalBoost

::: openboost.NaturalBoost
    options:
      show_root_heading: true
      members:
        - __init__
        - fit
        - predict
        - predict_interval
        - predict_distribution
        - sample
        - score

### NaturalBoostNormal

::: openboost.NaturalBoostNormal
    options:
      show_root_heading: true

### NaturalBoostLogNormal

::: openboost.NaturalBoostLogNormal
    options:
      show_root_heading: true

### NaturalBoostGamma

::: openboost.NaturalBoostGamma
    options:
      show_root_heading: true

### NaturalBoostPoisson

::: openboost.NaturalBoostPoisson
    options:
      show_root_heading: true

### NaturalBoostStudentT

::: openboost.NaturalBoostStudentT
    options:
      show_root_heading: true

### NaturalBoostTweedie

::: openboost.NaturalBoostTweedie
    options:
      show_root_heading: true

### NaturalBoostNegBin

::: openboost.NaturalBoostNegBin
    options:
      show_root_heading: true

## sklearn Wrappers

### OpenBoostRegressor

::: openboost.OpenBoostRegressor
    options:
      show_root_heading: true
      members:
        - __init__
        - fit
        - predict
        - score

### OpenBoostClassifier

::: openboost.OpenBoostClassifier
    options:
      show_root_heading: true
      members:
        - __init__
        - fit
        - predict
        - predict_proba
        - score

### OpenBoostDistributionalRegressor

::: openboost.OpenBoostDistributionalRegressor
    options:
      show_root_heading: true

### OpenBoostLinearLeafRegressor

::: openboost.OpenBoostLinearLeafRegressor
    options:
      show_root_heading: true
