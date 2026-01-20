#!/usr/bin/env python
"""Basic regression example with OpenBoost.

This example demonstrates:
- Training a basic GradientBoosting model for regression
- Making predictions and evaluating performance
- Feature importance analysis
- Using callbacks (early stopping, logging)

Dataset: California Housing (built into sklearn)
"""

import numpy as np

# OpenBoost imports
import openboost as ob
from openboost import (
    GradientBoosting,
    OpenBoostRegressor,
    EarlyStopping,
    Logger,
    compute_feature_importances,
)

# For data loading and evaluation
try:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def generate_synthetic_data(n_samples: int = 2000, n_features: int = 8, seed: int = 42):
    """Generate synthetic regression data if sklearn not available."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Non-linear relationship
    y = (
        2 * X[:, 0] 
        + X[:, 1] ** 2 
        - 0.5 * X[:, 2] * X[:, 3]
        + np.sin(X[:, 4])
        + np.random.randn(n_samples).astype(np.float32) * 0.5
    )
    return X, y


def main():
    print("=" * 60)
    print("OpenBoost Basic Regression Example")
    print("=" * 60)
    
    # --- Load Data ---
    print("\n1. Loading data...")
    if SKLEARN_AVAILABLE:
        housing = fetch_california_housing()
        X, y = housing.data.astype(np.float32), housing.target.astype(np.float32)
        feature_names = housing.feature_names
        print(f"   Dataset: California Housing")
    else:
        X, y = generate_synthetic_data()
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        print(f"   Dataset: Synthetic (sklearn not available)")
    
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    ) if SKLEARN_AVAILABLE else (X[:1600], X[1600:], y[:1600], y[1600:])
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    ) if SKLEARN_AVAILABLE else (X_train[:1400], X_train[1400:], y_train[:1400], y_train[1400:])
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # --- Method 1: Low-level GradientBoosting API ---
    print("\n2. Training with low-level GradientBoosting API...")
    
    model = GradientBoosting(
        n_trees=200,
        max_depth=6,
        learning_rate=0.1,
        loss='mse',
        subsample=0.8,
        reg_lambda=1.0,
    )
    
    # With callbacks
    callbacks = [
        EarlyStopping(patience=20, min_delta=0.001),
        Logger(every_n=50),
    ]
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        callbacks=callbacks,
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = ob.mse_score(y_test, y_pred)
    rmse = ob.rmse_score(y_test, y_pred)
    r2 = ob.r2_score(y_test, y_pred)
    
    print(f"\n   Results (GradientBoosting):")
    print(f"   - MSE:  {mse:.4f}")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - R²:   {r2:.4f}")
    print(f"   - Trees trained: {len(model.trees_)}")
    
    # --- Feature Importance ---
    print("\n3. Feature Importance Analysis...")
    
    importances = compute_feature_importances(model.trees_)
    
    print("   Top features by gain:")
    indices = np.argsort(importances)[::-1]
    for i, idx in enumerate(indices[:5]):
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # --- Method 2: sklearn-compatible API ---
    print("\n4. Training with sklearn-compatible API...")
    
    sklearn_model = OpenBoostRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        early_stopping_rounds=20,
    )
    
    sklearn_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    y_pred_sklearn = sklearn_model.predict(X_test)
    r2_sklearn = sklearn_model.score(X_test, y_test)
    
    print(f"   R² (sklearn API): {r2_sklearn:.4f}")
    print(f"   Best iteration: {sklearn_model.best_iteration_}")
    
    # Access feature importances via sklearn API
    print(f"   Feature importances via .feature_importances_:")
    for i, idx in enumerate(np.argsort(sklearn_model.feature_importances_)[::-1][:3]):
        print(f"   {i+1}. {feature_names[idx]}: {sklearn_model.feature_importances_[idx]:.4f}")
    
    # --- Method 3: Automatic Parameter Suggestion ---
    print("\n5. Using suggest_params for automatic configuration...")
    
    suggested = ob.suggest_params(X_train, y_train, task='regression')
    print(f"   Suggested parameters: {suggested}")
    
    # --- Cross-Validation ---
    print("\n6. Out-of-fold predictions with cross_val_predict...")
    
    cv_model = OpenBoostRegressor(n_estimators=50, max_depth=4)
    oof_predictions = ob.cross_val_predict(cv_model, X, y, cv=3)
    oof_r2 = ob.r2_score(y, oof_predictions)
    print(f"   OOF R²: {oof_r2:.4f}")
    
    # --- Backend Info ---
    print("\n7. Backend information:")
    print(f"   Backend: {ob.get_backend()}")
    print(f"   Using GPU: {ob.is_cuda()}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
