#!/usr/bin/env python
"""Binary classification example with OpenBoost.

This example demonstrates:
- Training a GradientBoosting model for binary classification
- Using logloss objective
- ROC AUC evaluation
- Probability calibration metrics
- sklearn-compatible classifier API

Dataset: Breast Cancer Wisconsin (built into sklearn)
"""

import numpy as np

# OpenBoost imports
import openboost as ob
from openboost import (
    GradientBoosting,
    OpenBoostClassifier,
    EarlyStopping,
    Logger,
)

# For data loading
try:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def generate_synthetic_classification(n_samples: int = 1000, n_features: int = 10, seed: int = 42):
    """Generate synthetic binary classification data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Create a decision boundary
    logits = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] * X[:, 3]
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(n_samples) < probs).astype(np.float32)
    return X, y


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def main():
    print("=" * 60)
    print("OpenBoost Binary Classification Example")
    print("=" * 60)
    
    # --- Load Data ---
    print("\n1. Loading data...")
    if SKLEARN_AVAILABLE:
        cancer = load_breast_cancer()
        X, y = cancer.data.astype(np.float32), cancer.target.astype(np.float32)
        feature_names = cancer.feature_names
        print(f"   Dataset: Breast Cancer Wisconsin")
    else:
        X, y = generate_synthetic_classification()
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        print(f"   Dataset: Synthetic (sklearn not available)")
    
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   Class distribution: {np.bincount(y.astype(int))}")
    
    # Split data
    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
    else:
        n = len(X)
        idx = np.random.permutation(n)
        X, y = X[idx], y[idx]
        X_train, X_val, X_test = X[:700], X[700:850], X[850:]
        y_train, y_val, y_test = y[:700], y[700:850], y[850:]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # --- Method 1: Low-level API with logloss ---
    print("\n2. Training with low-level GradientBoosting API...")
    
    model = GradientBoosting(
        n_trees=100,
        max_depth=4,
        learning_rate=0.1,
        loss='logloss',  # Binary cross-entropy
        subsample=0.8,
        reg_lambda=1.0,
    )
    
    callbacks = [
        EarlyStopping(patience=15, min_delta=0.001),
        Logger(every_n=25),
    ]
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        callbacks=callbacks,
    )
    
    # Raw predictions (logits)
    logits = model.predict(X_test)
    y_proba = sigmoid(logits)
    y_pred = (y_proba > 0.5).astype(np.float32)
    
    # Metrics
    accuracy = ob.accuracy_score(y_test, y_pred)
    auc = ob.roc_auc_score(y_test, y_proba)
    logloss = ob.log_loss_score(y_test, y_proba)
    
    print(f"\n   Results (GradientBoosting):")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - ROC AUC:  {auc:.4f}")
    print(f"   - Log Loss: {logloss:.4f}")
    print(f"   - Trees trained: {len(model.trees_)}")
    
    # --- Method 2: sklearn-compatible Classifier API ---
    print("\n3. Training with sklearn-compatible Classifier API...")
    
    clf = OpenBoostClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        early_stopping_rounds=15,
    )
    
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # The classifier API handles sigmoid internally
    y_proba_clf = clf.predict_proba(X_test)[:, 1]  # Probability of class 1
    y_pred_clf = clf.predict(X_test)
    
    accuracy_clf = ob.accuracy_score(y_test, y_pred_clf)
    auc_clf = ob.roc_auc_score(y_test, y_proba_clf)
    
    print(f"   Accuracy (sklearn API): {accuracy_clf:.4f}")
    print(f"   ROC AUC (sklearn API):  {auc_clf:.4f}")
    print(f"   Best iteration: {clf.best_iteration_}")
    
    # --- Calibration Analysis ---
    print("\n4. Calibration analysis...")
    
    # Brier score (lower is better)
    brier = ob.brier_score(y_test, y_proba_clf)
    print(f"   Brier Score: {brier:.4f}")
    
    # Expected Calibration Error
    ece = ob.expected_calibration_error(y_test, y_proba_clf, n_bins=10)
    print(f"   ECE (10 bins): {ece:.4f}")
    
    # Calibration curve data
    frac_pos, mean_pred, counts = ob.calibration_curve(y_test, y_proba_clf, n_bins=5)
    print("   Calibration curve (predicted vs actual):")
    for i, (pred, actual, n) in enumerate(zip(mean_pred, frac_pos, counts)):
        if n > 0:
            print(f"     Bin {i+1}: predicted={pred:.2f}, actual={actual:.2f} (n={n})")
    
    # --- Feature Importance ---
    print("\n5. Feature importance...")
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("   Top 5 features:")
    for i, idx in enumerate(indices[:5]):
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # --- Cross-Validation with Probabilities ---
    print("\n6. Out-of-fold probability predictions...")
    
    cv_clf = OpenBoostClassifier(n_estimators=50, max_depth=3)
    oof_proba = ob.cross_val_predict_proba(cv_clf, X, y, cv=3)
    oof_auc = ob.roc_auc_score(y, oof_proba[:, 1])
    print(f"   OOF ROC AUC: {oof_auc:.4f}")
    
    # --- Precision, Recall, F1 ---
    print("\n7. Additional metrics at threshold=0.5...")
    
    precision = ob.precision_score(y_test, y_pred_clf)
    recall = ob.recall_score(y_test, y_pred_clf)
    f1 = ob.f1_score(y_test, y_pred_clf)
    
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
