#!/usr/bin/env python
"""Multi-class classification example with OpenBoost.

This example demonstrates:
- Multi-class classification with softmax loss
- Per-class probability predictions
- Confusion matrix analysis
- Feature importance across classes

Dataset: Iris (built into sklearn) or synthetic
"""

import numpy as np

# OpenBoost imports
import openboost as ob
from openboost import MultiClassGradientBoosting

# For data loading
try:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def generate_multiclass_data(n_samples: int = 1000, n_classes: int = 4, seed: int = 42):
    """Generate synthetic multi-class classification data."""
    np.random.seed(seed)
    
    n_per_class = n_samples // n_classes
    X_list, y_list = [], []
    
    for c in range(n_classes):
        # Each class has a different center
        center = np.array([c * 2, (c % 2) * 2, np.sin(c)])
        X_c = np.random.randn(n_per_class, 3).astype(np.float32) + center
        y_c = np.full(n_per_class, c, dtype=np.int32)
        X_list.append(X_c)
        y_list.append(y_c)
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx].astype(np.float32)


def simple_train_test_split(X, y, test_size=0.2, seed=42):
    """Simple train/test split without sklearn."""
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


def main():
    print("=" * 60)
    print("OpenBoost Multi-Class Classification Example")
    print("=" * 60)
    
    # --- Load Data ---
    print("\n1. Loading data...")
    
    if SKLEARN_AVAILABLE:
        iris = load_iris()
        X, y = iris.data.astype(np.float32), iris.target.astype(np.float32)
        feature_names = iris.feature_names
        class_names = iris.target_names
        print(f"   Dataset: Iris")
    else:
        X, y = generate_multiclass_data(n_samples=1000, n_classes=4)
        feature_names = ['feature_0', 'feature_1', 'feature_2']
        class_names = ['class_0', 'class_1', 'class_2', 'class_3']
        print(f"   Dataset: Synthetic")
    
    n_classes = len(np.unique(y))
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   Classes: {n_classes} - {list(class_names)}")
    print(f"   Class distribution: {np.bincount(y.astype(int))}")
    
    # Split data
    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = simple_train_test_split(X, y)
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # --- Train Multi-Class Model ---
    print("\n2. Training MultiClassGradientBoosting...")
    
    model = MultiClassGradientBoosting(
        n_classes=n_classes,
        n_trees=50,
        max_depth=4,
        learning_rate=0.1,
    )
    model.fit(X_train, y_train)
    
    print(f"   Trees per class: {model.n_trees}")
    print(f"   Total trees: {model.n_trees * n_classes}")
    
    # --- Predictions ---
    print("\n3. Making predictions...")
    
    # Class predictions
    y_pred = model.predict(X_test)
    
    # Probability predictions (softmax)
    y_proba = model.predict_proba(X_test)
    
    print(f"   Prediction shape: {y_pred.shape}")
    print(f"   Probability shape: {y_proba.shape}")
    print(f"   Sample probabilities: {y_proba[0]}")
    
    # --- Evaluation ---
    print("\n4. Evaluation...")
    
    # Accuracy
    accuracy = ob.accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    print("\n   Per-class metrics:")
    for c in range(n_classes):
        mask = y_test == c
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == c).mean()
            class_proba = y_proba[mask, c].mean()
            print(f"   {class_names[c]:15} | Acc: {class_acc:.2%} | Avg prob: {class_proba:.3f} | n={mask.sum()}")
    
    # --- Confusion Matrix ---
    print("\n5. Confusion matrix...")
    
    # Build confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_test.astype(int), y_pred.astype(int)):
        conf_matrix[true, pred] += 1
    
    # Print
    header = "True\\Pred  " + "  ".join(f"{class_names[c][:8]:>8}" for c in range(n_classes))
    print(f"   {header}")
    print("   " + "-" * len(header))
    for c in range(n_classes):
        row = f"   {class_names[c][:8]:8} |"
        for p in range(n_classes):
            if c == p:
                row += f"  [{conf_matrix[c, p]:4}]"  # Diagonal
            else:
                row += f"   {conf_matrix[c, p]:4} "
        print(row)
    
    # --- Confidence Analysis ---
    print("\n6. Prediction confidence analysis...")
    
    # Max probability (confidence)
    confidence = y_proba.max(axis=1)
    
    # Calibration: high confidence should correlate with correct predictions
    correct = (y_pred == y_test)
    
    conf_bins = [0.0, 0.5, 0.7, 0.9, 1.0]
    print("   Confidence  | Accuracy | Count")
    print("   " + "-" * 35)
    for i in range(len(conf_bins) - 1):
        mask = (confidence >= conf_bins[i]) & (confidence < conf_bins[i+1])
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            print(f"   {conf_bins[i]:.1f} - {conf_bins[i+1]:.1f}   | {bin_acc:8.1%} | {mask.sum():5}")
    
    # --- Feature Importance ---
    print("\n7. Feature importance...")
    
    # Get all trees across classes
    all_trees = []
    for class_trees in model.trees_:
        all_trees.extend(class_trees)
    
    if all_trees:
        importances = ob.compute_feature_importances(all_trees)
        
        indices = np.argsort(importances)[::-1]
        print("   Features by importance:")
        for i, idx in enumerate(indices):
            bar = "#" * int(importances[idx] / importances.max() * 20)
            print(f"   {i+1}. {feature_names[idx]:20} {importances[idx]:.4f} {bar}")
    
    # --- Misclassification Analysis ---
    print("\n8. Misclassification analysis...")
    
    misclassified = y_pred != y_test
    n_errors = misclassified.sum()
    
    print(f"   Total errors: {n_errors} ({n_errors/len(y_test):.1%})")
    
    if n_errors > 0:
        print("\n   Error breakdown:")
        error_pairs = {}
        for i, (true, pred) in enumerate(zip(y_test[misclassified].astype(int), 
                                              y_pred[misclassified].astype(int))):
            key = (class_names[true], class_names[pred])
            error_pairs[key] = error_pairs.get(key, 0) + 1
        
        for (true_name, pred_name), count in sorted(error_pairs.items(), key=lambda x: -x[1]):
            print(f"   {true_name} -> {pred_name}: {count}")
    
    # --- Low Confidence Predictions ---
    print("\n9. Low confidence predictions (potential edge cases)...")
    
    low_conf_mask = confidence < 0.6
    n_low_conf = low_conf_mask.sum()
    
    print(f"   Predictions with <60% confidence: {n_low_conf} ({n_low_conf/len(y_test):.1%})")
    
    if n_low_conf > 0:
        low_conf_acc = correct[low_conf_mask].mean()
        print(f"   Accuracy on low-confidence: {low_conf_acc:.1%}")
        
        # Show a few examples
        print("\n   Sample low-confidence predictions:")
        low_conf_idx = np.where(low_conf_mask)[0][:5]
        for idx in low_conf_idx:
            true_class = class_names[int(y_test[idx])]
            pred_class = class_names[int(y_pred[idx])]
            conf = confidence[idx]
            probs = y_proba[idx]
            
            prob_str = ", ".join(f"{class_names[c][:3]}:{p:.2f}" for c, p in enumerate(probs))
            status = "✓" if y_pred[idx] == y_test[idx] else "✗"
            print(f"   {status} True: {true_class:10} | Pred: {pred_class:10} | Conf: {conf:.2f} | [{prob_str}]")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
