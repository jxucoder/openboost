#!/usr/bin/env python
"""Custom loss function example with OpenBoost.

This example demonstrates:
- Defining custom loss functions with gradient and hessian
- Quantile regression for different percentiles
- Asymmetric loss functions
- Huber loss for outlier robustness
- Using custom losses with GradientBoosting

OpenBoost accepts any callable with signature: (pred, y) -> (grad, hess)
"""

import numpy as np

# OpenBoost imports
import openboost as ob
from openboost import GradientBoosting

# For data
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def generate_data_with_outliers(n_samples: int = 2000, outlier_frac: float = 0.1, seed: int = 42):
    """Generate regression data with outliers."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, 5).astype(np.float32)
    
    # True relationship
    y = 2 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2]
    
    # Add noise
    noise = np.random.randn(n_samples).astype(np.float32) * 0.5
    
    # Add outliers (large positive deviations)
    n_outliers = int(n_samples * outlier_frac)
    outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
    noise[outlier_idx] += np.random.uniform(5, 15, n_outliers).astype(np.float32)
    
    y = (y + noise).astype(np.float32)
    return X, y, outlier_idx


def simple_train_test_split(X, y, test_size=0.2, seed=42):
    """Simple train/test split without sklearn."""
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


# =============================================================================
# Custom Loss Functions
# =============================================================================

def quantile_loss(tau: float):
    """
    Quantile loss function (pinball loss).
    
    For tau=0.5, this is equivalent to MAE and predicts the median.
    For tau=0.1, predicts the 10th percentile.
    For tau=0.9, predicts the 90th percentile.
    
    Args:
        tau: Quantile to predict (0 < tau < 1)
    
    Returns:
        Loss function callable
    """
    def loss_fn(pred: np.ndarray, y: np.ndarray) -> tuple:
        residual = y - pred
        
        # Gradient: -tau if residual > 0, else (1-tau)
        grad = np.where(residual > 0, -tau, 1 - tau)
        
        # Hessian: constant (for stability)
        hess = np.ones_like(pred)
        
        return grad.astype(np.float32), hess.astype(np.float32)
    
    return loss_fn


def huber_loss(delta: float = 1.0):
    """
    Huber loss - robust to outliers.
    
    L(r) = 0.5 * r^2           if |r| <= delta
    L(r) = delta * |r| - 0.5 * delta^2  if |r| > delta
    
    Gradient: r if |r| <= delta, else delta * sign(r)
    Hessian: 1 if |r| <= delta, else 0 (we use small value for stability)
    
    Args:
        delta: Threshold for switching from quadratic to linear
    
    Returns:
        Loss function callable
    """
    def loss_fn(pred: np.ndarray, y: np.ndarray) -> tuple:
        residual = pred - y  # Note: pred - y for gradient descent
        abs_residual = np.abs(residual)
        
        # Gradient
        grad = np.where(
            abs_residual <= delta,
            residual,  # Quadratic region
            delta * np.sign(residual)  # Linear region
        )
        
        # Hessian
        hess = np.where(
            abs_residual <= delta,
            np.ones_like(pred),  # Quadratic region
            0.01 * np.ones_like(pred)  # Linear region (small for stability)
        )
        
        return grad.astype(np.float32), hess.astype(np.float32)
    
    return loss_fn


def asymmetric_loss(under_weight: float = 1.0, over_weight: float = 2.0):
    """
    Asymmetric squared error loss.
    
    Penalizes over-prediction and under-prediction differently.
    Useful when costs are asymmetric (e.g., inventory: stockout vs overstock).
    
    Args:
        under_weight: Weight for under-prediction (pred < y)
        over_weight: Weight for over-prediction (pred > y)
    
    Returns:
        Loss function callable
    """
    def loss_fn(pred: np.ndarray, y: np.ndarray) -> tuple:
        residual = pred - y
        
        # Weight based on sign of residual
        weights = np.where(residual > 0, over_weight, under_weight)
        
        # Weighted MSE gradient: 2 * w * (pred - y)
        grad = 2 * weights * residual
        
        # Weighted MSE hessian: 2 * w
        hess = 2 * weights
        
        return grad.astype(np.float32), hess.astype(np.float32)
    
    return loss_fn


def log_cosh_loss():
    """
    Log-cosh loss - smooth approximation to MAE.
    
    L(r) = log(cosh(r))
    Gradient: tanh(r)
    Hessian: sech^2(r) = 1 - tanh^2(r)
    
    Behaves like MSE for small residuals, like MAE for large ones.
    
    Returns:
        Loss function callable
    """
    def loss_fn(pred: np.ndarray, y: np.ndarray) -> tuple:
        residual = pred - y
        
        # Clip for numerical stability
        residual = np.clip(residual, -20, 20)
        
        # Gradient: tanh(r)
        grad = np.tanh(residual)
        
        # Hessian: 1 - tanh^2(r)
        hess = 1 - grad ** 2
        hess = np.maximum(hess, 0.01)  # Minimum hessian for stability
        
        return grad.astype(np.float32), hess.astype(np.float32)
    
    return loss_fn


def focal_loss_regression(gamma: float = 2.0):
    """
    Focal loss adapted for regression.
    
    Down-weights well-predicted samples, focuses on hard examples.
    Inspired by focal loss for classification.
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
    
    Returns:
        Loss function callable
    """
    def loss_fn(pred: np.ndarray, y: np.ndarray) -> tuple:
        residual = pred - y
        abs_res = np.abs(residual)
        
        # Focal weight: focuses on large residuals
        # w = |r|^gamma normalized
        w = (abs_res + 1e-6) ** gamma
        w = w / w.mean()  # Normalize
        
        # Weighted MSE
        grad = 2 * w * residual
        hess = 2 * w
        hess = np.maximum(hess, 0.01)
        
        return grad.astype(np.float32), hess.astype(np.float32)
    
    return loss_fn


def main():
    print("=" * 60)
    print("OpenBoost Custom Loss Function Example")
    print("=" * 60)
    
    # --- Generate Data ---
    print("\n1. Generating data with outliers...")
    X, y, outlier_idx = generate_data_with_outliers(n_samples=2000, outlier_frac=0.1)
    
    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = simple_train_test_split(X, y)
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Outliers: ~10% of samples have large positive deviations")
    print(f"   Mean y: {y.mean():.2f}, Median y: {np.median(y):.2f}")
    
    # --- Baseline: Standard MSE ---
    print("\n2. Training baseline with MSE loss...")
    
    mse_model = GradientBoosting(
        n_trees=100,
        max_depth=4,
        learning_rate=0.1,
        loss='mse',
    )
    mse_model.fit(X_train, y_train)
    y_pred_mse = mse_model.predict(X_test)
    
    mae_mse = ob.mae_score(y_test, y_pred_mse)
    print(f"   MSE model - MAE: {mae_mse:.4f}")
    
    # --- Huber Loss (Outlier Robust) ---
    print("\n3. Training with Huber loss (robust to outliers)...")
    
    huber_model = GradientBoosting(
        n_trees=100,
        max_depth=4,
        learning_rate=0.1,
        loss=huber_loss(delta=1.5),  # Custom loss!
    )
    huber_model.fit(X_train, y_train)
    y_pred_huber = huber_model.predict(X_test)
    
    mae_huber = ob.mae_score(y_test, y_pred_huber)
    print(f"   Huber model - MAE: {mae_huber:.4f}")
    print(f"   Improvement over MSE: {(mae_mse - mae_huber) / mae_mse * 100:.1f}%")
    
    # --- Quantile Regression ---
    print("\n4. Training quantile regression models...")
    
    quantiles = [0.1, 0.5, 0.9]
    quantile_preds = {}
    
    for tau in quantiles:
        model = GradientBoosting(
            n_trees=100,
            max_depth=4,
            learning_rate=0.1,
            loss=quantile_loss(tau=tau),
        )
        model.fit(X_train, y_train)
        quantile_preds[tau] = model.predict(X_test)
    
    # Evaluate quantile calibration
    for tau in quantiles:
        below = np.mean(y_test < quantile_preds[tau])
        print(f"   Q{tau:.0%}: {below:.1%} below (target: {tau:.0%})")
    
    # Coverage check
    coverage = np.mean(
        (y_test >= quantile_preds[0.1]) & (y_test <= quantile_preds[0.9])
    )
    print(f"   10%-90% interval coverage: {coverage:.1%} (target: 80%)")
    
    # --- Asymmetric Loss ---
    print("\n5. Training with asymmetric loss (penalize over-prediction more)...")
    
    asym_model = GradientBoosting(
        n_trees=100,
        max_depth=4,
        learning_rate=0.1,
        loss=asymmetric_loss(under_weight=1.0, over_weight=3.0),
    )
    asym_model.fit(X_train, y_train)
    y_pred_asym = asym_model.predict(X_test)
    
    over_pred_mse = np.mean(y_pred_mse > y_test)
    over_pred_asym = np.mean(y_pred_asym > y_test)
    
    print(f"   MSE model - over-predictions: {over_pred_mse:.1%}")
    print(f"   Asymmetric model - over-predictions: {over_pred_asym:.1%}")
    print(f"   (Asymmetric loss pushes predictions lower)")
    
    # --- Log-Cosh Loss ---
    print("\n6. Training with log-cosh loss (smooth MAE approximation)...")
    
    logcosh_model = GradientBoosting(
        n_trees=100,
        max_depth=4,
        learning_rate=0.1,
        loss=log_cosh_loss(),
    )
    logcosh_model.fit(X_train, y_train)
    y_pred_logcosh = logcosh_model.predict(X_test)
    
    mae_logcosh = ob.mae_score(y_test, y_pred_logcosh)
    print(f"   Log-cosh model - MAE: {mae_logcosh:.4f}")
    
    # --- Focal Loss ---
    print("\n7. Training with focal loss (focus on hard examples)...")
    
    focal_model = GradientBoosting(
        n_trees=100,
        max_depth=4,
        learning_rate=0.1,
        loss=focal_loss_regression(gamma=2.0),
    )
    focal_model.fit(X_train, y_train)
    y_pred_focal = focal_model.predict(X_test)
    
    mae_focal = ob.mae_score(y_test, y_pred_focal)
    print(f"   Focal model - MAE: {mae_focal:.4f}")
    
    # --- Summary Comparison ---
    print("\n" + "=" * 60)
    print("Summary: MAE on test set")
    print("=" * 60)
    
    results = [
        ("MSE (baseline)", mae_mse),
        ("Huber (delta=1.5)", mae_huber),
        ("Log-cosh", mae_logcosh),
        ("Focal (gamma=2)", mae_focal),
        ("Quantile (median)", ob.mae_score(y_test, quantile_preds[0.5])),
    ]
    
    results.sort(key=lambda x: x[1])
    
    for name, mae in results:
        best = " <- best" if mae == results[0][1] else ""
        print(f"   {name:25} MAE: {mae:.4f}{best}")
    
    # --- Using Built-in Loss Functions ---
    print("\n8. Using built-in loss functions from OpenBoost...")
    
    # OpenBoost provides common losses
    print("   Available built-in losses:")
    print("   - 'mse': Mean Squared Error")
    print("   - 'mae': Mean Absolute Error")
    print("   - 'huber': Huber loss")
    print("   - 'logloss': Binary cross-entropy")
    print("   - 'softmax': Multi-class cross-entropy")
    print("   - 'poisson': Poisson deviance")
    print("   - 'gamma': Gamma deviance")
    print("   - 'tweedie': Tweedie deviance")
    
    # Example with built-in MAE
    mae_builtin = GradientBoosting(
        n_trees=100,
        max_depth=4,
        learning_rate=0.1,
        loss='mae',
    )
    mae_builtin.fit(X_train, y_train)
    y_pred_mae = mae_builtin.predict(X_test)
    print(f"\n   Built-in MAE loss - MAE: {ob.mae_score(y_test, y_pred_mae):.4f}")
    
    # --- Custom Loss Tips ---
    print("\n" + "=" * 60)
    print("Tips for Custom Loss Functions")
    print("=" * 60)
    print("""
   1. Return gradient and hessian as float32 arrays
   2. Hessian should be positive (use minimum value for stability)
   3. Clip values to avoid numerical issues
   4. Test on synthetic data first
   5. Compare with built-in losses as sanity check
   
   Signature: def my_loss(pred: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]
   
   Example template:
   
   def custom_loss(pred, y):
       residual = pred - y
       
       # Your gradient
       grad = ...  # d(loss)/d(pred)
       
       # Your hessian
       hess = ...  # d^2(loss)/d(pred)^2
       hess = np.maximum(hess, 1e-6)  # Ensure positive
       
       return grad.astype(np.float32), hess.astype(np.float32)
    """)
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
