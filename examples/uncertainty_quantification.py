#!/usr/bin/env python
"""Uncertainty quantification example with NaturalBoost.

This example demonstrates:
- Training NaturalBoost for probabilistic predictions
- Prediction intervals and uncertainty bounds
- Sampling from predicted distributions
- Heteroscedastic uncertainty (varying noise)
- Proper scoring rules (CRPS, NLL)

NaturalBoost learns full probability distributions, not just point estimates!
"""

import numpy as np

# OpenBoost imports
import openboost as ob
from openboost import (
    NaturalBoostNormal,
    NaturalBoostStudentT,
    DistributionalGBDT,
)

# For data loading
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def generate_heteroscedastic_data(n_samples: int = 2000, seed: int = 42):
    """Generate data with heteroscedastic (varying) noise.
    
    The noise increases with x, demonstrating that NaturalBoost
    can learn to predict higher uncertainty in certain regions.
    """
    np.random.seed(seed)
    X = np.random.uniform(-3, 3, (n_samples, 1)).astype(np.float32)
    
    # Mean is a simple function
    mean = 2 * X[:, 0] + np.sin(2 * X[:, 0])
    
    # Noise increases with |x| - heteroscedastic!
    noise_std = 0.3 + 0.5 * np.abs(X[:, 0])
    noise = np.random.randn(n_samples).astype(np.float32) * noise_std
    
    y = (mean + noise).astype(np.float32)
    return X, y, noise_std


def simple_train_test_split(X, y, test_size=0.2, seed=42):
    """Simple train/test split without sklearn."""
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


def main():
    print("=" * 60)
    print("OpenBoost Uncertainty Quantification Example")
    print("=" * 60)
    
    # --- Generate Heteroscedastic Data ---
    print("\n1. Generating heteroscedastic data...")
    X, y, true_noise_std = generate_heteroscedastic_data(n_samples=2000)
    
    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = simple_train_test_split(X, y)
    
    # Get true noise for test set
    true_noise_test = 0.3 + 0.5 * np.abs(X_test[:, 0])
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   The noise varies from ~0.3 to ~1.8 depending on |x|")
    
    # --- Train NaturalBoost Normal ---
    print("\n2. Training NaturalBoostNormal...")
    
    model = NaturalBoostNormal(
        n_trees=100,
        max_depth=4,
        learning_rate=0.1,
    )
    model.fit(X_train, y_train)
    
    print(f"   Trained {len(model.trees_['loc'])} trees for mean")
    print(f"   Trained {len(model.trees_['scale'])} trees for std")
    
    # --- Point Predictions ---
    print("\n3. Making predictions...")
    
    # Point prediction (mean of distribution)
    y_pred = model.predict(X_test)
    
    # Get full distribution parameters
    params = model.predict_params(X_test)
    pred_mean = params['loc']
    pred_std = params['scale']
    
    print(f"   Predicted mean range: [{pred_mean.min():.2f}, {pred_mean.max():.2f}]")
    print(f"   Predicted std range:  [{pred_std.min():.2f}, {pred_std.max():.2f}]")
    print(f"   True noise std range: [{true_noise_test.min():.2f}, {true_noise_test.max():.2f}]")
    
    # --- Prediction Intervals ---
    print("\n4. Prediction intervals...")
    
    # 90% prediction interval
    lower_90, upper_90 = model.predict_interval(X_test, alpha=0.1)
    coverage_90 = np.mean((y_test >= lower_90) & (y_test <= upper_90))
    
    # 80% prediction interval  
    lower_80, upper_80 = model.predict_interval(X_test, alpha=0.2)
    coverage_80 = np.mean((y_test >= lower_80) & (y_test <= upper_80))
    
    # 50% prediction interval
    lower_50, upper_50 = model.predict_interval(X_test, alpha=0.5)
    coverage_50 = np.mean((y_test >= lower_50) & (y_test <= upper_50))
    
    print(f"   90% interval coverage: {coverage_90:.1%} (target: 90%)")
    print(f"   80% interval coverage: {coverage_80:.1%} (target: 80%)")
    print(f"   50% interval coverage: {coverage_50:.1%} (target: 50%)")
    
    # --- Quantile Predictions ---
    print("\n5. Quantile predictions...")
    
    q10 = model.predict_quantile(X_test, 0.1)
    q50 = model.predict_quantile(X_test, 0.5)
    q90 = model.predict_quantile(X_test, 0.9)
    
    # Check quantile coverage
    below_q10 = np.mean(y_test < q10)
    below_q50 = np.mean(y_test < q50)
    below_q90 = np.mean(y_test < q90)
    
    print(f"   % below Q10: {below_q10:.1%} (target: 10%)")
    print(f"   % below Q50: {below_q50:.1%} (target: 50%)")
    print(f"   % below Q90: {below_q90:.1%} (target: 90%)")
    
    # --- Sampling from Distribution ---
    print("\n6. Sampling from predicted distribution...")
    
    # Draw 1000 samples for each test point
    samples = model.sample(X_test, n_samples=1000, seed=42)
    print(f"   Sample shape: {samples.shape}")
    
    # Compare sample statistics to predicted params
    sample_means = samples.mean(axis=1)
    sample_stds = samples.std(axis=1)
    
    mean_mae = np.mean(np.abs(sample_means - pred_mean))
    std_mae = np.mean(np.abs(sample_stds - pred_std))
    
    print(f"   Sample mean MAE vs predicted: {mean_mae:.4f}")
    print(f"   Sample std MAE vs predicted:  {std_mae:.4f}")
    
    # --- Proper Scoring Rules ---
    print("\n7. Evaluating with proper scoring rules...")
    
    # CRPS (Continuous Ranked Probability Score) - gold standard for probabilistic forecasts
    crps = ob.crps_gaussian(y_test, pred_mean, pred_std)
    print(f"   CRPS (Gaussian): {crps:.4f}")
    
    # Also compute CRPS from samples
    crps_mc = ob.crps_empirical(y_test, samples)
    print(f"   CRPS (Monte Carlo): {crps_mc:.4f}")
    
    # Negative log-likelihood
    nll = ob.negative_log_likelihood(y_test, pred_mean, pred_std)
    print(f"   NLL (mean): {nll:.4f}")
    
    # Interval score
    interval_score_90 = ob.interval_score(y_test, lower_90, upper_90, alpha=0.1)
    print(f"   Interval Score (90%): {interval_score_90:.4f}")
    
    # Pinball loss at different quantiles
    pinball_10 = ob.pinball_loss(y_test, q10, quantile=0.1)
    pinball_90 = ob.pinball_loss(y_test, q90, quantile=0.9)
    print(f"   Pinball loss Q10: {pinball_10:.4f}")
    print(f"   Pinball loss Q90: {pinball_90:.4f}")
    
    # --- Coverage Evaluation Utility ---
    print("\n8. Using evaluate_coverage utility...")
    
    coverage_results = ob.evaluate_coverage(
        y_test, lower_90, upper_90, nominal_coverage=0.9
    )
    print(f"   Empirical coverage: {coverage_results['coverage']:.1%}")
    print(f"   Nominal coverage:   {coverage_results['nominal']:.1%}")
    print(f"   Mean interval width: {coverage_results['mean_width']:.4f}")
    
    # --- Student-t for Heavy Tails ---
    print("\n9. Training NaturalBoostStudentT for heavy tails...")
    
    model_t = NaturalBoostStudentT(
        n_trees=100,
        max_depth=4,
        learning_rate=0.1,
    )
    model_t.fit(X_train, y_train)
    
    # Student-t has df (degrees of freedom) parameter
    params_t = model_t.predict_params(X_test)
    print(f"   Parameters: {list(params_t.keys())}")
    print(f"   Learned df range: [{params_t['df'].min():.1f}, {params_t['df'].max():.1f}]")
    
    # Compare NLL (lower is better)
    nll_normal = model.nll(X_test, y_test)
    nll_student = model_t.nll(X_test, y_test)
    print(f"\n   NLL comparison:")
    print(f"   - Normal:   {nll_normal:.4f}")
    print(f"   - Student-t: {nll_student:.4f}")
    
    # --- Using DistributionOutput for convenience ---
    print("\n10. Using DistributionOutput convenience methods...")
    
    output = model.predict_distribution(X_test)
    
    # All in one object
    print(f"   mean(): {output.mean()[:3]}")
    print(f"   std():  {output.std()[:3]}")
    print(f"   variance(): {output.variance()[:3]}")
    
    # Get interval directly
    lower, upper = output.interval(alpha=0.1)
    print(f"   interval(alpha=0.1): lower={lower[:3]}, upper={upper[:3]}")
    
    # --- Cross-validation with intervals ---
    print("\n11. Cross-validation with prediction intervals...")
    
    cv_model = NaturalBoostNormal(n_trees=50, max_depth=3)
    lower_cv, upper_cv = ob.cross_val_predict_interval(cv_model, X, y, alpha=0.1, cv=3)
    coverage_cv = np.mean((y >= lower_cv) & (y <= upper_cv))
    print(f"   OOF 90% coverage: {coverage_cv:.1%}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
