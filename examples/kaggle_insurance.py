#!/usr/bin/env python
"""Kaggle-style insurance claims example with Tweedie distribution.

This example demonstrates:
- NaturalBoostTweedie for insurance claim prediction
- Zero-inflated positive continuous data
- Full probability distributions for claims
- Probability of exceeding thresholds

The Tweedie distribution is perfect for insurance because:
- Many zeros (no claim)
- Positive continuous values when claim occurs
- Single model predicts both probability and severity

Used in Kaggle competitions like:
- Porto Seguro Safe Driver Prediction
- Allstate Claims Severity
"""

import numpy as np

# OpenBoost imports
import openboost as ob
from openboost import NaturalBoostTweedie

# For data splitting
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def generate_insurance_data(n_samples: int = 5000, seed: int = 42):
    """Generate synthetic insurance claims data.
    
    Simulates:
    - Risk factors (age, driving experience, etc.)
    - Claim probability depending on risk
    - Claim severity depending on risk
    """
    np.random.seed(seed)
    
    # Risk factors
    age = np.random.uniform(18, 70, n_samples).astype(np.float32)
    experience = np.clip(age - 18 + np.random.randn(n_samples) * 3, 0, 50).astype(np.float32)
    accidents_last_5y = np.random.poisson(0.5, n_samples).astype(np.float32)
    speeding_tickets = np.random.poisson(0.3, n_samples).astype(np.float32)
    car_value = np.random.lognormal(10, 0.5, n_samples).astype(np.float32)
    urban = np.random.binomial(1, 0.6, n_samples).astype(np.float32)
    male = np.random.binomial(1, 0.5, n_samples).astype(np.float32)
    credit_score = np.random.normal(700, 80, n_samples).clip(300, 850).astype(np.float32)
    
    X = np.column_stack([
        age, experience, accidents_last_5y, speeding_tickets,
        car_value, urban, male, credit_score
    ])
    
    feature_names = [
        'age', 'experience', 'accidents_last_5y', 'speeding_tickets',
        'car_value', 'urban', 'male', 'credit_score'
    ]
    
    # Risk score affects both claim probability and severity
    risk_score = (
        -0.02 * (age - 40) ** 2 / 100
        - 0.03 * experience
        + 0.4 * accidents_last_5y
        + 0.2 * speeding_tickets
        + 0.1 * np.log(car_value) / 10
        + 0.3 * urban
        + 0.2 * male
        - 0.005 * (credit_score - 600) / 100
    )
    
    # Claim probability: higher risk = higher probability
    claim_prob = 1 / (1 + np.exp(-risk_score))
    claim_prob = np.clip(claim_prob * 0.5, 0.05, 0.4)  # Scale to realistic range
    
    # Has claim
    has_claim = np.random.binomial(1, claim_prob)
    
    # Claim amount (when claim occurs)
    base_severity = np.exp(7 + risk_score)  # ~$1000 base
    claim_amount = np.random.gamma(2, base_severity / 2, n_samples)
    
    # Final: 0 if no claim, positive if claim
    y = (has_claim * claim_amount).astype(np.float32)
    
    return X, y, feature_names, claim_prob, has_claim


def simple_train_test_split(X, y, test_size=0.2, seed=42):
    """Simple train/test split without sklearn."""
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


def main():
    print("=" * 60)
    print("OpenBoost Kaggle Insurance Example (Tweedie)")
    print("=" * 60)
    
    # --- Generate Data ---
    print("\n1. Generating synthetic insurance claims data...")
    X, y, feature_names, true_claim_prob, has_claim = generate_insurance_data(n_samples=5000)
    
    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = simple_train_test_split(X, y)
    
    # Also split the auxiliary info for evaluation
    n_train = len(X_train)
    idx_test = np.arange(len(X)) >= n_train
    true_claim_prob_test = true_claim_prob[idx_test][:len(X_test)]
    
    print(f"   Total samples: {len(X)}")
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Zero claims (train): {(y_train == 0).sum()} ({(y_train == 0).mean():.1%})")
    print(f"   Zero claims (test): {(y_test == 0).sum()} ({(y_test == 0).mean():.1%})")
    print(f"   Mean claim (non-zero): ${y_train[y_train > 0].mean():.2f}")
    
    # --- Train Tweedie Model ---
    print("\n2. Training NaturalBoostTweedie...")
    
    # power=1.5 is typical for insurance (between Poisson and Gamma)
    model = NaturalBoostTweedie(
        power=1.5,
        n_trees=100,
        max_depth=5,
        learning_rate=0.1,
    )
    model.fit(X_train, y_train)
    
    print(f"   Trained {len(model.trees_['mu'])} trees for mu (mean)")
    print(f"   Trained {len(model.trees_['phi'])} trees for phi (dispersion)")
    
    # --- Predictions ---
    print("\n3. Making predictions...")
    
    # Point prediction (expected claim)
    y_pred = model.predict(X_test)
    
    # Full distribution parameters
    params = model.predict_params(X_test)
    pred_mu = params['mu']
    pred_phi = params['phi']
    
    print(f"   Predicted mean range: [${pred_mu.min():.2f}, ${pred_mu.max():.2f}]")
    print(f"   Actual mean (test): ${y_test.mean():.2f}")
    print(f"   Predicted mean (test): ${y_pred.mean():.2f}")
    
    # --- Evaluation ---
    print("\n4. Model evaluation...")
    
    # MSE and RMSE
    mse = ob.mse_score(y_test, y_pred)
    rmse = ob.rmse_score(y_test, y_pred)
    mae = ob.mae_score(y_test, y_pred)
    
    print(f"   MSE:  ${mse:,.2f}")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   MAE:  ${mae:,.2f}")
    
    # NLL (proper probabilistic score)
    nll = model.nll(X_test, y_test)
    print(f"   NLL:  {nll:.4f}")
    
    # --- Risk Segmentation ---
    print("\n5. Risk segmentation analysis...")
    
    # Group by predicted risk
    risk_quantiles = np.percentile(y_pred, [25, 50, 75])
    
    risk_groups = np.digitize(y_pred, risk_quantiles)
    group_names = ['Low Risk', 'Medium-Low', 'Medium-High', 'High Risk']
    
    print("   Risk Group Analysis:")
    for i in range(4):
        mask = risk_groups == i
        if mask.sum() > 0:
            actual_mean = y_test[mask].mean()
            pred_mean = y_pred[mask].mean()
            claim_rate = (y_test[mask] > 0).mean()
            print(f"   {group_names[i]:15} | n={mask.sum():4} | "
                  f"Actual: ${actual_mean:8.2f} | Pred: ${pred_mean:8.2f} | "
                  f"Claim Rate: {claim_rate:.1%}")
    
    # --- Probability of Large Claims ---
    print("\n6. Probability of exceeding claim thresholds...")
    
    # Using sampling to estimate probabilities
    samples = model.sample(X_test, n_samples=1000, seed=42)
    
    thresholds = [1000, 5000, 10000, 25000]
    print("   Sample predictions for probability of claim > threshold:")
    
    for threshold in thresholds:
        # Probability from samples
        prob_exceed = (samples > threshold).mean(axis=1)
        
        # Actual rate in test set
        actual_rate = (y_test > threshold).mean()
        
        print(f"   P(claim > ${threshold:,}) | "
              f"Predicted: {prob_exceed.mean():.3f} | "
              f"Actual: {actual_rate:.3f}")
    
    # --- Per-Sample Risk Assessment ---
    print("\n7. Individual risk assessment (sample cases)...")
    
    # Show a few high-risk and low-risk cases
    high_risk_idx = np.argsort(y_pred)[-3:]  # Top 3 predicted
    low_risk_idx = np.argsort(y_pred)[:3]    # Bottom 3 predicted
    
    print("\n   HIGH RISK cases:")
    for idx in high_risk_idx:
        print(f"   - Predicted: ${y_pred[idx]:,.2f} | Actual: ${y_test[idx]:,.2f}")
        print(f"     Age: {X_test[idx, 0]:.0f}, Exp: {X_test[idx, 1]:.0f}y, "
              f"Accidents: {X_test[idx, 2]:.0f}, Urban: {'Yes' if X_test[idx, 5] else 'No'}")
    
    print("\n   LOW RISK cases:")
    for idx in low_risk_idx:
        print(f"   - Predicted: ${y_pred[idx]:,.2f} | Actual: ${y_test[idx]:,.2f}")
        print(f"     Age: {X_test[idx, 0]:.0f}, Exp: {X_test[idx, 1]:.0f}y, "
              f"Accidents: {X_test[idx, 2]:.0f}, Urban: {'Yes' if X_test[idx, 5] else 'No'}")
    
    # --- Prediction Intervals ---
    print("\n8. Prediction intervals for high-value claims...")
    
    # For non-zero predictions
    nonzero_mask = y_pred > 100
    if nonzero_mask.sum() > 0:
        lower_90, upper_90 = model.predict_interval(X_test[nonzero_mask], alpha=0.1)
        
        # Coverage
        y_nonzero = y_test[nonzero_mask]
        coverage = np.mean((y_nonzero >= lower_90) & (y_nonzero <= upper_90))
        avg_width = (upper_90 - lower_90).mean()
        
        print(f"   90% interval coverage: {coverage:.1%}")
        print(f"   Average interval width: ${avg_width:,.2f}")
    
    # --- Feature Importance ---
    print("\n9. Feature importance for risk factors...")
    
    importances = ob.compute_feature_importances(
        model.trees_['mu'] + model.trees_['phi']
    )
    
    indices = np.argsort(importances)[::-1]
    print("   Risk factors by importance:")
    for i, idx in enumerate(indices):
        print(f"   {i+1}. {feature_names[idx]:20} {importances[idx]:.4f}")
    
    # --- Compare with Simple Model ---
    print("\n10. Comparison: Tweedie vs Simple MSE...")
    
    simple_model = ob.GradientBoosting(
        n_trees=100,
        max_depth=5,
        learning_rate=0.1,
        loss='mse',
    )
    simple_model.fit(X_train, y_train)
    y_pred_simple = simple_model.predict(X_test)
    
    # Clip negative predictions (MSE can go negative)
    y_pred_simple = np.maximum(y_pred_simple, 0)
    
    rmse_tweedie = ob.rmse_score(y_test, y_pred)
    rmse_simple = ob.rmse_score(y_test, y_pred_simple)
    
    print(f"   RMSE (Tweedie): ${rmse_tweedie:,.2f}")
    print(f"   RMSE (MSE):     ${rmse_simple:,.2f}")
    print(f"   Tweedie advantage: +uncertainty quantification, proper zeros handling")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
