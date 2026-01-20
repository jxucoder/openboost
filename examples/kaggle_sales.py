#!/usr/bin/env python
"""Kaggle-style sales forecasting example with Negative Binomial distribution.

This example demonstrates:
- NaturalBoostNegBin for sales/demand prediction
- Overdispersed count data (variance > mean)
- Full probability distributions for inventory planning
- Probability of stockouts

The Negative Binomial distribution is perfect for sales because:
- Naturally handles count data (0, 1, 2, ... items sold)
- Captures overdispersion (high variance on weekends, holidays)
- Single model predicts both expected sales and uncertainty

Used in Kaggle competitions like:
- Rossmann Store Sales
- Bike Sharing Demand
- Walmart Sales Forecasting
"""

import numpy as np

# OpenBoost imports
import openboost as ob
from openboost import NaturalBoostNegBin

# For data splitting
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def generate_sales_data(n_samples: int = 5000, seed: int = 42):
    """Generate synthetic retail sales data.
    
    Simulates:
    - Day of week effects
    - Promotional effects
    - Store characteristics
    - Seasonality (month)
    """
    np.random.seed(seed)
    
    # Time features
    day_of_week = np.random.randint(0, 7, n_samples).astype(np.float32)
    is_weekend = (day_of_week >= 5).astype(np.float32)
    month = np.random.randint(1, 13, n_samples).astype(np.float32)
    is_holiday = np.random.binomial(1, 0.05, n_samples).astype(np.float32)
    
    # Store features
    store_size = np.random.choice([1, 2, 3], n_samples).astype(np.float32)  # small, medium, large
    store_age = np.random.uniform(1, 20, n_samples).astype(np.float32)
    competition_dist = np.random.lognormal(1, 0.5, n_samples).astype(np.float32)
    
    # Marketing
    promo = np.random.binomial(1, 0.3, n_samples).astype(np.float32)
    promo_strength = promo * np.random.uniform(1, 3, n_samples).astype(np.float32)
    
    X = np.column_stack([
        day_of_week, is_weekend, month, is_holiday,
        store_size, store_age, competition_dist,
        promo, promo_strength
    ])
    
    feature_names = [
        'day_of_week', 'is_weekend', 'month', 'is_holiday',
        'store_size', 'store_age', 'competition_dist',
        'promo', 'promo_strength'
    ]
    
    # Expected sales (log scale)
    log_mu = (
        4.0  # base ~55 items
        + 0.3 * is_weekend
        + 0.5 * is_holiday
        + 0.3 * (store_size - 2)  # larger stores sell more
        + 0.02 * store_age
        - 0.1 * np.log(competition_dist + 1)
        + 0.5 * promo
        + 0.2 * promo_strength
        + 0.1 * np.sin(2 * np.pi * month / 12)  # seasonality
    )
    
    mu = np.exp(log_mu)
    
    # Overdispersion (variance / mean ratio)
    # Higher on weekends/holidays
    r = 3 + 2 * is_weekend + 5 * is_holiday
    
    # Generate from negative binomial
    # NB params: mean = mu, variance = mu + mu^2/r
    p = r / (r + mu)
    y = np.random.negative_binomial(r.astype(int), p).astype(np.float32)
    
    return X, y, feature_names, mu


def simple_train_test_split(X, y, test_size=0.2, seed=42):
    """Simple train/test split without sklearn."""
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


def main():
    print("=" * 60)
    print("OpenBoost Kaggle Sales Forecasting Example (Negative Binomial)")
    print("=" * 60)
    
    # --- Generate Data ---
    print("\n1. Generating synthetic retail sales data...")
    X, y, feature_names, true_mu = generate_sales_data(n_samples=5000)
    
    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = simple_train_test_split(X, y)
    
    print(f"   Total samples: {len(X)}")
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Mean sales: {y.mean():.1f} items")
    print(f"   Variance: {y.var():.1f} (overdispersed: var/mean = {y.var()/y.mean():.1f})")
    print(f"   Range: [{y.min():.0f}, {y.max():.0f}] items")
    
    # --- Train Negative Binomial Model ---
    print("\n2. Training NaturalBoostNegBin...")
    
    model = NaturalBoostNegBin(
        n_trees=100,
        max_depth=5,
        learning_rate=0.1,
    )
    model.fit(X_train, y_train)
    
    print(f"   Trained {len(model.trees_['mu'])} trees for mu (mean)")
    print(f"   Trained {len(model.trees_['r'])} trees for r (dispersion)")
    
    # --- Predictions ---
    print("\n3. Making predictions...")
    
    # Point prediction (expected sales)
    y_pred = model.predict(X_test)
    
    # Full distribution parameters
    params = model.predict_params(X_test)
    pred_mu = params['mu']
    pred_r = params['r']
    
    # Predicted variance: mu + mu^2/r
    pred_var = pred_mu + pred_mu**2 / pred_r
    
    print(f"   Predicted mean range: [{pred_mu.min():.1f}, {pred_mu.max():.1f}] items")
    print(f"   Predicted r (dispersion): [{pred_r.min():.1f}, {pred_r.max():.1f}]")
    print(f"   Actual test mean: {y_test.mean():.1f}, Predicted: {y_pred.mean():.1f}")
    
    # --- Evaluation ---
    print("\n4. Model evaluation...")
    
    mse = ob.mse_score(y_test, y_pred)
    rmse = ob.rmse_score(y_test, y_pred)
    mae = ob.mae_score(y_test, y_pred)
    
    print(f"   MSE:  {mse:.2f}")
    print(f"   RMSE: {rmse:.2f} items")
    print(f"   MAE:  {mae:.2f} items")
    
    # NLL
    nll = model.nll(X_test, y_test)
    print(f"   NLL:  {nll:.4f}")
    
    # --- Inventory Planning ---
    print("\n5. Inventory planning analysis...")
    
    # Sample to estimate stockout probabilities
    samples = model.sample(X_test, n_samples=1000, seed=42)
    
    # Service level analysis: What inventory level achieves X% service?
    service_levels = [0.90, 0.95, 0.99]
    
    print("   Required inventory for service level targets:")
    for sl in service_levels:
        required_inventory = np.percentile(samples, sl * 100, axis=1)
        avg_required = required_inventory.mean()
        avg_demand = y_pred.mean()
        safety_stock = avg_required - avg_demand
        print(f"   {sl:.0%} service: {avg_required:.0f} items "
              f"(safety stock: +{safety_stock:.0f})")
    
    # --- Day-of-Week Analysis ---
    print("\n6. Day-of-week demand patterns...")
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    print("   Day       | Pred Mean | Pred Var | Actual Mean")
    print("   " + "-" * 50)
    
    for day in range(7):
        mask = X_test[:, 0] == day
        if mask.sum() > 0:
            pred_mean_day = y_pred[mask].mean()
            pred_var_day = (pred_mu[mask] + pred_mu[mask]**2 / pred_r[mask]).mean()
            actual_mean_day = y_test[mask].mean()
            print(f"   {day_names[day]:9} | {pred_mean_day:9.1f} | {pred_var_day:8.1f} | {actual_mean_day:11.1f}")
    
    # --- Promo Effect Analysis ---
    print("\n7. Promotional effect analysis...")
    
    promo_mask = X_test[:, 7] == 1
    no_promo_mask = X_test[:, 7] == 0
    
    if promo_mask.sum() > 0 and no_promo_mask.sum() > 0:
        promo_mean = y_pred[promo_mask].mean()
        no_promo_mean = y_pred[no_promo_mask].mean()
        lift = (promo_mean - no_promo_mean) / no_promo_mean * 100
        
        print(f"   With promo:    {promo_mean:.1f} items (n={promo_mask.sum()})")
        print(f"   Without promo: {no_promo_mean:.1f} items (n={no_promo_mask.sum()})")
        print(f"   Promotional lift: +{lift:.1f}%")
    
    # --- Prediction Intervals ---
    print("\n8. Prediction intervals...")
    
    lower_80, upper_80 = model.predict_interval(X_test, alpha=0.2)
    lower_95, upper_95 = model.predict_interval(X_test, alpha=0.05)
    
    coverage_80 = np.mean((y_test >= lower_80) & (y_test <= upper_80))
    coverage_95 = np.mean((y_test >= lower_95) & (y_test <= upper_95))
    
    print(f"   80% interval coverage: {coverage_80:.1%} (target: 80%)")
    print(f"   95% interval coverage: {coverage_95:.1%} (target: 95%)")
    print(f"   Avg 80% interval width: {(upper_80 - lower_80).mean():.1f} items")
    
    # --- Probability of High Demand ---
    print("\n9. Probability of demand exceeding thresholds...")
    
    thresholds = [50, 100, 150, 200]
    
    for threshold in thresholds:
        prob_exceed = (samples > threshold).mean(axis=1)
        actual_rate = (y_test > threshold).mean()
        print(f"   P(demand > {threshold:3d}) | "
              f"Predicted: {prob_exceed.mean():.3f} | "
              f"Actual: {actual_rate:.3f}")
    
    # --- Feature Importance ---
    print("\n10. Feature importance...")
    
    importances = ob.compute_feature_importances(
        model.trees_['mu'] + model.trees_['r']
    )
    
    indices = np.argsort(importances)[::-1]
    print("   Demand drivers by importance:")
    for i, idx in enumerate(indices):
        print(f"   {i+1}. {feature_names[idx]:20} {importances[idx]:.4f}")
    
    # --- Compare with Poisson ---
    print("\n11. Comparison: Negative Binomial vs Poisson...")
    
    poisson_model = ob.NaturalBoostPoisson(
        n_trees=100,
        max_depth=5,
        learning_rate=0.1,
    )
    poisson_model.fit(X_train, y_train)
    
    nll_negbin = model.nll(X_test, y_test)
    nll_poisson = poisson_model.nll(X_test, y_test)
    
    print(f"   NLL (Negative Binomial): {nll_negbin:.4f}")
    print(f"   NLL (Poisson):           {nll_poisson:.4f}")
    print(f"   NegBin is better when variance > mean (overdispersion)")
    
    # --- Sample Forecast Output ---
    print("\n12. Sample forecast for next week (simulated)...")
    
    # Create a sample week's data for a store
    sample_week = np.array([
        [0, 0, 6, 0, 2, 10, 5, 0, 0],  # Monday, no promo
        [1, 0, 6, 0, 2, 10, 5, 0, 0],  # Tuesday
        [2, 0, 6, 0, 2, 10, 5, 1, 2],  # Wednesday, with promo
        [3, 0, 6, 0, 2, 10, 5, 1, 2],  # Thursday, with promo
        [4, 0, 6, 0, 2, 10, 5, 0, 0],  # Friday
        [5, 1, 6, 0, 2, 10, 5, 0, 0],  # Saturday (weekend)
        [6, 1, 6, 0, 2, 10, 5, 0, 0],  # Sunday (weekend)
    ], dtype=np.float32)
    
    week_pred = model.predict(sample_week)
    week_lower, week_upper = model.predict_interval(sample_week, alpha=0.2)
    
    print("   Forecast (Medium store, June, 10yr old, competition ~5km):")
    print("   Day       | Forecast | 80% Interval")
    print("   " + "-" * 40)
    for i, day in enumerate(day_names):
        print(f"   {day:9} | {week_pred[i]:8.0f} | [{week_lower[i]:.0f}, {week_upper[i]:.0f}]")
    
    print(f"\n   Weekly total: {week_pred.sum():.0f} items")
    print(f"   Recommended inventory (95% service): "
          f"{np.percentile(model.sample(sample_week, 1000).sum(axis=0), 95):.0f} items")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
