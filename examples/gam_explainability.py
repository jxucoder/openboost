#!/usr/bin/env python
"""GAM explainability example with OpenBoostGAM.

This example demonstrates:
- Training an interpretable Generalized Additive Model (GAM)
- Visualizing shape functions for each feature
- Understanding feature contributions
- Comparing interpretability vs black-box models

OpenBoostGAM is similar to Microsoft's InterpretML EBM, but GPU-accelerated!
"""

import numpy as np

# OpenBoost imports
import openboost as ob
from openboost import OpenBoostGAM, GradientBoosting

# For data loading
try:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def generate_interpretable_data(n_samples: int = 3000, seed: int = 42):
    """Generate data with known non-linear relationships."""
    np.random.seed(seed)
    
    # Features with interpretable meanings
    age = np.random.uniform(20, 80, n_samples).astype(np.float32)
    income = np.random.lognormal(10.5, 0.5, n_samples).astype(np.float32)
    education_years = np.random.uniform(8, 22, n_samples).astype(np.float32)
    work_hours = np.random.uniform(20, 60, n_samples).astype(np.float32)
    debt_ratio = np.random.uniform(0, 1, n_samples).astype(np.float32)
    
    X = np.column_stack([age, income, education_years, work_hours, debt_ratio])
    feature_names = ['age', 'income', 'education_years', 'work_hours', 'debt_ratio']
    
    # True relationships (known for interpretation validation):
    # - Age: U-shaped (young and old have lower score)
    # - Income: log relationship (diminishing returns)
    # - Education: positive linear
    # - Work hours: optimal around 40
    # - Debt ratio: negative exponential
    
    y = (
        -0.01 * (age - 50) ** 2  # U-shape centered at 50
        + 5 * np.log(income / 30000)  # Log relationship
        + 0.5 * education_years  # Linear
        - 0.02 * (work_hours - 40) ** 2  # Optimal at 40
        - 10 * debt_ratio ** 2  # Quadratic penalty
        + np.random.randn(n_samples).astype(np.float32) * 2  # Noise
    )
    
    return X, y, feature_names


def simple_train_test_split(X, y, test_size=0.2, seed=42):
    """Simple train/test split without sklearn."""
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


def plot_shape_functions_text(gam, feature_names, n_points: int = 20):
    """Print ASCII visualization of shape functions."""
    n_features = len(feature_names)
    
    for feat_idx in range(n_features):
        feat_name = feature_names[feat_idx]
        
        # Get shape values and bin edges
        shape_vals = gam.shape_values_[feat_idx]
        bin_edges = gam.X_binned_.bin_edges_[feat_idx]
        
        # Sample points
        valid_mask = ~np.isnan(shape_vals) & ~np.isinf(shape_vals)
        if valid_mask.sum() == 0:
            print(f"\n{feat_name}: No valid shape values")
            continue
        
        # Get non-zero bins
        nonzero_mask = shape_vals != 0
        if nonzero_mask.sum() < 2:
            print(f"\n{feat_name}: Insufficient shape values")
            continue
        
        # Find range
        min_val = shape_vals[valid_mask].min()
        max_val = shape_vals[valid_mask].max()
        
        print(f"\n{feat_name}:")
        print(f"  Shape function range: [{min_val:.2f}, {max_val:.2f}]")
        
        # ASCII plot
        width = 50
        height = 10
        
        # Sample bins for display
        n_bins = len(shape_vals)
        step = max(1, n_bins // n_points)
        sample_indices = list(range(0, n_bins, step))[:n_points]
        
        if max_val > min_val:
            for row in range(height):
                threshold = max_val - (row + 0.5) * (max_val - min_val) / height
                line = "  |"
                for idx in sample_indices:
                    if shape_vals[idx] >= threshold:
                        line += "#"
                    else:
                        line += " "
                if row == 0:
                    line += f" {max_val:.2f}"
                elif row == height - 1:
                    line += f" {min_val:.2f}"
                print(line)
            print("  +" + "-" * len(sample_indices))


def main():
    print("=" * 60)
    print("OpenBoost GAM Explainability Example")
    print("=" * 60)
    
    # --- Load Data ---
    print("\n1. Loading data...")
    
    if SKLEARN_AVAILABLE:
        housing = fetch_california_housing()
        X, y = housing.data.astype(np.float32), housing.target.astype(np.float32)
        feature_names = list(housing.feature_names)
        print(f"   Dataset: California Housing")
    else:
        X, y, feature_names = generate_interpretable_data()
        print(f"   Dataset: Synthetic interpretable data")
    
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   Features: {feature_names}")
    
    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = simple_train_test_split(X, y)
    
    # --- Train GAM ---
    print("\n2. Training OpenBoostGAM...")
    
    gam = OpenBoostGAM(
        n_rounds=500,
        learning_rate=0.05,
        reg_lambda=1.0,
        n_bins=256,
    )
    gam.fit(X_train, y_train)
    
    print(f"   Trained for {gam.n_rounds} rounds")
    print(f"   Using {gam.n_bins} bins per feature")
    
    # --- Predictions ---
    print("\n3. Making predictions...")
    
    y_pred = gam.predict(X_test)
    
    rmse = ob.rmse_score(y_test, y_pred)
    r2 = ob.r2_score(y_test, y_pred)
    mae = ob.mae_score(y_test, y_pred)
    
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    print(f"   MAE:  {mae:.4f}")
    
    # --- Feature Importance ---
    print("\n4. Feature importance (based on shape function variance)...")
    
    importances = gam.get_feature_importance()
    
    indices = np.argsort(importances)[::-1]
    print("   Features ranked by importance:")
    for i, idx in enumerate(indices):
        bar = "#" * int(importances[idx] / importances.max() * 20)
        print(f"   {i+1}. {feature_names[idx]:20} {importances[idx]:.4f} {bar}")
    
    # --- Shape Functions ---
    print("\n5. Shape functions (feature effects)...")
    
    print("""
   GAM formula: prediction = intercept + f1(x1) + f2(x2) + ... + fn(xn)
   
   Each f_i is a shape function showing how feature i affects the prediction.
   Positive values increase prediction, negative values decrease it.
    """)
    
    # Show top 3 most important features
    top_features = indices[:3]
    
    for feat_idx in top_features:
        feat_name = feature_names[feat_idx]
        shape_vals = gam.shape_values_[feat_idx]
        bin_edges = gam.X_binned_.bin_edges_[feat_idx]
        
        # Filter valid values
        valid_mask = ~np.isnan(shape_vals) & ~np.isinf(shape_vals) & (shape_vals != 0)
        valid_bins = np.where(valid_mask)[0]
        
        if len(valid_bins) < 2:
            print(f"\n   {feat_name}: Insufficient data")
            continue
        
        # Get feature range from data
        feat_min = X[:, feat_idx].min()
        feat_max = X[:, feat_idx].max()
        
        print(f"\n   --- {feat_name} ---")
        print(f"   Feature range: [{feat_min:.2f}, {feat_max:.2f}]")
        print(f"   Shape range: [{shape_vals[valid_mask].min():.3f}, {shape_vals[valid_mask].max():.3f}]")
        
        # Sample some points
        n_show = 8
        step = max(1, len(valid_bins) // n_show)
        sample_bins = valid_bins[::step][:n_show]
        
        print(f"   Sample values:")
        for bin_idx in sample_bins:
            if bin_idx < len(bin_edges) - 1:
                feat_val = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            else:
                feat_val = bin_edges[-1]
            shape_val = shape_vals[bin_idx]
            
            # Visual indicator
            if shape_val > 0.1:
                indicator = "+" * min(int(shape_val * 5), 10)
            elif shape_val < -0.1:
                indicator = "-" * min(int(-shape_val * 5), 10)
            else:
                indicator = "~"
            
            print(f"     x={feat_val:8.2f} -> effect={shape_val:+.3f} {indicator}")
    
    # --- Interpret a Single Prediction ---
    print("\n6. Explaining a single prediction...")
    
    # Pick a sample
    sample_idx = 0
    x_sample = X_test[sample_idx:sample_idx+1]
    y_actual = y_test[sample_idx]
    y_predicted = gam.predict(x_sample)[0]
    
    print(f"   Sample {sample_idx}:")
    print(f"   Actual: {y_actual:.3f}, Predicted: {y_predicted:.3f}")
    print(f"\n   Feature contributions:")
    
    # Calculate per-feature contribution
    total_contribution = 0
    contributions = []
    
    for feat_idx in range(len(feature_names)):
        feat_val = x_sample[0, feat_idx]
        
        # Find the bin for this feature value
        bin_edges = gam.X_binned_.bin_edges_[feat_idx]
        bin_idx = np.searchsorted(bin_edges, feat_val, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, len(gam.shape_values_[feat_idx]) - 1)
        
        contribution = gam.shape_values_[feat_idx][bin_idx]
        contributions.append((feature_names[feat_idx], feat_val, contribution))
        total_contribution += contribution
    
    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for name, val, contrib in contributions:
        sign = "+" if contrib >= 0 else ""
        bar = "#" * min(int(abs(contrib) * 10), 15)
        print(f"   {name:20} = {val:8.2f} -> {sign}{contrib:.3f} {bar}")
    
    print(f"\n   Sum of contributions: {total_contribution:.3f}")
    
    # --- Compare with Black-Box Model ---
    print("\n7. Comparison: GAM vs Black-Box GradientBoosting...")
    
    # Train a standard GBDT
    gbdt = GradientBoosting(
        n_trees=100,
        max_depth=6,
        learning_rate=0.1,
    )
    gbdt.fit(X_train, y_train)
    y_pred_gbdt = gbdt.predict(X_test)
    
    rmse_gam = ob.rmse_score(y_test, y_pred)
    rmse_gbdt = ob.rmse_score(y_test, y_pred_gbdt)
    r2_gam = ob.r2_score(y_test, y_pred)
    r2_gbdt = ob.r2_score(y_test, y_pred_gbdt)
    
    print(f"   Model            | RMSE   | R²")
    print(f"   " + "-" * 40)
    print(f"   GAM              | {rmse_gam:.4f} | {r2_gam:.4f}")
    print(f"   GBDT (black-box) | {rmse_gbdt:.4f} | {r2_gbdt:.4f}")
    
    accuracy_diff = (r2_gbdt - r2_gam) / r2_gbdt * 100
    print(f"\n   GAM interpretability trade-off: {accuracy_diff:+.1f}% R²")
    
    # --- Use Cases ---
    print("\n8. When to use GAM vs GBDT...")
    
    use_cases = """
   USE GAM WHEN:
   - Interpretability is required (regulated industries, healthcare)
   - You need to explain predictions to stakeholders
   - You want to understand feature relationships
   - Feature effects should be additive (no complex interactions)
   - Debugging/validating that model learned reasonable patterns
   
   USE GBDT WHEN:
   - Maximum accuracy is the priority
   - Feature interactions are important
   - Black-box predictions are acceptable
   - You have enough data to capture complex patterns
   
   GAM ADVANTAGES:
   - Each feature has a single, visualizable effect curve
   - Easy to detect unreasonable patterns (data leakage, etc.)
   - Predictions can be manually verified
   - Faster inference (just lookup tables)
   
   GAM LIMITATIONS:
   - No feature interactions (f(x1, x2) is not captured)
   - May underfit for complex relationships
   - Accuracy typically 1-5% lower than full GBDT
"""
    print(use_cases)
    
    # --- Plotting with matplotlib (if available) ---
    print("\n9. Visualization...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        print("   Creating shape function plots...")
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for feat_idx in range(min(8, len(feature_names))):
            ax = axes[feat_idx]
            feat_name = feature_names[feat_idx]
            
            shape_vals = gam.shape_values_[feat_idx]
            bin_edges = gam.X_binned_.bin_edges_[feat_idx]
            
            # Get bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Filter valid
            valid_mask = ~np.isnan(shape_vals[:len(bin_centers)]) & (shape_vals[:len(bin_centers)] != 0)
            
            if valid_mask.sum() > 2:
                ax.plot(bin_centers[valid_mask], shape_vals[:len(bin_centers)][valid_mask], 'b-', linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.fill_between(
                    bin_centers[valid_mask], 
                    0, 
                    shape_vals[:len(bin_centers)][valid_mask],
                    alpha=0.3
                )
            
            ax.set_xlabel(feat_name)
            ax.set_ylabel('Effect on prediction')
            ax.set_title(f'{feat_name}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gam_shape_functions.png', dpi=150)
        print("   Saved: gam_shape_functions.png")
        plt.close()
        
    except ImportError:
        print("   matplotlib not installed. Install with: pip install matplotlib")
        print("   Skipping visualization.")
    
    # --- Built-in Plotting ---
    print("\n10. Built-in plotting method (requires matplotlib)...")
    
    try:
        # OpenBoostGAM has a built-in plot method
        gam.plot_shape_function(0, feature_name=feature_names[0])
        print(f"   Plotted shape function for '{feature_names[0]}'")
        print("   (Opens in matplotlib window or saves to file)")
    except Exception as e:
        print(f"   Built-in plotting not available: {e}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
