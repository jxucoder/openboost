#!/usr/bin/env python
"""sklearn Pipeline integration example with OpenBoost.

This example demonstrates:
- Using OpenBoost in sklearn Pipelines
- Combining with preprocessing (scaling, encoding)
- GridSearchCV for hyperparameter tuning
- Cross-validation workflows

OpenBoost's sklearn wrappers are fully compatible with the sklearn ecosystem.
"""

import numpy as np

# OpenBoost imports
import openboost as ob
from openboost import (
    OpenBoostRegressor,
    OpenBoostClassifier,
    get_param_grid,
)

# sklearn imports
try:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import (
        GridSearchCV,
        cross_val_score,
        cross_validate,
    )
    from sklearn.datasets import fetch_california_housing, load_breast_cancer
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("This example requires sklearn. Install with: pip install scikit-learn")
    exit(1)


def main():
    print("=" * 60)
    print("OpenBoost sklearn Pipeline Integration Example")
    print("=" * 60)
    
    # --- Basic Pipeline ---
    print("\n1. Basic pipeline with preprocessing...")
    
    # Load data
    housing = fetch_california_housing()
    X, y = housing.data.astype(np.float32), housing.target.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', OpenBoostRegressor(n_estimators=100, max_depth=4)),
    ])
    
    # Fit
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    r2 = ob.r2_score(y_test, y_pred)
    
    print(f"   Pipeline R²: {r2:.4f}")
    
    # --- GridSearchCV ---
    print("\n2. GridSearchCV for hyperparameter tuning...")
    
    # Define search space
    # Note: use 'model__' prefix for pipeline parameters
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [3, 5],
        'model__learning_rate': [0.05, 0.1],
    }
    
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=1,  # OpenBoost handles parallelism internally
        verbose=1,
    )
    
    print("   Running grid search...")
    search.fit(X_train, y_train)
    
    print(f"\n   Best parameters: {search.best_params_}")
    print(f"   Best CV score: {search.best_score_:.4f}")
    print(f"   Test score: {search.score(X_test, y_test):.4f}")
    
    # --- Built-in Param Grid ---
    print("\n3. Using OpenBoost's suggested param grid...")
    
    # OpenBoost provides suggested parameter grids
    suggested_grid = get_param_grid('regression')
    print(f"   Suggested grid: {suggested_grid}")
    
    # Adapt for pipeline
    pipeline_grid = {f'model__{k}': v for k, v in suggested_grid.items()}
    
    # --- Cross-Validation Utilities ---
    print("\n4. Cross-validation with multiple metrics...")
    
    simple_model = OpenBoostRegressor(n_estimators=50, max_depth=4)
    
    # Single metric
    scores = cross_val_score(simple_model, X, y, cv=5, scoring='r2')
    print(f"   CV R² scores: {scores}")
    print(f"   Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Multiple metrics
    cv_results = cross_validate(
        simple_model, X, y,
        cv=5,
        scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
        return_train_score=True,
    )
    
    print("\n   Multi-metric CV results:")
    for key in ['test_r2', 'test_neg_mean_squared_error']:
        scores = cv_results[key]
        print(f"   {key}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # --- Classification Pipeline ---
    print("\n5. Classification pipeline...")
    
    cancer = load_breast_cancer()
    X_clf, y_clf = cancer.data.astype(np.float32), cancer.target.astype(np.float32)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    clf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', OpenBoostClassifier(n_estimators=50, max_depth=3)),
    ])
    
    clf_pipeline.fit(X_train_clf, y_train_clf)
    
    y_pred_clf = clf_pipeline.predict(X_test_clf)
    y_proba_clf = clf_pipeline.predict_proba(X_test_clf)[:, 1]
    
    accuracy = ob.accuracy_score(y_test_clf, y_pred_clf)
    auc = ob.roc_auc_score(y_test_clf, y_proba_clf)
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC AUC: {auc:.4f}")
    
    # --- Mixed Feature Types ---
    print("\n6. Handling mixed feature types...")
    
    # Create synthetic mixed data
    np.random.seed(42)
    n = 500
    
    # Numeric features
    num_feat_1 = np.random.randn(n).astype(np.float32)
    num_feat_2 = np.random.exponential(2, n).astype(np.float32)
    
    # Categorical features (as integers for simplicity)
    cat_feat_1 = np.random.choice(['A', 'B', 'C'], n)
    cat_feat_2 = np.random.choice(['X', 'Y'], n)
    
    # Target
    y_mixed = (
        2 * num_feat_1 
        + np.log(num_feat_2 + 1)
        + (cat_feat_1 == 'A') * 1.5
        + (cat_feat_2 == 'X') * 0.5
        + np.random.randn(n).astype(np.float32) * 0.5
    ).astype(np.float32)
    
    # Create DataFrame-like structure
    X_mixed = np.column_stack([
        num_feat_1.reshape(-1, 1),
        num_feat_2.reshape(-1, 1),
        cat_feat_1.reshape(-1, 1),
        cat_feat_2.reshape(-1, 1),
    ])
    
    # ColumnTransformer for mixed preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [0, 1]),  # Numeric columns
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [2, 3]),  # Categorical
        ],
        remainder='passthrough'
    )
    
    mixed_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', OpenBoostRegressor(n_estimators=50, max_depth=4)),
    ])
    
    # Split and fit
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_mixed, y_mixed, test_size=0.2, random_state=42
    )
    
    mixed_pipeline.fit(X_train_m, y_train_m)
    
    y_pred_m = mixed_pipeline.predict(X_test_m)
    r2_mixed = ob.r2_score(y_test_m, y_pred_m)
    
    print(f"   Mixed features pipeline R²: {r2_mixed:.4f}")
    
    # --- Early Stopping in Pipeline ---
    print("\n7. Early stopping with validation set...")
    
    # Split for early stopping
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    es_model = OpenBoostRegressor(
        n_estimators=500,  # High number
        max_depth=4,
        early_stopping_rounds=20,
    )
    
    # Note: early stopping requires eval_set in fit
    es_model.fit(X_t, y_t, eval_set=[(X_v, y_v)])
    
    print(f"   Requested trees: 500")
    print(f"   Actual trees (early stopped): {len(es_model.booster_.trees_)}")
    print(f"   Best iteration: {es_model.best_iteration_}")
    
    # --- Feature Importances in Pipeline ---
    print("\n8. Accessing feature importances from pipeline...")
    
    # Get the model from pipeline
    model_in_pipeline = pipeline.named_steps['model']
    
    importances = model_in_pipeline.feature_importances_
    feature_names = housing.feature_names
    
    indices = np.argsort(importances)[::-1]
    print("   Top features:")
    for i, idx in enumerate(indices[:5]):
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # --- Model Persistence with Pipeline ---
    print("\n9. Saving/loading pipeline...")
    
    import joblib
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'pipeline.joblib')
        
        # Save
        joblib.dump(pipeline, path)
        print(f"   Saved pipeline to {path}")
        
        # Load
        loaded_pipeline = joblib.load(path)
        
        # Verify
        y_pred_loaded = loaded_pipeline.predict(X_test)
        np.testing.assert_array_almost_equal(y_pred, y_pred_loaded)
        print("   Loaded pipeline produces identical predictions ✓")
    
    # --- Tips ---
    print("\n" + "=" * 60)
    print("Tips for sklearn Integration")
    print("=" * 60)
    
    tips = """
   1. USE FLOAT32:
      X = X.astype(np.float32)  # OpenBoost is optimized for float32
   
   2. PIPELINE PARAMETER NAMING:
      In GridSearchCV, use 'model__param' for model parameters:
      param_grid = {'model__n_estimators': [50, 100]}
   
   3. EARLY STOPPING:
      Pass eval_set to fit() for early stopping:
      pipeline.fit(X_train, y_train, model__eval_set=[(X_val, y_val)])
   
   4. GPU ACCELERATION:
      Works automatically - OpenBoost detects GPU regardless of pipeline
   
   5. FEATURE NAMES:
      After preprocessing, feature names may change. Track them through
      ColumnTransformer.get_feature_names_out()
   
   6. PARALLELISM:
      Use n_jobs=1 in GridSearchCV - OpenBoost parallelizes internally
   
   7. MEMORY:
      Large GridSearchCV can be memory-intensive. Consider:
      - RandomizedSearchCV for faster search
      - Smaller CV folds
      - Sequential parameter search
"""
    print(tips)
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
