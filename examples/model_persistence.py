#!/usr/bin/env python
"""Model persistence (save/load) example with OpenBoost.

This example demonstrates:
- Saving and loading trained models
- Different serialization formats
- Cross-backend compatibility (train on GPU, load on CPU)
- Saving model checkpoints during training

All OpenBoost models support save/load out of the box!
"""

import numpy as np
import tempfile
import os

# OpenBoost imports
import openboost as ob
from openboost import (
    GradientBoosting,
    NaturalBoostNormal,
    OpenBoostGAM,
    DART,
    ModelCheckpoint,
    EarlyStopping,
)


def generate_data(n_samples: int = 1000, seed: int = 42):
    """Generate sample data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, 5).astype(np.float32)
    y = 2 * X[:, 0] + X[:, 1] ** 2 + np.random.randn(n_samples).astype(np.float32) * 0.5
    return X, y


def main():
    print("=" * 60)
    print("OpenBoost Model Persistence Example")
    print("=" * 60)
    
    # Generate data
    X, y = generate_data()
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    # --- Basic Save/Load ---
    print("\n1. Basic save/load with joblib...")
    
    model = GradientBoosting(n_trees=50, max_depth=4)
    model.fit(X_train, y_train)
    
    # Original predictions
    y_pred_original = model.predict(X_test)
    rmse_original = ob.rmse_score(y_test, y_pred_original)
    print(f"   Original model RMSE: {rmse_original:.4f}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        path = os.path.join(tmpdir, 'model.joblib')
        model.save(path)
        print(f"   Saved to: {path}")
        
        # Check file size
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"   File size: {size_mb:.2f} MB")
        
        # Load
        loaded = GradientBoosting.load(path)
        
        # Verify
        y_pred_loaded = loaded.predict(X_test)
        rmse_loaded = ob.rmse_score(y_test, y_pred_loaded)
        print(f"   Loaded model RMSE: {rmse_loaded:.4f}")
        
        # Check identical
        if np.allclose(y_pred_original, y_pred_loaded):
            print("   Predictions are identical ✓")
        else:
            print("   Warning: Predictions differ!")
    
    # --- Different File Formats ---
    print("\n2. Saving with different methods...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Method 1: Model's save method (recommended)
        path1 = os.path.join(tmpdir, 'model_save.joblib')
        model.save(path1)
        print(f"   model.save(): {os.path.getsize(path1) / 1024:.1f} KB")
        
        # Method 2: Direct joblib
        import joblib
        path2 = os.path.join(tmpdir, 'model_joblib.joblib')
        joblib.dump(model, path2)
        print(f"   joblib.dump(): {os.path.getsize(path2) / 1024:.1f} KB")
        
        # Method 3: Pickle
        import pickle
        path3 = os.path.join(tmpdir, 'model_pickle.pkl')
        with open(path3, 'wb') as f:
            pickle.dump(model, f)
        print(f"   pickle.dump(): {os.path.getsize(path3) / 1024:.1f} KB")
        
        # All should load correctly
        loaded1 = GradientBoosting.load(path1)
        loaded2 = joblib.load(path2)
        with open(path3, 'rb') as f:
            loaded3 = pickle.load(f)
        
        # Verify all produce same predictions
        pred1 = loaded1.predict(X_test)
        pred2 = loaded2.predict(X_test)
        pred3 = loaded3.predict(X_test)
        
        assert np.allclose(pred1, pred2) and np.allclose(pred2, pred3)
        print("   All formats produce identical predictions ✓")
    
    # --- Distributional Models ---
    print("\n3. Saving NaturalBoost (distributional) models...")
    
    prob_model = NaturalBoostNormal(n_trees=30, max_depth=3)
    prob_model.fit(X_train, y_train)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'prob_model.joblib')
        prob_model.save(path)
        
        loaded_prob = NaturalBoostNormal.load(path)
        
        # Check distribution parameters
        params_original = prob_model.predict_params(X_test)
        params_loaded = loaded_prob.predict_params(X_test)
        
        assert np.allclose(params_original['loc'], params_loaded['loc'])
        assert np.allclose(params_original['scale'], params_loaded['scale'])
        print("   Distribution parameters preserved ✓")
        
        # Check intervals
        lower_orig, upper_orig = prob_model.predict_interval(X_test, alpha=0.1)
        lower_load, upper_load = loaded_prob.predict_interval(X_test, alpha=0.1)
        
        assert np.allclose(lower_orig, lower_load)
        assert np.allclose(upper_orig, upper_load)
        print("   Prediction intervals preserved ✓")
    
    # --- GAM Models ---
    print("\n4. Saving OpenBoostGAM models...")
    
    gam = OpenBoostGAM(n_rounds=100, learning_rate=0.05)
    gam.fit(X_train, y_train)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'gam.joblib')
        gam.save(path)
        
        loaded_gam = OpenBoostGAM.load(path)
        
        # Check shape functions preserved
        for feat_idx in range(X.shape[1]):
            orig_shape = gam.shape_values_[feat_idx]
            load_shape = loaded_gam.shape_values_[feat_idx]
            assert np.allclose(orig_shape, load_shape)
        
        print("   Shape functions preserved ✓")
        
        # Check predictions
        pred_orig = gam.predict(X_test)
        pred_load = loaded_gam.predict(X_test)
        assert np.allclose(pred_orig, pred_load)
        print("   Predictions identical ✓")
    
    # --- DART Models ---
    print("\n5. Saving DART models...")
    
    dart = DART(n_trees=50, max_depth=4, dropout_rate=0.1)
    dart.fit(X_train, y_train)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'dart.joblib')
        dart.save(path)
        
        loaded_dart = DART.load(path)
        
        # Check tree weights preserved
        assert np.allclose(dart.tree_weights_, loaded_dart.tree_weights_)
        print("   Tree weights preserved ✓")
        
        pred_orig = dart.predict(X_test)
        pred_load = loaded_dart.predict(X_test)
        assert np.allclose(pred_orig, pred_load)
        print("   Predictions identical ✓")
    
    # --- Model Checkpoints During Training ---
    print("\n6. Saving checkpoints during training...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'checkpoint_{iteration}.joblib')
        
        checkpoint_model = GradientBoosting(
            n_trees=100,
            max_depth=4,
            learning_rate=0.1,
        )
        
        callbacks = [
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_loss',
            ),
            EarlyStopping(patience=10),
        ]
        
        # Split for validation
        X_t, X_v = X_train[:600], X_train[600:]
        y_t, y_v = y_train[:600], y_train[600:]
        
        checkpoint_model.fit(
            X_t, y_t,
            eval_set=(X_v, y_v),
            callbacks=callbacks,
        )
        
        # List saved checkpoints
        checkpoints = [f for f in os.listdir(tmpdir) if f.startswith('checkpoint')]
        print(f"   Saved {len(checkpoints)} checkpoint(s)")
        
        if checkpoints:
            best_checkpoint = sorted(checkpoints)[-1]
            print(f"   Best checkpoint: {best_checkpoint}")
    
    # --- Cross-Backend Compatibility ---
    print("\n7. Cross-backend compatibility...")
    
    print(f"   Current backend: {ob.get_backend()}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train model
        model = GradientBoosting(n_trees=30, max_depth=3)
        model.fit(X_train, y_train)
        
        path = os.path.join(tmpdir, 'cross_backend.joblib')
        model.save(path)
        
        # Save current backend
        original_backend = ob.get_backend()
        
        # Switch to CPU
        ob.set_backend('cpu')
        print(f"   Switched to: {ob.get_backend()}")
        
        # Load and predict
        loaded = GradientBoosting.load(path)
        pred_cpu = loaded.predict(X_test)
        
        # Switch back
        ob.set_backend(original_backend)
        print(f"   Switched back to: {ob.get_backend()}")
        
        pred_original = model.predict(X_test)
        
        # Verify same predictions
        assert np.allclose(pred_cpu, pred_original)
        print("   Cross-backend predictions identical ✓")
    
    # --- Model Metadata ---
    print("\n8. Model metadata...")
    
    model = GradientBoosting(
        n_trees=50,
        max_depth=4,
        learning_rate=0.1,
    )
    model.fit(X_train, y_train)
    
    print(f"   Model type: {type(model).__name__}")
    print(f"   Number of trees: {len(model.trees_)}")
    print(f"   Max depth: {model.max_depth}")
    print(f"   Learning rate: {model.learning_rate}")
    
    # Estimate model size
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'model.joblib')
        model.save(path)
        size_kb = os.path.getsize(path) / 1024
        print(f"   Serialized size: {size_kb:.1f} KB")
    
    # --- Best Practices ---
    print("\n" + "=" * 60)
    print("Best Practices for Model Persistence")
    print("=" * 60)
    
    practices = """
   FILE FORMAT:
   - Use .joblib extension (default, recommended)
   - Also works with .pkl (pickle)
   - Choose based on your infrastructure
   
   SAVE METHOD:
   - model.save(path): Recommended, handles edge cases
   - joblib.dump(model, path): Direct, widely compatible
   - pickle.dump(model, f): Standard Python
   
   LOADING:
   - ModelClass.load(path): Type-safe, recommended
   - joblib.load(path): Generic, returns object
   - pickle.load(f): Standard Python
   
   CHECKPOINTING:
   - Use ModelCheckpoint callback for training checkpoints
   - Set save_best_only=True to keep only best model
   - Useful for long training runs
   
   CROSS-BACKEND:
   - Models are saved in backend-agnostic format
   - Train on GPU, load on CPU works!
   - Predictions will be identical
   
   VERSION COMPATIBILITY:
   - Models saved with OpenBoost X.Y should load with X.Y+
   - Major version changes may break compatibility
   - Consider saving model config separately for reproducibility
   
   LARGE MODELS:
   - joblib handles large arrays efficiently
   - Use compression: joblib.dump(model, path, compress=3)
   - For very large models, consider saving trees separately
"""
    print(practices)
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
