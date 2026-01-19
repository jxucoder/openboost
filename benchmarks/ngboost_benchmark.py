"""Phase 16: Benchmark OpenBoost NGBoost vs Official NGBoost.

Compare:
1. Training speed
2. Prediction speed  
3. NLL (negative log-likelihood) - prediction quality
4. Calibration of prediction intervals
"""

import time
import numpy as np
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split


def benchmark_synthetic(n_samples: int = 10000, n_features: int = 20, n_trees: int = 100):
    """Benchmark on synthetic data."""
    print(f"\n{'='*70}")
    print(f"SYNTHETIC DATA: {n_samples:,} samples, {n_features} features, {n_trees} trees")
    print('='*70)
    
    # Generate data
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=10, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # --- Official NGBoost ---
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        
        print("\n[Official NGBoost]")
        model_official = NGBRegressor(
            Dist=Normal,
            n_estimators=n_trees,
            learning_rate=0.1,
            verbose=False,
        )
        
        t0 = time.perf_counter()
        model_official.fit(X_train, y_train)
        train_time_official = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        pred_official = model_official.predict(X_test)
        pred_time_official = time.perf_counter() - t0
        
        # Get distribution params for NLL
        dist_official = model_official.pred_dist(X_test)
        nll_official = -dist_official.logpdf(y_test).mean()
        
        # Prediction intervals
        lower_official = dist_official.ppf(0.05)
        upper_official = dist_official.ppf(0.95)
        coverage_official = np.mean((y_test >= lower_official) & (y_test <= upper_official))
        
        rmse_official = np.sqrt(np.mean((pred_official - y_test)**2))
        
        results['official'] = {
            'train_time': train_time_official,
            'pred_time': pred_time_official,
            'nll': nll_official,
            'rmse': rmse_official,
            'coverage_90': coverage_official,
        }
        
        print(f"  Train time:  {train_time_official:.2f}s")
        print(f"  Pred time:   {pred_time_official*1000:.1f}ms")
        print(f"  NLL:         {nll_official:.4f}")
        print(f"  RMSE:        {rmse_official:.4f}")
        print(f"  90% coverage: {coverage_official:.1%}")
        
    except Exception as e:
        print(f"  Error: {e}")
        results['official'] = None
    
    # --- OpenBoost NGBoost ---
    try:
        import openboost as ob
        
        print("\n[OpenBoost NGBoost]")
        model_openboost = ob.NGBoostNormal(
            n_trees=n_trees,
            learning_rate=0.1,
            max_depth=3,
        )
        
        t0 = time.perf_counter()
        model_openboost.fit(X_train, y_train)
        train_time_openboost = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        pred_openboost = model_openboost.predict(X_test)
        pred_time_openboost = time.perf_counter() - t0
        
        # NLL
        nll_openboost = model_openboost.nll(X_test, y_test)
        if hasattr(nll_openboost, 'mean'):
            nll_openboost = nll_openboost.mean()
        
        # Prediction intervals
        lower_openboost, upper_openboost = model_openboost.predict_interval(X_test, alpha=0.1)
        coverage_openboost = np.mean((y_test >= lower_openboost) & (y_test <= upper_openboost))
        
        rmse_openboost = np.sqrt(np.mean((pred_openboost - y_test)**2))
        
        results['openboost'] = {
            'train_time': train_time_openboost,
            'pred_time': pred_time_openboost,
            'nll': nll_openboost,
            'rmse': rmse_openboost,
            'coverage_90': coverage_openboost,
        }
        
        print(f"  Train time:  {train_time_openboost:.2f}s")
        print(f"  Pred time:   {pred_time_openboost*1000:.1f}ms")
        print(f"  NLL:         {nll_openboost:.4f}")
        print(f"  RMSE:        {rmse_openboost:.4f}")
        print(f"  90% coverage: {coverage_openboost:.1%}")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results['openboost'] = None
    
    # --- Comparison ---
    if results.get('official') and results.get('openboost'):
        print("\n[Comparison]")
        speedup = results['official']['train_time'] / results['openboost']['train_time']
        print(f"  Training speedup: {speedup:.2f}x {'(OpenBoost faster)' if speedup > 1 else '(NGBoost faster)'}")
        
        pred_speedup = results['official']['pred_time'] / results['openboost']['pred_time']
        print(f"  Prediction speedup: {pred_speedup:.2f}x")
        
        nll_diff = results['openboost']['nll'] - results['official']['nll']
        print(f"  NLL difference: {nll_diff:+.4f} {'(OpenBoost better)' if nll_diff < 0 else '(NGBoost better)'}")
    
    return results


def benchmark_california_housing(n_trees: int = 100):
    """Benchmark on California Housing dataset."""
    print(f"\n{'='*70}")
    print(f"CALIFORNIA HOUSING DATASET: {n_trees} trees")
    print('='*70)
    
    # Load data
    data = fetch_california_housing()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    
    results = {}
    
    # --- Official NGBoost ---
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        
        print("\n[Official NGBoost]")
        model_official = NGBRegressor(
            Dist=Normal,
            n_estimators=n_trees,
            learning_rate=0.1,
            verbose=False,
        )
        
        t0 = time.perf_counter()
        model_official.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        pred = model_official.predict(X_test)
        dist = model_official.pred_dist(X_test)
        nll = -dist.logpdf(y_test).mean()
        rmse = np.sqrt(np.mean((pred - y_test)**2))
        
        results['official'] = {'train_time': train_time, 'nll': nll, 'rmse': rmse}
        print(f"  Train time: {train_time:.2f}s | NLL: {nll:.4f} | RMSE: {rmse:.4f}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # --- OpenBoost NGBoost ---
    try:
        import openboost as ob
        
        print("\n[OpenBoost NGBoost]")
        model = ob.NGBoostNormal(n_trees=n_trees, learning_rate=0.1, max_depth=3)
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        pred = model.predict(X_test)
        nll = model.nll(X_test, y_test)
        if hasattr(nll, 'mean'):
            nll = nll.mean()
        rmse = np.sqrt(np.mean((pred - y_test)**2))
        
        results['openboost'] = {'train_time': train_time, 'nll': nll, 'rmse': rmse}
        print(f"  Train time: {train_time:.2f}s | NLL: {nll:.4f} | RMSE: {rmse:.4f}")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # --- Comparison ---
    if results.get('official') and results.get('openboost'):
        print("\n[Comparison]")
        speedup = results['official']['train_time'] / results['openboost']['train_time']
        print(f"  Training speedup: {speedup:.2f}x {'(OpenBoost faster)' if speedup > 1 else '(NGBoost faster)'}")
    
    return results


def benchmark_scaling():
    """Benchmark training time scaling with data size."""
    print(f"\n{'='*70}")
    print("SCALING BENCHMARK")
    print('='*70)
    
    sizes = [1000, 5000, 10000, 20000]
    n_trees = 50
    
    print(f"\n{'Size':<10} {'NGBoost':<12} {'OpenBoost':<12} {'Speedup':<10}")
    print("-" * 44)
    
    for n in sizes:
        X, y = make_regression(n_samples=n, n_features=10, noise=10, random_state=42)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # Official NGBoost
        try:
            from ngboost import NGBRegressor
            from ngboost.distns import Normal
            
            model = NGBRegressor(Dist=Normal, n_estimators=n_trees, learning_rate=0.1, verbose=False)
            t0 = time.perf_counter()
            model.fit(X, y)
            time_official = time.perf_counter() - t0
        except:
            time_official = float('nan')
        
        # OpenBoost
        try:
            import openboost as ob
            
            model = ob.NGBoostNormal(n_trees=n_trees, learning_rate=0.1, max_depth=3)
            t0 = time.perf_counter()
            model.fit(X, y)
            time_openboost = time.perf_counter() - t0
        except:
            time_openboost = float('nan')
        
        speedup = time_official / time_openboost if time_openboost > 0 else 0
        print(f"{n:<10} {time_official:<12.2f}s {time_openboost:<12.2f}s {speedup:<10.2f}x")


if __name__ == '__main__':
    print("="*70)
    print("PHASE 16: OpenBoost NGBoost vs Official NGBoost Benchmark")
    print("="*70)
    
    # Quick benchmark
    benchmark_synthetic(n_samples=5000, n_features=10, n_trees=50)
    
    # Real dataset
    benchmark_california_housing(n_trees=50)
    
    # Scaling
    benchmark_scaling()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
