"""Generate comprehensive performance report for OpenBoost.

Phase 22 Sprint 4: GPU Performance Validation

This script runs all performance benchmarks and generates a report comparing:
1. NaturalBoost vs NGBoost (distributional GBDT)
2. OpenBoostGAM vs InterpretML EBM (interpretable models)

Run on Modal:
    uv run modal run benchmarks/performance_report.py

Run locally (if you have GPU):
    uv run python benchmarks/performance_report.py --local
"""

import modal
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

app = modal.App("openboost-perf-report")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "numpy>=1.24",
        "numba>=0.60",
        "scikit-learn>=1.0",
        "ngboost>=0.5",
        "interpret>=0.6",
        "xgboost>=2.0",
        "tabulate>=0.9",
        "scipy>=1.10",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "openboost"),
        remote_path="/root/openboost",
    )
)


def generate_data(n_samples: int, n_features: int, noise: float = 10.0, seed: int = 42):
    """Generate synthetic regression data."""
    import numpy as np
    
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # True model: linear + non-linear effects
    y = (
        2.0 * X[:, 0] +
        np.sin(X[:, 1] * 2) +
        0.5 * X[:, 2] ** 2 +
        noise * np.random.randn(n_samples)
    ).astype(np.float32)
    
    return X, y


@app.function(gpu="A100", image=image, timeout=14400)  # 4 hours for large-scale benchmarks
def run_performance_report():
    """Run all performance benchmarks."""
    import sys
    sys.path.insert(0, "/root")

    import numpy as np
    from sklearn.datasets import make_regression, fetch_california_housing
    from sklearn.model_selection import train_test_split
    from numba import cuda

    gpu_name = cuda.get_current_device().name
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    
    results = {
        "gpu_device": gpu_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {}
    }

    # =========================================================================
    # Benchmark 1: NaturalBoost vs NGBoost
    # =========================================================================
    print("=" * 70)
    print("BENCHMARK 1: NaturalBoost vs NGBoost")
    print("=" * 70)

    import openboost as ob
    from ngboost import NGBRegressor
    from ngboost.distns import Normal

    ob.set_backend("cuda")
    print(f"OpenBoost backend: {ob.get_backend()}")

    ngboost_results = []

    for n_samples in [250_000, 500_000, 1_000_000]:
        print(f"\n{n_samples:,} samples:")

        X, y = generate_data(n_samples, n_features=20, noise=10.0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # NGBoost
        ngb = NGBRegressor(
            Dist=Normal,
            n_estimators=100,
            learning_rate=0.1,
            verbose=False
        )
        start = time.perf_counter()
        ngb.fit(X_train, y_train)
        ngb_time = time.perf_counter() - start

        # NGBoost predictions and metrics
        ngb_pred = ngb.predict(X_test)
        ngb_dist = ngb.pred_dist(X_test)
        ngb_nll = float(-ngb_dist.logpdf(y_test).mean())
        ngb_lower = ngb_dist.ppf(0.05)
        ngb_upper = ngb_dist.ppf(0.95)
        ngb_coverage = float(np.mean((y_test >= ngb_lower) & (y_test <= ngb_upper)))

        # NaturalBoost - warmup first
        nb_warmup = ob.NaturalBoostNormal(n_trees=2, max_depth=3)
        nb_warmup.fit(X_train[:500], y_train[:500])
        cuda.synchronize()

        # NaturalBoost
        nb = ob.NaturalBoostNormal(n_trees=100, learning_rate=0.1, max_depth=3)
        start = time.perf_counter()
        nb.fit(X_train, y_train)
        cuda.synchronize()
        nb_time = time.perf_counter() - start

        # NaturalBoost predictions and metrics
        nb_pred = nb.predict(X_test)
        nb_nll = float(nb.nll(X_test, y_test))
        nb_lower, nb_upper = nb.predict_interval(X_test, alpha=0.1)
        nb_coverage = float(np.mean((y_test >= nb_lower) & (y_test <= nb_upper)))

        speedup = ngb_time / nb_time

        print(f"  NGBoost:      {ngb_time:.2f}s (NLL: {ngb_nll:.4f}, Coverage: {ngb_coverage:.1%})")
        print(f"  NaturalBoost: {nb_time:.2f}s (NLL: {nb_nll:.4f}, Coverage: {nb_coverage:.1%})")
        print(f"  Speedup:      {speedup:.2f}x")

        ngboost_results.append({
            "samples": n_samples,
            "ngboost_time": ngb_time,
            "ngboost_nll": ngb_nll,
            "ngboost_coverage": ngb_coverage,
            "naturalboost_time": nb_time,
            "naturalboost_nll": nb_nll,
            "naturalboost_coverage": nb_coverage,
            "speedup": speedup,
        })

    results["benchmarks"]["ngboost"] = ngboost_results

    # California Housing benchmark
    print(f"\nCalifornia Housing Dataset:")
    data = fetch_california_housing()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Samples: {len(X_train):,} train, {len(X_test):,} test")

    # NGBoost on California Housing
    ngb = NGBRegressor(Dist=Normal, n_estimators=100, learning_rate=0.1, verbose=False)
    start = time.perf_counter()
    ngb.fit(X_train, y_train)
    ngb_time_cal = time.perf_counter() - start

    # NaturalBoost on California Housing
    nb = ob.NaturalBoostNormal(n_trees=100, learning_rate=0.1, max_depth=3)
    start = time.perf_counter()
    nb.fit(X_train, y_train)
    cuda.synchronize()
    nb_time_cal = time.perf_counter() - start

    speedup_cal = ngb_time_cal / nb_time_cal
    print(f"  NGBoost:      {ngb_time_cal:.2f}s")
    print(f"  NaturalBoost: {nb_time_cal:.2f}s")
    print(f"  Speedup:      {speedup_cal:.2f}x")

    results["benchmarks"]["ngboost_california"] = {
        "samples": len(X_train),
        "ngboost_time": ngb_time_cal,
        "naturalboost_time": nb_time_cal,
        "speedup": speedup_cal,
    }

    # =========================================================================
    # Benchmark 2: OpenBoostGAM vs InterpretML EBM
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK 2: OpenBoostGAM vs InterpretML EBM")
    print("=" * 70)

    from interpret.glassbox import ExplainableBoostingRegressor
    from openboost import OpenBoostGAM
    from sklearn.metrics import r2_score

    ebm_results = []

    for n_samples in [500_000, 1_000_000, 2_000_000]:
        print(f"\n{n_samples:,} samples:")

        X, y = generate_data(n_samples, n_features=20, noise=0.1)

        # InterpretML EBM
        ebm = ExplainableBoostingRegressor(
            max_rounds=200,
            learning_rate=0.05,
            outer_bags=1,
            inner_bags=0,
            interactions=0,
            n_jobs=-1,
        )
        start = time.perf_counter()
        ebm.fit(X, y)
        ebm_time = time.perf_counter() - start
        ebm_r2 = r2_score(y, ebm.predict(X))

        # OpenBoostGAM - warmup
        gam_warmup = OpenBoostGAM(n_rounds=10)
        gam_warmup.fit(X[:1000], y[:1000])
        cuda.synchronize()

        # OpenBoostGAM
        gam = OpenBoostGAM(n_rounds=200, learning_rate=0.05)
        start = time.perf_counter()
        gam.fit(X, y)
        cuda.synchronize()
        gam_time = time.perf_counter() - start
        gam_r2 = r2_score(y, gam.predict(X))

        speedup = ebm_time / gam_time

        print(f"  EBM:          {ebm_time:.2f}s (R²: {ebm_r2:.4f})")
        print(f"  OpenBoostGAM: {gam_time:.2f}s (R²: {gam_r2:.4f})")
        print(f"  Speedup:      {speedup:.1f}x")

        ebm_results.append({
            "samples": n_samples,
            "ebm_time": ebm_time,
            "ebm_r2": ebm_r2,
            "openboostgam_time": gam_time,
            "openboostgam_r2": gam_r2,
            "speedup": speedup,
        })

    results["benchmarks"]["ebm"] = ebm_results

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    print("\nNaturalBoost vs NGBoost (100 trees, Normal distribution):")
    print(f"{'Samples':<14} {'NGBoost (s)':<14} {'NaturalBoost (s)':<18} {'Speedup':<10}")
    print("-" * 56)
    for r in ngboost_results:
        print(f"{r['samples']:<14,} {r['ngboost_time']:<14.2f} {r['naturalboost_time']:<18.2f} {r['speedup']:<10.2f}x")

    print("\nOpenBoostGAM vs InterpretML EBM (200 rounds, 20 features):")
    print(f"{'Samples':<14} {'EBM (s)':<12} {'OpenBoostGAM (s)':<18} {'Speedup':<10}")
    print("-" * 54)
    for r in ebm_results:
        print(f"{r['samples']:<14,} {r['ebm_time']:<12.2f} {r['openboostgam_time']:<18.2f} {r['speedup']:<10.1f}x")

    # Acceptance criteria check
    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA")
    print("=" * 70)

    # Check NaturalBoost >1.3x faster
    nb_faster = all(r['speedup'] > 1.0 for r in ngboost_results)
    nb_speedup_1m = next(r['speedup'] for r in ngboost_results if r['samples'] == 1_000_000)
    print(f"[{'✓' if nb_faster else '✗'}] NaturalBoost faster than NGBoost at all sizes")
    print(f"[{'✓' if nb_speedup_1m > 1.3 else '✗'}] NaturalBoost >1.3x faster at 1M samples (actual: {nb_speedup_1m:.2f}x)")

    # Check OpenBoostGAM >10x faster at 2M
    gam_speedup_2m = next(r['speedup'] for r in ebm_results if r['samples'] == 2_000_000)
    print(f"[{'✓' if gam_speedup_2m > 10 else '✗'}] OpenBoostGAM >10x faster at 2M samples (actual: {gam_speedup_2m:.1f}x)")

    # Check comparable accuracy
    gam_r2_comparable = all(abs(r['openboostgam_r2'] - r['ebm_r2']) < 0.1 for r in ebm_results)
    print(f"[{'✓' if gam_r2_comparable else '✗'}] OpenBoostGAM comparable R² to EBM")

    results["acceptance_criteria"] = {
        "naturalboost_faster_all": nb_faster,
        "naturalboost_speedup_1m": nb_speedup_1m,
        "openboostgam_speedup_2m": gam_speedup_2m,
        "r2_comparable": gam_r2_comparable,
    }

    return results


class BytesEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles bytes objects."""
    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        return super().default(obj)


def convert_bytes_in_dict(obj):
    """Recursively convert bytes to strings in a dictionary."""
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    elif isinstance(obj, dict):
        return {k: convert_bytes_in_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bytes_in_dict(item) for item in obj]
    return obj


@app.local_entrypoint()
def main():
    """Run performance report."""
    print("Running OpenBoost Performance Report on Modal A100...")
    print("This may take 10-20 minutes.\n")

    results = run_performance_report.remote()
    
    # Convert any bytes to strings (e.g., GPU device name)
    results = convert_bytes_in_dict(results)

    # Save results
    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"performance_report_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, cls=BytesEncoder)

    print(f"\nResults saved to: {results_file}")

    # Print summary for README
    print("\n" + "=" * 70)
    print("README MARKDOWN (copy-paste ready):")
    print("=" * 70)

    print("""
### NaturalBoost vs NGBoost

| Samples | NGBoost | NaturalBoost (GPU) | Speedup |
|---------|---------|-------------------|---------|""")
    for r in results["benchmarks"]["ngboost"]:
        print(f"| {r['samples']:,}   | {r['ngboost_time']:.1f}s    | {r['naturalboost_time']:.1f}s             | {r['speedup']:.1f}x    |")

    print("""
*Benchmark: Normal distribution, 100 trees, 20 features, A100 GPU*

### OpenBoostGAM vs InterpretML EBM

| Samples | EBM (CPU) | OpenBoostGAM (GPU) | Speedup |
|---------|-----------|-------------------|---------|""")
    for r in results["benchmarks"]["ebm"]:
        print(f"| {r['samples']:,}  | {r['ebm_time']:.0f}s       | {r['openboostgam_time']:.1f}s              | {r['speedup']:.0f}x     |")

    print("""
*Benchmark: 200 rounds, 20 features, pure GAM (no interactions), A100 GPU*
""")


# For local execution without Modal
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        print("Running locally (requires GPU)...")

        import numpy as np
        from sklearn.model_selection import train_test_split

        # Add openboost to path
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        import openboost as ob
        print(f"OpenBoost backend: {ob.get_backend()}")

        # Quick benchmark
        print("\nQuick NaturalBoost vs NGBoost benchmark (5K samples):")
        X, y = generate_data(5000, 20, noise=10.0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        try:
            from ngboost import NGBRegressor
            from ngboost.distns import Normal

            ngb = NGBRegressor(Dist=Normal, n_estimators=50, learning_rate=0.1, verbose=False)
            start = time.perf_counter()
            ngb.fit(X_train, y_train)
            ngb_time = time.perf_counter() - start
            print(f"  NGBoost:      {ngb_time:.2f}s")
        except ImportError:
            print("  NGBoost not installed. Run: pip install ngboost")
            ngb_time = None

        try:
            nb = ob.NaturalBoostNormal(n_trees=50, learning_rate=0.1, max_depth=3)
            start = time.perf_counter()
            nb.fit(X_train, y_train)
            nb_time = time.perf_counter() - start
            print(f"  NaturalBoost: {nb_time:.2f}s")

            if ngb_time:
                print(f"  Speedup:      {ngb_time / nb_time:.2f}x")
        except Exception as e:
            print(f"  NaturalBoost failed: {e}")

    else:
        print("Usage:")
        print("  Modal:  uv run modal run benchmarks/performance_report.py")
        print("  Local:  uv run python benchmarks/performance_report.py --local")
