"""Run CUDA verification tests on Modal GPU.

Phase 21.2: Modal-based GPU testing infrastructure.

Usage:
    uv run modal run tests/modal_gpu_tests.py
    uv run modal run tests/modal_gpu_tests.py::run_all_gpu_tests
    uv run modal run tests/modal_gpu_tests.py::verify_cuda_available
"""

import modal

app = modal.App("openboost-cuda-verification")

# Build image with CUDA and OpenBoost dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "numpy>=1.24",
        "numba>=0.60",
        "pytest>=8.0",
        "scipy>=1.10",
        "joblib>=1.2",
        "scikit-learn>=1.3",
    )
    .add_local_dir("src/openboost", remote_path="/root/src/openboost", copy=True)
    .add_local_file("tests/test_cuda_verification.py", remote_path="/root/tests/test_cuda_verification.py", copy=True)
    .env({"PYTHONPATH": "/root/src"})
)


@app.function(gpu="T4", image=image, timeout=300)
def verify_cuda_available() -> dict:
    """Quick check that CUDA is working on Modal GPU.
    
    Returns:
        Dict with CUDA status and quick training test results.
    """
    import numpy as np
    import sys
    
    results = {
        "cuda_available": False,
        "device_name": None,
        "numba_version": None,
        "openboost_backend": None,
        "training_successful": False,
        "prediction_shape": None,
        "error": None,
    }
    
    # Check numba CUDA
    try:
        from numba import cuda
        import numba
        results["numba_version"] = numba.__version__
        results["cuda_available"] = cuda.is_available()
        if results["cuda_available"]:
            device = cuda.get_current_device()
            results["device_name"] = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
    except Exception as e:
        results["error"] = f"Numba CUDA check failed: {e}"
        return results
    
    # Check OpenBoost backend
    try:
        sys.path.insert(0, "/root/src")
        import openboost as ob
        
        results["openboost_backend"] = ob.get_backend()
    except Exception as e:
        results["error"] = f"OpenBoost import failed: {e}"
        return results
    
    # Quick training test
    try:
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        model = ob.GradientBoosting(n_trees=5, max_depth=3)
        model.fit(X, y)
        predictions = model.predict(X)
        
        results["training_successful"] = len(model.trees_) == 5
        results["prediction_shape"] = predictions.shape
    except Exception as e:
        results["error"] = f"Training test failed: {e}"
    
    return results


@app.function(gpu="T4", image=image, timeout=1800)
def run_all_gpu_tests() -> dict:
    """Run all CUDA verification tests on Modal GPU.
    
    Returns:
        Dict with test results, stdout, and stderr.
    """
    import subprocess
    import sys
    
    # Run pytest
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "/root/tests/test_cuda_verification.py",
            "-v", "--tb=short",
            "-x",  # Stop on first failure for faster feedback
        ],
        capture_output=True,
        text=True,
        timeout=1700,
    )
    
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }


@app.function(gpu="T4", image=image, timeout=600)
def run_specific_test(test_name: str) -> dict:
    """Run a specific test on Modal GPU.
    
    Args:
        test_name: Name of test to run (e.g., "TestGradientBoostingGPU::test_gradient_boosting_fit_predict")
    
    Returns:
        Dict with test results.
    """
    import subprocess
    import sys
    
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            f"/root/tests/test_cuda_verification.py::{test_name}",
            "-v", "--tb=long",
        ],
        capture_output=True,
        text=True,
        timeout=550,
    )
    
    return {
        "test_name": test_name,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }


@app.function(gpu="T4", image=image, timeout=600)
def benchmark_gpu_vs_cpu() -> dict:
    """Benchmark GPU vs CPU training speed.
    
    Returns:
        Dict with timing results for different dataset sizes.
    """
    import numpy as np
    import time
    import sys
    
    sys.path.insert(0, "/root/src")
    import openboost as ob
    
    results = []
    
    for n_samples in [1000, 5000, 10000, 25000]:
        np.random.seed(42)
        X = np.random.randn(n_samples, 20).astype(np.float32)
        y = np.random.randn(n_samples).astype(np.float32)
        
        # CPU timing
        ob.set_backend("cpu")
        model_cpu = ob.GradientBoosting(n_trees=50, max_depth=6)
        start = time.time()
        model_cpu.fit(X, y)
        cpu_time = time.time() - start
        
        # GPU timing
        ob.set_backend("cuda")
        model_gpu = ob.GradientBoosting(n_trees=50, max_depth=6)
        
        # Warm up CUDA
        _ = ob.GradientBoosting(n_trees=1, max_depth=2).fit(X[:100], y[:100])
        
        start = time.time()
        model_gpu.fit(X, y)
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        results.append({
            "n_samples": n_samples,
            "cpu_time_s": round(cpu_time, 3),
            "gpu_time_s": round(gpu_time, 3),
            "speedup": round(speedup, 2),
        })
    
    return {
        "benchmark_results": results,
        "gpu_device": ob.get_backend(),
    }


@app.function(gpu="T4", image=image, timeout=300)
def test_all_models_gpu() -> dict:
    """Test all model types on GPU.
    
    Returns:
        Dict with test results for each model type.
    """
    import numpy as np
    import sys
    
    sys.path.insert(0, "/root/src")
    import openboost as ob
    
    np.random.seed(42)
    X = np.random.randn(300, 5).astype(np.float32)
    y = X[:, 0] + np.random.randn(300).astype(np.float32) * 0.5
    y_positive = np.abs(y) + 0.1
    y_binary = (y > 0).astype(np.float32)
    y_count = np.random.poisson(5, 300).astype(np.float32)
    
    ob.set_backend("cuda")
    
    results = {}
    
    # Test each model type
    models_to_test = [
        ("GradientBoosting", ob.GradientBoosting(n_trees=10, max_depth=3), y),
        ("NaturalBoostNormal", ob.NaturalBoostNormal(n_trees=10, max_depth=3), y),
        ("NaturalBoostLogNormal", ob.NaturalBoostLogNormal(n_trees=10, max_depth=3), y_positive),
        ("NaturalBoostGamma", ob.NaturalBoostGamma(n_trees=10, max_depth=3), y_positive),
        ("NaturalBoostPoisson", ob.NaturalBoostPoisson(n_trees=10, max_depth=3), y_count),
        ("OpenBoostGAM", ob.OpenBoostGAM(n_rounds=10, learning_rate=0.1), y),
        ("DART", ob.DART(n_trees=10, max_depth=3, dropout_rate=0.1), y),
        ("LinearLeafGBDT", ob.LinearLeafGBDT(n_trees=10, max_depth=3), y),
        ("OpenBoostRegressor", ob.OpenBoostRegressor(n_estimators=10, max_depth=3), y),
        ("OpenBoostClassifier", ob.OpenBoostClassifier(n_estimators=10, max_depth=3), y_binary.astype(np.int32)),
    ]
    
    for name, model, target in models_to_test:
        try:
            model.fit(X, target)
            predictions = model.predict(X)
            results[name] = {
                "status": "passed",
                "prediction_shape": predictions.shape,
            }
        except Exception as e:
            results[name] = {
                "status": "failed",
                "error": str(e),
            }
    
    return results


@app.local_entrypoint()
def main():
    """Run all CUDA verification tests."""
    print("=" * 70)
    print("OpenBoost Phase 21: CUDA GPU End-to-End Verification")
    print("=" * 70)
    
    # Step 1: Quick CUDA check
    print("\n1. Verifying CUDA availability on Modal GPU...")
    print("-" * 70)
    
    info = verify_cuda_available.remote()
    
    print(f"   Numba version:      {info['numba_version']}")
    print(f"   CUDA Available:     {info['cuda_available']}")
    print(f"   GPU Device:         {info['device_name']}")
    print(f"   OpenBoost Backend:  {info['openboost_backend']}")
    print(f"   Quick Train OK:     {info['training_successful']}")
    
    if info['error']:
        print(f"   ERROR: {info['error']}")
    
    if not info['cuda_available']:
        print("\n❌ CUDA not available. Cannot run GPU tests.")
        return
    
    if info['openboost_backend'] != 'cuda':
        print("\n⚠️  OpenBoost not using CUDA backend. Investigating...")
    
    # Step 2: Test all model types
    print("\n2. Testing all model types on GPU...")
    print("-" * 70)
    
    model_results = test_all_models_gpu.remote()
    
    all_passed = True
    for model_name, result in model_results.items():
        status = "✅" if result['status'] == 'passed' else "❌"
        print(f"   {status} {model_name}: {result['status']}")
        if result['status'] == 'failed':
            print(f"      Error: {result.get('error', 'Unknown')}")
            all_passed = False
    
    # Step 3: Run full test suite
    print("\n3. Running full pytest test suite...")
    print("-" * 70)
    
    test_result = run_all_gpu_tests.remote()
    
    # Print condensed output
    lines = test_result['stdout'].split('\n')
    for line in lines:
        if any(x in line for x in ['PASSED', 'FAILED', 'ERROR', 'test_', '===', '---']):
            print(f"   {line}")
    
    if test_result['stderr']:
        print("\n   STDERR:")
        for line in test_result['stderr'].split('\n')[:10]:
            print(f"   {line}")
    
    # Step 4: Performance benchmark
    print("\n4. GPU vs CPU Performance Benchmark...")
    print("-" * 70)
    
    benchmark = benchmark_gpu_vs_cpu.remote()
    
    print(f"   {'Samples':<10} {'CPU (s)':<10} {'GPU (s)':<10} {'Speedup':<10}")
    print(f"   {'-'*40}")
    for r in benchmark['benchmark_results']:
        print(f"   {r['n_samples']:<10} {r['cpu_time_s']:<10} {r['gpu_time_s']:<10} {r['speedup']}x")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    checks = [
        ("CUDA Detection", info['cuda_available']),
        ("OpenBoost Backend", info['openboost_backend'] == 'cuda'),
        ("Quick Training", info['training_successful']),
        ("All Models", all_passed),
        ("Test Suite", test_result['passed']),
    ]
    
    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name:<20} {status}")
    
    overall_passed = all(passed for _, passed in checks)
    
    print("\n" + "=" * 70)
    if overall_passed:
        print("✅ ALL CUDA VERIFICATION TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED - Review output above")
    print("=" * 70)
