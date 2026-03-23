"""Targeted timing of training loop phases on Modal A100.

Measures: gradient, tree build, predict, conversion, Python overhead.
"""

from __future__ import annotations

from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent.parent

app = modal.App("openboost-timing-exp")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("numpy>=1.24", "numba>=0.60", "joblib>=1.3")
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "openboost"),
        remote_path="/root/openboost_pkg/openboost",
    )
)


@app.function(gpu="A100", image=image, timeout=1800)
def timing_experiment():
    import sys
    sys.path.insert(0, "/root/openboost_pkg")

    import time
    import numpy as np
    import openboost as ob
    from numba import cuda

    ob.set_backend("cuda")
    gpu_name = cuda.get_current_device().name
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    print(f"GPU: {gpu_name}")

    # Generate data
    n_samples, n_features = 1_000_000, 100
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = (2.0 * np.sin(X[:, 0] * 1.5) + 1.5 * X[:, 1] * X[:, 2]
         + rng.randn(n_samples).astype(np.float32) * 0.5).astype(np.float32)

    # Warmup
    for _ in range(2):
        ob.GradientBoosting(n_trees=3, max_depth=8, loss="mse").fit(X[:2000], y[:2000])
    cuda.synchronize()

    # Now manually run the training loop with precise timing
    from openboost._array import array as ob_array, BinnedArray
    from openboost._array import as_numba_array
    from openboost._backends._cuda import build_tree_gpu_native, mse_grad_inplace_gpu
    from openboost._core._tree import Tree, fit_tree_gpu_native
    from openboost._core._predict import predict_tree_add_gpu

    n_trees = 200
    max_depth = 8
    learning_rate = 0.1

    # Bin data
    X_binned = ob_array(X)
    binned_gpu = as_numba_array(X_binned.data)
    y_gpu = cuda.to_device(y)
    pred_gpu = cuda.device_array(n_samples, dtype=np.float32)
    cuda.to_device(np.zeros(n_samples, dtype=np.float32), to=pred_gpu)
    grad_gpu = cuda.device_array(n_samples, dtype=np.float32)
    hess_gpu = cuda.device_array(n_samples, dtype=np.float32)
    cuda.to_device(np.full(n_samples, 2.0, dtype=np.float32), to=hess_gpu)

    t_grad = 0.0
    t_tree = 0.0
    t_pred = 0.0
    t_python = 0.0

    cuda.synchronize()
    t_total_start = time.perf_counter()

    for i in range(n_trees):
        t0 = time.perf_counter()
        mse_grad_inplace_gpu(pred_gpu, y_gpu, grad_gpu)
        t1 = time.perf_counter()
        t_grad += t1 - t0

        legacy_tree = fit_tree_gpu_native(
            X_binned, grad_gpu, hess_gpu,
            max_depth=max_depth, reg_lambda=1.0,
            min_child_weight=1.0, min_gain=0.0,
        )
        t2 = time.perf_counter()
        t_tree += t2 - t1

        predict_tree_add_gpu(legacy_tree, X_binned, pred_gpu, learning_rate)
        t3 = time.perf_counter()
        t_pred += t3 - t2

    cuda.synchronize()
    t_total = time.perf_counter() - t_total_start

    # Synced timing: cuda.synchronize() after each phase
    cuda.to_device(np.zeros(n_samples, dtype=np.float32), to=pred_gpu)
    t_grad_sync = 0.0
    t_tree_sync = 0.0
    t_pred_sync = 0.0

    cuda.synchronize()
    t_total_sync_start = time.perf_counter()

    for i in range(n_trees):
        t0 = time.perf_counter()
        mse_grad_inplace_gpu(pred_gpu, y_gpu, grad_gpu)
        cuda.synchronize()
        t1 = time.perf_counter()
        t_grad_sync += t1 - t0

        legacy_tree = fit_tree_gpu_native(
            X_binned, grad_gpu, hess_gpu,
            max_depth=max_depth, reg_lambda=1.0,
            min_child_weight=1.0, min_gain=0.0,
        )
        cuda.synchronize()
        t2 = time.perf_counter()
        t_tree_sync += t2 - t1

        predict_tree_add_gpu(legacy_tree, X_binned, pred_gpu, learning_rate)
        cuda.synchronize()
        t3 = time.perf_counter()
        t_pred_sync += t3 - t2

    t_total_sync = time.perf_counter() - t_total_sync_start

    print("\n=== ASYNC TIMING (host-side, no sync between phases) ===")
    print(f"Total wall time:      {t_total:.3f}s")
    print(f"  gradient launches:  {t_grad:.3f}s ({100*t_grad/t_total:.1f}%)")
    print(f"  tree build calls:   {t_tree:.3f}s ({100*t_tree/t_total:.1f}%)")
    print(f"  predict launches:   {t_pred:.3f}s ({100*t_pred/t_total:.1f}%)")
    print(f"  GPU wait (final):   {t_total - t_grad - t_tree - t_pred:.3f}s")

    print("\n=== SYNCED TIMING (cuda.synchronize between each phase) ===")
    print(f"Total wall time:      {t_total_sync:.3f}s")
    print(f"  gradient (GPU):     {t_grad_sync:.3f}s ({100*t_grad_sync/t_total_sync:.1f}%)")
    print(f"  tree build (GPU):   {t_tree_sync:.3f}s ({100*t_tree_sync/t_total_sync:.1f}%)")
    print(f"  predict (GPU):      {t_pred_sync:.3f}s ({100*t_pred_sync/t_total_sync:.1f}%)")

    print(f"\nSync overhead: {t_total_sync - t_total:.3f}s "
          f"({100*(t_total_sync - t_total)/t_total:.1f}%)")

    # Per-tree breakdown
    print(f"\n=== PER-TREE (synced) ===")
    print(f"  gradient:  {1000*t_grad_sync/n_trees:.2f}ms")
    print(f"  tree:      {1000*t_tree_sync/n_trees:.2f}ms")
    print(f"  predict:   {1000*t_pred_sync/n_trees:.2f}ms")

    return {
        "total_async": t_total,
        "total_sync": t_total_sync,
        "grad_sync": t_grad_sync,
        "tree_sync": t_tree_sync,
        "pred_sync": t_pred_sync,
    }


@app.local_entrypoint()
def main():
    result = timing_experiment.remote()
    print("\nDone.")


if __name__ == "__main__":
    print("Usage: uv run modal run development/autoresearch/timing_experiment_modal.py")
