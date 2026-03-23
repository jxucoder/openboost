"""Time breakdown of GPU training loop on Modal A100.

Measures: gradient computation, tree building, prediction, and conversion overhead.
"""

from __future__ import annotations

from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent.parent

app = modal.App("openboost-time-breakdown")

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
def time_breakdown_remote():
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

    # Now manually run the training loop with timing
    from openboost._array import array as ob_array
    from openboost._loss import _mse_gradient_gpu
    from openboost._core._tree import fit_tree_gpu_native
    from openboost._core._predict import predict_tree_add_gpu
    from openboost._array import as_numba_array

    n_trees = 200
    max_depth = 8
    learning_rate = 0.1

    # Bin data
    X_binned = ob_array(X)
    binned_gpu = as_numba_array(X_binned.data)
    y_gpu = cuda.to_device(y)
    pred_gpu = cuda.device_array(n_samples, dtype=np.float32)
    # Zero predictions
    cuda.to_device(np.zeros(n_samples, dtype=np.float32), to=pred_gpu)

    t_grad = 0.0
    t_tree = 0.0
    t_pred = 0.0
    t_conv = 0.0
    t_other = 0.0

    cuda.synchronize()
    t_total_start = time.perf_counter()

    for i in range(n_trees):
        # Gradient
        cuda.synchronize()
        t0 = time.perf_counter()
        grad_gpu, hess_gpu = _mse_gradient_gpu(pred_gpu, y_gpu)
        cuda.synchronize()
        t_grad += time.perf_counter() - t0

        # Tree building
        t0 = time.perf_counter()
        legacy_tree = fit_tree_gpu_native(
            X_binned, grad_gpu, hess_gpu,
            max_depth=max_depth, reg_lambda=1.0,
            min_child_weight=1.0, min_gain=0.0,
        )
        cuda.synchronize()
        t_tree += time.perf_counter() - t0

        # Prediction update
        t0 = time.perf_counter()
        predict_tree_add_gpu(legacy_tree, X_binned, pred_gpu, learning_rate)
        cuda.synchronize()
        t_pred += time.perf_counter() - t0

        # Tree conversion (GPU -> CPU arrays + TreeStructure)
        t0 = time.perf_counter()
        features, thresholds, values, left, right = legacy_tree.to_arrays()
        t_conv += time.perf_counter() - t0

    t_total = time.perf_counter() - t_total_start

    # === Kernel-level profiling: one tree build with per-depth timing ===
    print("\n=== KERNEL-LEVEL BREAKDOWN (single tree) ===")
    from openboost._backends._cuda import (
        build_tree_gpu_native, _get_tree_workspace,
        _build_histogram_shared_kernel, _find_level_splits_kernel,
        _partition_samples_kernel, _zero_level_histograms_kernel,
        _init_tree_nodes_kernel, _init_sample_nodes_kernel,
        _compute_leaf_sums_kernel, _compute_leaf_values_kernel,
        _create_children_kernel, _zero_float_array_kernel,
    )
    import math

    # Fresh gradient for profiling
    grad_gpu2, hess_gpu2 = _mse_gradient_gpu(pred_gpu, y_gpu)
    binned = as_numba_array(X_binned.data)
    grad = as_numba_array(grad_gpu2)
    hess = as_numba_array(hess_gpu2)

    n_feat, n_samp = binned.shape
    max_nodes = 2**(max_depth + 1) - 1
    threads = 256
    blocks = math.ceil(max_nodes / threads)
    sample_blocks = math.ceil(n_samp / threads)

    ws = _get_tree_workspace(n_samp, n_feat, max_depth)
    sample_node_ids = ws['sample_node_ids']
    histograms = ws['histograms']
    node_gains = ws['node_gains']
    node_sum_grad = ws['node_sum_grad']
    node_sum_hess = ws['node_sum_hess']

    node_features = cuda.device_array(max_nodes, dtype=np.int32)
    node_thresholds = cuda.device_array(max_nodes, dtype=np.int32)
    node_values = cuda.device_array(max_nodes, dtype=np.float32)
    node_left = cuda.device_array(max_nodes, dtype=np.int32)
    node_right = cuda.device_array(max_nodes, dtype=np.int32)

    cuda.synchronize()
    _init_tree_nodes_kernel[blocks, threads](node_features, node_thresholds, node_left, node_right, max_nodes)
    _init_sample_nodes_kernel[sample_blocks, threads](sample_node_ids, n_samp)
    cuda.synchronize()

    CHUNK_SIZE = 4096
    n_chunks = math.ceil(n_samp / CHUNK_SIZE)

    depth_times = {}
    for depth in range(max_depth):
        level_start = 2**depth - 1
        level_end = 2**(depth + 1) - 1
        n_nodes_at_level = level_end - level_start

        # Zero histograms
        cuda.synchronize()
        t0 = time.perf_counter()
        zero_grid = (n_nodes_at_level, n_feat)
        _zero_level_histograms_kernel[zero_grid, 256](histograms, level_start, level_end, n_feat)
        cuda.synchronize()
        t_zero = time.perf_counter() - t0

        # Build histograms
        t0 = time.perf_counter()
        hist_grid = (n_feat, n_chunks)
        n_passes = (n_nodes_at_level + 15) // 16
        for pass_idx in range(n_passes):
            node_offset = pass_idx * 16
            nodes_this_pass = min(16, n_nodes_at_level - node_offset)
            _build_histogram_shared_kernel[hist_grid, 256](
                binned, grad, hess, sample_node_ids,
                level_start, nodes_this_pass, node_offset, histograms
            )
        cuda.synchronize()
        t_hist = time.perf_counter() - t0

        # Find splits
        t0 = time.perf_counter()
        _find_level_splits_kernel[n_nodes_at_level, 256](
            histograms, level_start, level_end,
            np.float32(1.0), np.float32(1.0), np.float32(0.0),
            node_features, node_thresholds, node_gains,
            node_sum_grad, node_sum_hess
        )
        cuda.synchronize()
        t_split = time.perf_counter() - t0

        # Create children
        t0 = time.perf_counter()
        children_blocks = math.ceil(n_nodes_at_level / threads)
        _create_children_kernel[children_blocks, threads](
            level_start, level_end, node_features, node_left, node_right
        )
        cuda.synchronize()
        t_children = time.perf_counter() - t0

        # Partition
        t0 = time.perf_counter()
        _partition_samples_kernel[sample_blocks, threads](
            binned, sample_node_ids, node_features, node_thresholds,
            node_left, node_right, level_start, level_end
        )
        cuda.synchronize()
        t_part = time.perf_counter() - t0

        depth_times[depth] = {
            'zero': t_zero, 'histogram': t_hist, 'split': t_split,
            'children': t_children, 'partition': t_part,
            'n_nodes': n_nodes_at_level, 'n_passes': n_passes,
        }

    # Final leaf computation
    cuda.synchronize()
    t0 = time.perf_counter()
    _zero_float_array_kernel[blocks, threads](node_sum_grad, max_nodes)
    _zero_float_array_kernel[blocks, threads](node_sum_hess, max_nodes)
    _compute_leaf_sums_kernel[sample_blocks, threads](
        grad, hess, sample_node_ids, node_sum_grad, node_sum_hess
    )
    _compute_leaf_values_kernel[blocks, threads](
        node_features, node_sum_grad, node_sum_hess,
        np.float32(1.0), node_values, max_nodes
    )
    cuda.synchronize()
    t_leaf = time.perf_counter() - t0

    print(f"\n{'Depth':<6} {'Nodes':>6} {'Passes':>7} {'Zero':>8} {'Hist':>8} {'Split':>8} {'Part':>8} {'Total':>8}")
    print("-" * 70)
    tree_total = 0
    for d in range(max_depth):
        dt = depth_times[d]
        total = dt['zero'] + dt['histogram'] + dt['split'] + dt['children'] + dt['partition']
        tree_total += total
        print(f"{d:<6} {dt['n_nodes']:>6} {dt['n_passes']:>7} "
              f"{dt['zero']*1000:>7.2f} {dt['histogram']*1000:>7.2f} "
              f"{dt['split']*1000:>7.2f} {dt['partition']*1000:>7.2f} "
              f"{total*1000:>7.2f}")
    tree_total += t_leaf
    print(f"{'leaf':<6} {'':>6} {'':>7} {'':>8} {'':>8} {'':>8} {'':>8} {t_leaf*1000:>7.2f}")
    print(f"{'TOTAL':<6} {'':>6} {'':>7} {'':>8} {'':>8} {'':>8} {'':>8} {tree_total*1000:>7.2f}")

    print(f"\n{'Phase':<25} {'Time (s)':>10} {'%':>8} {'Per-tree (ms)':>14}")
    print("-" * 60)
    phases = [
        ("gradient", t_grad),
        ("tree_build", t_tree),
        ("prediction_update", t_pred),
        ("tree_conversion", t_conv),
    ]
    accounted = sum(t for _, t in phases)
    phases.append(("other", t_total - accounted))

    for name, t in phases:
        pct = 100 * t / t_total
        per_tree = 1000 * t / n_trees
        print(f"{name:<25} {t:>10.3f} {pct:>7.1f}% {per_tree:>14.2f}")

    print(f"\n{'TOTAL':<25} {t_total:>10.3f}")
    print(f"\nNote: cuda.synchronize() between phases adds ~0.01ms overhead per call")

    return {
        "total": t_total,
        "gradient": t_grad,
        "tree_build": t_tree,
        "prediction_update": t_pred,
        "tree_conversion": t_conv,
        "gpu": gpu_name,
    }


@app.local_entrypoint()
def main():
    report = time_breakdown_remote.remote()
    print("\nDone.")


if __name__ == "__main__":
    print("Usage: uv run modal run development/autoresearch/time_breakdown_modal.py")
