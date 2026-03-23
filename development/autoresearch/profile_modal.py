"""Profile OpenBoost training on Modal A100 to find actual bottlenecks."""

from __future__ import annotations

import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent.parent

app = modal.App("openboost-profile")

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
def profile_remote():
    import sys
    sys.path.insert(0, "/root/openboost_pkg")

    import time
    import numpy as np
    import openboost as ob
    from openboost._profiler import ProfilingCallback, print_profile_summary
    from numba import cuda

    ob.set_backend("cuda")
    gpu_name = cuda.get_current_device().name
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    print(f"GPU: {gpu_name}")

    # Generate 1M x 100 dataset
    n_samples, n_features = 1_000_000, 100
    rng = np.random.RandomState(42)
    X = np.empty((n_samples, n_features), dtype=np.float32)

    n_gauss = n_features // 5
    n_uniform = n_features // 5
    n_lognorm = n_features // 10
    n_binary = n_features // 10
    n_ordinal = n_features // 10
    n_corr = n_features // 10
    n_noise = n_features - n_gauss - n_uniform - n_lognorm - n_binary - n_ordinal - n_corr

    idx = 0
    X[:, idx:idx + n_gauss] = rng.randn(n_samples, n_gauss)
    gauss_end = idx + n_gauss
    idx = gauss_end
    X[:, idx:idx + n_uniform] = rng.uniform(0, 1, (n_samples, n_uniform))
    idx += n_uniform
    X[:, idx:idx + n_lognorm] = rng.lognormal(0, 1, (n_samples, n_lognorm))
    idx += n_lognorm
    X[:, idx:idx + n_binary] = rng.binomial(1, 0.3, (n_samples, n_binary))
    binary_start = idx
    idx += n_binary
    X[:, idx:idx + n_ordinal] = rng.randint(1, 6, (n_samples, n_ordinal))
    idx += n_ordinal
    for j in range(n_corr):
        src = rng.randint(0, gauss_end)
        X[:, idx + j] = (0.8 * X[:, src] + 0.2 * rng.randn(n_samples)).astype(np.float32)
    idx += n_corr
    X[:, idx:idx + n_noise] = rng.randn(n_samples, n_noise)

    y = (
        2.0 * np.sin(X[:, 0] * 1.5)
        + 1.5 * X[:, 1] * X[:, 2]
        + np.where(X[:, 3] > 0, X[:, 4], -X[:, 4])
        + 0.8 * X[:, gauss_end] ** 2
        + 1.0 * X[:, binary_start]
        + 0.3 * X[:, 0] * X[:, binary_start]
        + rng.randn(n_samples).astype(np.float32) * 0.5
    ).astype(np.float32)

    # Warmup
    print("Warming up...")
    for _ in range(2):
        ob.GradientBoosting(n_trees=3, max_depth=8, loss="mse").fit(X[:2000], y[:2000])
    cuda.synchronize()

    # Profile
    print(f"Profiling: {n_samples:,} x {n_features}, 200 trees, depth=8")
    profiler = ProfilingCallback(output_dir="/tmp/logs/")
    model = ob.GradientBoosting(n_trees=200, max_depth=8, learning_rate=0.1, loss="mse")

    cuda.synchronize()
    t0 = time.perf_counter()
    model.fit(X, y, callbacks=[profiler])
    cuda.synchronize()
    wall = time.perf_counter() - t0

    print(f"\nWall time: {wall:.2f}s\n")

    report = profiler.report
    report["_path"] = str(profiler.report_path)
    print_profile_summary(report)

    # Detailed phase table
    print(f"\n{'Phase':<20} {'Time (s)':>10} {'%':>8} {'Calls':>8} {'Mean (ms)':>10}")
    print("-" * 60)
    for phase, data in report["phases"].items():
        calls = str(data["calls"]) if data["calls"] is not None else "-"
        mean_ms = f"{data['mean_s'] * 1000:.2f}" if data["mean_s"] is not None else "-"
        print(f"{phase:<20} {data['total_s']:>10.3f} {data['pct']:>7.1f}% {calls:>8} {mean_ms:>10}")

    return report


@app.local_entrypoint()
def main():
    report = profile_remote.remote()
    print("\nDone.")


if __name__ == "__main__":
    print("Usage: uv run modal run development/autoresearch/profile_modal.py")
