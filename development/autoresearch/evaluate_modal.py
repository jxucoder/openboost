"""Autoresearch evaluation harness — runs on Modal A100.

Uploads current source to a Modal GPU, runs the benchmark,
and reports the score locally.

Usage:
    uv run modal run development/autoresearch/evaluate_modal.py
    uv run modal run development/autoresearch/evaluate_modal.py --quick
    uv run modal run development/autoresearch/evaluate_modal.py --trials 5
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCORE_FILE = PROJECT_ROOT / "development" / "autoresearch" / "scores.jsonl"

app = modal.App("openboost-autoresearch")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install(
        "numpy>=1.24",
        "numba>=0.60",
        "joblib>=1.3",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "openboost"),
        remote_path="/root/openboost_pkg/openboost",
    )
)

# ===========================================================================
# Benchmark config — must match evaluate.py
# ===========================================================================

BENCHMARK_CONFIG = {
    "n_samples": 1_000_000,
    "n_features": 100,
    "n_trees": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "loss": "mse",
    "seed": 42,
}

WARMUP_CONFIG = {
    "n_samples": 2_000,
    "n_trees": 3,
}

# Realistic target has noise std=0.5, so irreducible MSE ~0.25. Gate at 1.0
# catches broken models while passing reasonable fits (R2 > 0.8).
MAX_MSE = 1.0


# ===========================================================================
# Remote function — runs on A100
# ===========================================================================

@app.function(gpu="A100", image=image, timeout=1800)
def run_benchmark_remote(trials: int = 3) -> dict:
    """Run the benchmark on a Modal A100 GPU."""
    import sys
    sys.path.insert(0, "/root/openboost_pkg")

    import time

    import numpy as np

    import openboost as ob
    ob.set_backend("cuda")

    from numba import cuda
    gpu_name = cuda.get_current_device().name
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    print(f"GPU: {gpu_name}")
    print(f"Backend: {ob.get_backend()}")

    cfg = BENCHMARK_CONFIG

    # Generate data
    print(f"Generating {cfg['n_samples']:,} x {cfg['n_features']} dataset...")
    rng = np.random.RandomState(cfg["seed"])
    X = np.empty((cfg["n_samples"], cfg["n_features"]), dtype=np.float32)

    n_f = cfg["n_features"]
    n_gauss = n_f // 5
    n_uniform = n_f // 5
    n_lognorm = n_f // 10
    n_binary = n_f // 10
    n_ordinal = n_f // 10
    n_corr = n_f // 10
    n_noise = n_f - n_gauss - n_uniform - n_lognorm - n_binary - n_ordinal - n_corr

    idx = 0
    X[:, idx:idx + n_gauss] = rng.randn(cfg["n_samples"], n_gauss)
    gauss_end = idx + n_gauss
    idx = gauss_end
    X[:, idx:idx + n_uniform] = rng.uniform(0, 1, (cfg["n_samples"], n_uniform))
    idx += n_uniform
    X[:, idx:idx + n_lognorm] = rng.lognormal(0, 1, (cfg["n_samples"], n_lognorm))
    idx += n_lognorm
    X[:, idx:idx + n_binary] = rng.binomial(1, 0.3, (cfg["n_samples"], n_binary))
    binary_start = idx
    idx += n_binary
    X[:, idx:idx + n_ordinal] = rng.randint(1, 6, (cfg["n_samples"], n_ordinal))
    idx += n_ordinal
    for j in range(n_corr):
        src = rng.randint(0, gauss_end)
        X[:, idx + j] = (0.8 * X[:, src] + 0.2 * rng.randn(cfg["n_samples"])).astype(np.float32)
    idx += n_corr
    X[:, idx:idx + n_noise] = rng.randn(cfg["n_samples"], n_noise)

    y = (
        2.0 * np.sin(X[:, 0] * 1.5)
        + 1.5 * X[:, 1] * X[:, 2]
        + np.where(X[:, 3] > 0, X[:, 4], -X[:, 4])
        + 0.8 * X[:, gauss_end] ** 2
        + 1.0 * X[:, binary_start]
        + 0.3 * X[:, 0] * X[:, binary_start]
        + rng.randn(cfg["n_samples"]).astype(np.float32) * 0.5
    ).astype(np.float32)

    # Warmup JIT + CUDA
    print("Warming up JIT + CUDA...")
    for _ in range(2):
        ob.GradientBoosting(
            n_trees=WARMUP_CONFIG["n_trees"],
            max_depth=cfg["max_depth"],
            loss=cfg["loss"],
        ).fit(X[:WARMUP_CONFIG["n_samples"]], y[:WARMUP_CONFIG["n_samples"]])
    cuda.synchronize()

    # Timed trials
    print(f"Running {trials} trial(s)...")
    fit_times = []
    model = None
    for t in range(trials):
        model = ob.GradientBoosting(
            n_trees=cfg["n_trees"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            loss=cfg["loss"],
        )
        cuda.synchronize()
        t0 = time.perf_counter()
        model.fit(X, y)
        cuda.synchronize()
        elapsed = time.perf_counter() - t0
        fit_times.append(elapsed)
        print(f"  trial {t + 1}/{trials}: {elapsed:.3f}s")

    # Accuracy
    pred = model.predict(X)
    mse = float(np.mean((pred - y) ** 2))
    r2 = float(1 - np.sum((pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2))

    return {
        "fit_times": fit_times,
        "median_time": float(sorted(fit_times)[len(fit_times) // 2]),
        "min_time": float(min(fit_times)),
        "max_time": float(max(fit_times)),
        "mse": mse,
        "r2": r2,
        "gpu": gpu_name,
        "config": cfg,
    }


# ===========================================================================
# Local entrypoint
# ===========================================================================

@app.local_entrypoint()
def main(trials: int = 3, quick: bool = False):
    """Run autoresearch evaluation on Modal A100."""
    import subprocess

    if quick:
        trials = 1

    git_sha = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except Exception:
        pass

    # Load previous score
    previous_score = None
    if SCORE_FILE.exists():
        try:
            lines = SCORE_FILE.read_text().strip().split("\n")
            if lines and lines[-1]:
                previous_score = json.loads(lines[-1]).get("score")
        except Exception:
            pass

    cfg = BENCHMARK_CONFIG
    print("=" * 60)
    print("AUTORESEARCH EVALUATION (Modal A100)")
    print("=" * 60)
    print(f"Git SHA: {git_sha}")
    print(f"Config: {cfg['n_samples']:,} samples, "
          f"{cfg['n_features']} features, "
          f"{cfg['n_trees']} trees, "
          f"depth={cfg['max_depth']}")
    print(f"Trials: {trials}")
    print()

    # Run on Modal
    print("Launching on Modal A100...")
    benchmark = run_benchmark_remote.remote(trials=trials)

    print(f"\nGPU: {benchmark['gpu']}")
    print(f"Fit times: {[f'{t:.3f}s' for t in benchmark['fit_times']]}")
    print(f"Median: {benchmark['median_time']:.3f}s")
    print(f"MSE: {benchmark['mse']:.6f}")
    print(f"R2: {benchmark['r2']:.4f}")

    # Accuracy gate
    if benchmark["mse"] > MAX_MSE:
        print()
        print("=" * 60)
        print(f"RESULT: FAIL (MSE {benchmark['mse']:.6f} > {MAX_MSE})")
        print(f"SCORE: {benchmark['median_time']:.3f}")
        print("=" * 60)
        _save_score(benchmark["median_time"], "FAIL_ACCURACY", git_sha, benchmark)
        sys.exit(1)

    # Report
    score = benchmark["median_time"]
    print()
    print("=" * 60)
    print(f"RESULT: PASS")
    print(f"SCORE: {score:.3f}")

    if previous_score and previous_score != float("inf"):
        delta = score - previous_score
        delta_pct = 100 * delta / previous_score
        sign = "+" if delta > 0 else ""
        improved = "IMPROVED" if delta < 0 else "REGRESSED" if delta > 0 else "UNCHANGED"
        print(f"PREVIOUS: {previous_score:.3f}")
        print(f"DELTA: {sign}{delta:.3f}s ({sign}{delta_pct:.1f}%) [{improved}]")

    print("=" * 60)

    _save_score(score, "PASS", git_sha, benchmark)


def _save_score(score, result, git_sha, benchmark):
    SCORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha,
        "score": score,
        "result": result,
        "gpu": benchmark.get("gpu", "unknown"),
        "mse": benchmark["mse"],
        "r2": benchmark["r2"],
        "fit_times": benchmark["fit_times"],
    }
    with open(SCORE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
