"""Autoresearch evaluation harness for OpenBoost.

Runs a fixed benchmark, checks correctness, and outputs a single score.
This file is IMMUTABLE — the agent must not modify it.

Usage:
    uv run python development/autoresearch/evaluate.py
    uv run python development/autoresearch/evaluate.py --trials 5
    uv run python development/autoresearch/evaluate.py --quick
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Fixed benchmark parameters — never change these
BENCHMARK_CONFIG = {
    "n_samples": 1_000_000,
    "n_features": 100,
    "n_trees": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "loss": "mse",
    "seed": 42,
}

# Warmup config (smaller, to compile Numba JIT)
WARMUP_CONFIG = {
    "n_samples": 2_000,
    "n_trees": 3,
}

# Accuracy gate: MSE must stay below this (generous, prevents broken models)
# Realistic target has noise std=0.5, so irreducible MSE ~0.25. Gate at 1.0
# catches broken models while passing reasonable fits (R2 > 0.8).
MAX_MSE = 1.0

# Score history file
SCORE_FILE = PROJECT_ROOT / "development" / "autoresearch" / "scores.jsonl"


def generate_data(n_samples: int, n_features: int, seed: int):
    """Generate realistic mixed-feature dataset.

    Feature types: gaussian, uniform, log-normal, binary, ordinal, correlated.
    Target: non-linear with interactions, thresholds, and noise.
    """
    rng = np.random.RandomState(seed)
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

    return X, y


def run_tests() -> tuple[bool, str]:
    """Run fast tests. Returns (passed, output)."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_core.py", "tests/test_loss_correctness.py",
            "-x", "-q", "--tb=short", "-n", "0",
            "-m", "not slow and not gpu and not benchmark",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
    )
    passed = result.returncode == 0
    output = result.stdout + result.stderr
    return passed, output


def run_benchmark(trials: int = 3) -> dict:
    """Run fixed benchmark and return results."""
    import openboost as ob

    cfg = BENCHMARK_CONFIG
    X, y = generate_data(cfg["n_samples"], cfg["n_features"], cfg["seed"])

    # Warmup JIT
    warmup_model = ob.GradientBoosting(
        n_trees=WARMUP_CONFIG["n_trees"],
        max_depth=cfg["max_depth"],
        loss=cfg["loss"],
    )
    warmup_model.fit(X[:WARMUP_CONFIG["n_samples"]], y[:WARMUP_CONFIG["n_samples"]])

    # Timed trials
    fit_times = []
    for trial in range(trials):
        model = ob.GradientBoosting(
            n_trees=cfg["n_trees"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            loss=cfg["loss"],
        )
        t0 = time.perf_counter()
        model.fit(X, y)
        fit_times.append(time.perf_counter() - t0)

    # Accuracy check on last model
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
        "config": cfg,
    }


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def load_previous_score() -> float | None:
    """Load the most recent score from history."""
    if not SCORE_FILE.exists():
        return None
    try:
        lines = SCORE_FILE.read_text().strip().split("\n")
        if lines:
            last = json.loads(lines[-1])
            return last.get("score")
    except Exception:
        pass
    return None


def save_score(score: float, result: str, git_sha: str, benchmark: dict):
    """Append score to history file."""
    SCORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha,
        "score": score,
        "result": result,
        "mse": benchmark["mse"],
        "r2": benchmark["r2"],
        "fit_times": benchmark["fit_times"],
    }
    with open(SCORE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Autoresearch evaluation harness")
    parser.add_argument("--trials", type=int, default=3, help="Number of benchmark trials")
    parser.add_argument("--quick", action="store_true", help="Single trial, skip tests")
    parser.add_argument("--no-tests", action="store_true", help="Skip test suite")
    args = parser.parse_args()

    if args.quick:
        args.trials = 1
        args.no_tests = True

    git_sha = get_git_sha()
    previous_score = load_previous_score()

    print("=" * 60)
    print("AUTORESEARCH EVALUATION")
    print("=" * 60)
    print(f"Git SHA: {git_sha}")
    print(f"Config: {BENCHMARK_CONFIG['n_samples']:,} samples, "
          f"{BENCHMARK_CONFIG['n_features']} features, "
          f"{BENCHMARK_CONFIG['n_trees']} trees, "
          f"depth={BENCHMARK_CONFIG['max_depth']}")
    print(f"Trials: {args.trials}")
    print()

    # Step 1: Run tests (correctness gate)
    if not args.no_tests:
        print("--- Step 1: Correctness Tests ---")
        tests_passed, test_output = run_tests()
        if tests_passed:
            print("Tests: PASSED")
        else:
            print("Tests: FAILED")
            print(test_output[-500:] if len(test_output) > 500 else test_output)
            print()
            print("=" * 60)
            print("RESULT: FAIL (tests failed)")
            print("SCORE: N/A")
            print("=" * 60)
            save_score(float("inf"), "FAIL_TESTS", git_sha, {
                "fit_times": [], "mse": 0, "r2": 0, "median_time": 0,
                "min_time": 0, "max_time": 0, "config": BENCHMARK_CONFIG,
            })
            sys.exit(1)
    else:
        print("--- Step 1: Correctness Tests (SKIPPED) ---")

    # Step 2: Run benchmark
    print()
    print("--- Step 2: Performance Benchmark ---")
    benchmark = run_benchmark(trials=args.trials)

    print(f"Fit times: {[f'{t:.3f}s' for t in benchmark['fit_times']]}")
    print(f"Median: {benchmark['median_time']:.3f}s")
    print(f"MSE: {benchmark['mse']:.6f}")
    print(f"R2: {benchmark['r2']:.4f}")

    # Step 3: Accuracy gate
    if benchmark["mse"] > MAX_MSE:
        print()
        print("=" * 60)
        print(f"RESULT: FAIL (MSE {benchmark['mse']:.6f} > {MAX_MSE})")
        print(f"SCORE: {benchmark['median_time']:.3f}")
        print("=" * 60)
        save_score(benchmark["median_time"], "FAIL_ACCURACY", git_sha, benchmark)
        sys.exit(1)

    # Step 4: Report score
    score = benchmark["median_time"]
    result = "PASS"

    print()
    print("=" * 60)
    print(f"RESULT: {result}")
    print(f"SCORE: {score:.3f}")

    if previous_score and previous_score != float("inf"):
        delta = score - previous_score
        delta_pct = 100 * delta / previous_score
        sign = "+" if delta > 0 else ""
        improved = "IMPROVED" if delta < 0 else "REGRESSED" if delta > 0 else "UNCHANGED"
        print(f"PREVIOUS: {previous_score:.3f}")
        print(f"DELTA: {sign}{delta:.3f}s ({sign}{delta_pct:.1f}%) [{improved}]")

    print("=" * 60)

    save_score(score, result, git_sha, benchmark)


if __name__ == "__main__":
    main()
