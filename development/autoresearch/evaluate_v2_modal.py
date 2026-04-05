"""Autoresearch v2 evaluation harness — runs on Modal A100.

Comprehensive evaluation across three dimensions:
  1. Speed: wall-clock training time vs XGBoost across scales
  2. Accuracy: metric parity vs XGBoost on synthetic datasets
  3. Coverage: feature completeness (missing values, categoricals, variants)

Usage:
    uv run modal run development/autoresearch/evaluate_v2_modal.py
    uv run modal run development/autoresearch/evaluate_v2_modal.py --quick
    uv run modal run development/autoresearch/evaluate_v2_modal.py --trials 5
    uv run modal run development/autoresearch/evaluate_v2_modal.py --no-xgb
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCORE_FILE = PROJECT_ROOT / "development" / "autoresearch" / "scores_v2.jsonl"

app = modal.App("openboost-autoresearch-v2")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install(
        "numpy>=1.24",
        "numba>=0.60",
        "joblib>=1.3",
        "xgboost>=2.0",
        "scikit-learn>=1.3",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "openboost"),
        remote_path="/root/openboost_pkg/openboost",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "development" / "autoresearch" / "lib"),
        remote_path="/root/autoresearch_lib/lib",
    )
    .add_local_file(
        str(PROJECT_ROOT / "development" / "autoresearch" / "lib" / "__init__.py"),
        remote_path="/root/autoresearch_lib/__init__.py",
    )
)


# ===========================================================================
# Remote function — runs on A100
# ===========================================================================

@app.function(gpu="A100", image=image, timeout=3600)
def run_evaluation_remote(
    trials: int = 3,
    quick: bool = False,
    include_xgb: bool = True,
) -> dict:
    """Run the full v2 evaluation on a Modal A100 GPU."""
    import sys
    sys.path.insert(0, "/root/openboost_pkg")
    sys.path.insert(0, "/root/autoresearch_lib")

    import openboost as ob
    ob.set_backend("cuda")

    from numba import cuda
    gpu_name = cuda.get_current_device().name
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    print(f"GPU: {gpu_name}")
    print(f"Backend: {ob.get_backend()}")

    from lib.speed_bench import run_speed_benchmarks
    from lib.accuracy_bench import run_accuracy_benchmarks
    from lib.coverage_bench import run_coverage_benchmarks
    from lib.scoring import compute_composite_score

    # --- Tier 1: Speed ---
    print("\n" + "=" * 60)
    print("TIER 1: SPEED BENCHMARKS")
    print("=" * 60)
    speed_result = run_speed_benchmarks(
        trials=trials, include_xgb=include_xgb, quick=quick
    )
    print(f"\n  Speed score: {speed_result['score']:.3f}  "
          f"gates: {'PASS' if speed_result['gates_passed'] else 'FAIL'}")

    # --- Tier 2: Accuracy ---
    print("\n" + "=" * 60)
    print("TIER 2: ACCURACY BENCHMARKS")
    print("=" * 60)
    accuracy_result = run_accuracy_benchmarks(
        include_xgb=include_xgb, quick=quick
    )
    print(f"\n  Accuracy score: {accuracy_result['score']:.3f}  "
          f"gates: {'PASS' if accuracy_result['gates_passed'] else 'FAIL'}")

    # --- Tier 3: Coverage ---
    print("\n" + "=" * 60)
    print("TIER 3: FEATURE COVERAGE")
    print("=" * 60)
    coverage_result = run_coverage_benchmarks(quick=quick)
    print(f"\n  Coverage score: {coverage_result['score']:.3f}  "
          f"gates: {'PASS' if coverage_result['gates_passed'] else 'FAIL'}  "
          f"({coverage_result['passed_count']}/{coverage_result['total_count']} passed)")

    # --- Composite ---
    composite, status = compute_composite_score(
        speed_result, accuracy_result, coverage_result
    )

    return {
        "version": 2,
        "composite_score": composite,
        "status": status,
        "speed": speed_result,
        "accuracy": accuracy_result,
        "coverage": coverage_result,
        "gpu": gpu_name,
    }


# ===========================================================================
# Local entrypoint
# ===========================================================================

def _run_test_gate() -> tuple[bool, str]:
    """Run fast tests locally as a correctness gate."""
    import subprocess

    print("Running test gate...")
    result = subprocess.run(
        [
            "uv", "run", "pytest",
            "tests/test_core.py", "tests/test_loss_correctness.py",
            "-x", "-q", "-n", "0",
        ],
        capture_output=True, text=True, timeout=120,
        cwd=str(PROJECT_ROOT),
        env={**__import__("os").environ, "OPENBOOST_BACKEND": "cpu"},
    )
    passed = result.returncode == 0
    output = result.stdout + result.stderr
    if passed:
        print("  Test gate: PASS")
    else:
        print("  Test gate: FAIL")
        print(output[-500:] if len(output) > 500 else output)
    return passed, output


def _save_score(result: dict, git_sha: str):
    """Append evaluation result to scores_v2.jsonl."""
    SCORE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Flatten for the score line — keep it readable but complete
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha,
        "composite_score": result["composite_score"],
        "status": result["status"],
        "gpu": result.get("gpu", "unknown"),
        "speed_score": result["speed"]["score"],
        "accuracy_score": result["accuracy"]["score"],
        "coverage_score": result["coverage"]["score"],
        "speed_geo_mean_speedup": result["speed"].get("geo_mean_speedup"),
        "accuracy_geo_mean_parity": result["accuracy"].get("geo_mean_parity"),
        "coverage_passed": f"{result['coverage']['passed_count']}/{result['coverage']['total_count']}",
    }
    with open(SCORE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _print_report(result: dict, previous_composite: float | None = None):
    """Print human-readable evaluation report."""
    print()
    print("=" * 60)
    print("AUTORESEARCH V2 EVALUATION REPORT")
    print("=" * 60)

    status = result["status"]
    composite = result["composite_score"]

    print(f"  Status:     {status}")
    if composite != float("inf"):
        print(f"  Composite:  {composite:.4f}")
    else:
        print(f"  Composite:  FAIL")
    print()

    # Sub-scores
    s = result["speed"]
    a = result["accuracy"]
    c = result["coverage"]
    print(f"  Speed:      {s['score']:.3f}  "
          f"({'PASS' if s['gates_passed'] else 'FAIL'})"
          f"  geo_mean_speedup={s.get('geo_mean_speedup', 'N/A')}")
    print(f"  Accuracy:   {a['score']:.3f}  "
          f"({'PASS' if a['gates_passed'] else 'FAIL'})"
          f"  geo_mean_parity={a.get('geo_mean_parity', 'N/A')}")
    print(f"  Coverage:   {c['score']:.3f}  "
          f"({'PASS' if c['gates_passed'] else 'FAIL'})"
          f"  {c['passed_count']}/{c['total_count']} tests passed")

    # Speed details
    if s.get("configs"):
        print()
        print("  Speed details:")
        for name, cfg in s["configs"].items():
            ob_t = cfg["ob_median_s"]
            xgb_t = cfg.get("xgb_median_s")
            speedup = cfg.get("speedup")
            xgb_str = f"  XGB={xgb_t:.3f}s  speedup={speedup:.2f}x" if xgb_t else ""
            print(f"    {name:15s}  OB={ob_t:.3f}s{xgb_str}")

    # Accuracy details
    if a.get("datasets"):
        print()
        print("  Accuracy details:")
        for name, ds in a["datasets"].items():
            ob_m = ds["ob_metrics"]
            parity = ds["parity"]
            metric_name = ob_m["primary_name"]
            metric_val = ob_m["primary"]
            xgb_str = ""
            if ds.get("xgb_metrics"):
                xgb_val = ds["xgb_metrics"]["primary"]
                xgb_str = f"  XGB={xgb_val:.4f}  parity={parity:.3f}"
            print(f"    {name:20s}  OB {metric_name}={metric_val:.4f}{xgb_str}")

    # Coverage details
    if c.get("tests"):
        print()
        print("  Coverage details:")
        for name, test in c["tests"].items():
            status_str = "PASS" if test["passed"] else "FAIL"
            error_str = f"  ({test['error']})" if test.get("error") else ""
            print(f"    {name:20s}  {status_str}{error_str}")

    # Delta from previous
    if previous_composite and previous_composite != float("inf") and composite != float("inf"):
        delta = composite - previous_composite
        delta_pct = 100 * delta / previous_composite
        sign = "+" if delta > 0 else ""
        direction = "IMPROVED" if delta > 0 else "REGRESSED" if delta < 0 else "UNCHANGED"
        print()
        print(f"  PREVIOUS: {previous_composite:.4f}")
        print(f"  DELTA:    {sign}{delta:.4f} ({sign}{delta_pct:.1f}%) [{direction}]")

    print("=" * 60)


@app.local_entrypoint()
def main(
    trials: int = 3,
    quick: bool = False,
    no_xgb: bool = False,
    skip_tests: bool = False,
):
    """Run autoresearch v2 evaluation on Modal A100."""
    import subprocess

    if quick:
        trials = 1

    # Git SHA
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

    # Load previous composite score
    previous_composite = None
    if SCORE_FILE.exists():
        try:
            lines = SCORE_FILE.read_text().strip().split("\n")
            if lines and lines[-1]:
                previous_composite = json.loads(lines[-1]).get("composite_score")
        except Exception:
            pass

    print("=" * 60)
    print("AUTORESEARCH V2 EVALUATION (Modal A100)")
    print("=" * 60)
    print(f"Git SHA: {git_sha}")
    print(f"Trials: {trials}")
    print(f"Quick: {quick}")
    print(f"XGBoost comparison: {not no_xgb}")
    print()

    # Gate 0: Fast tests
    if not skip_tests and not quick:
        tests_passed, test_output = _run_test_gate()
        if not tests_passed:
            fail_result = {
                "version": 2,
                "composite_score": float("inf"),
                "status": "FAIL_TESTS",
                "speed": {"score": 0, "gates_passed": False, "configs": {}, "geo_mean_speedup": None},
                "accuracy": {"score": 0, "gates_passed": False, "datasets": {}, "geo_mean_parity": None},
                "coverage": {"score": 0, "gates_passed": False, "tests": {}, "passed_count": 0, "total_count": 0},
                "gpu": "N/A",
            }
            _print_report(fail_result)
            _save_score(fail_result, git_sha)
            sys.exit(1)

    # Run benchmarks on Modal
    print("\nLaunching on Modal A100...")
    eval_result = run_evaluation_remote.remote(
        trials=trials, quick=quick, include_xgb=not no_xgb
    )

    # Report
    _print_report(eval_result, previous_composite)
    _save_score(eval_result, git_sha)

    # Exit code
    if eval_result["status"] != "PASS":
        print(f"\nRESULT: {eval_result['status']}")
        print(f"SCORE: {eval_result['composite_score']}")
        sys.exit(1)

    print(f"\nRESULT: PASS")
    print(f"SCORE: {eval_result['composite_score']:.4f}")
    print(f"SPEED: {eval_result['speed']['score']:.3f}")
    print(f"ACCURACY: {eval_result['accuracy']['score']:.3f}")
    print(f"COVERAGE: {eval_result['coverage']['score']:.3f}")
