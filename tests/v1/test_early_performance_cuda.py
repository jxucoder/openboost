"""104 actual same-host CPU/CUDA performance checkpoint; all failures retained."""

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pytest
from benchmarks.v1.performance_judge import judge

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ("squared", 10_000, 45),
    ("squared", 100_000, 60),
    ("normal", 1_000, 90),
    ("normal", 10_000, 30),
    ("normal", 100_000, 30),
]


def child(recipe, rows, backend, destination, timeout, *, profile=False):
    python = os.environ["OPENBOOST_FRESH_CPU_PYTHON"] if backend == "cpu" else sys.executable
    destination.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    cache = destination.parent / (backend + ("-profile" if profile else "") + "-cache")
    environment.update(CUPY_CACHE_DIR=str(cache / "cupy"), CUDA_CACHE_PATH=str(cache / "driver"))
    command = [
        python,
        str(ROOT / "benchmarks/v1/performance_checkpoint.py"),
        "--recipe",
        recipe,
        "--rows",
        str(rows),
        "--backend",
        backend,
        "--output",
        str(destination),
    ] + (["--profile"] if profile else [])
    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            env=environment,
            cwd=destination.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        status = "complete" if completed.returncode == 0 else "error"
        log = completed.stdout + completed.stderr
        code = completed.returncode
    except subprocess.TimeoutExpired as error:
        status, code = "timeout", None

        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

        log = decode(error.stdout) + decode(error.stderr)
    result = json.loads(destination.read_text()) if destination.exists() else {}
    result.update(
        status=status,
        child_exit_code=code,
        child_wall_seconds=perf_counter() - started,
        child_timeout_seconds=timeout,
        child_argv=command,
        child_log=log[-32768:],
    )
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


@pytest.mark.parametrize("recipe,rows,cpu_timeout", CASES, ids=[f"{r}-{n}" for r, n, _ in CASES])
def test_same_host_performance_and_quality(recipe, rows, cpu_timeout):
    destination = Path(os.environ["OPENBOOST_NORMAL_ARTIFACTS"]) / "checkpoint" / f"{recipe}-{rows}"
    cpu = child(recipe, rows, "cpu", destination / "cpu.json", cpu_timeout)
    cuda = child(recipe, rows, "cuda", destination / "cuda.json", 50)
    protocol = json.loads((ROOT / "v1-sprints/104-performance-run10.json").read_text())
    sources = {
        p: h for p, h in protocol["frozen_sources"].items() if p.startswith("src/openboost/")
    }
    result = judge(cpu, cuda, expected_sources=sources)
    (destination / "judgment.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["measurement_complete"], result["reasons"]
    assert result["quality_comparable"], result["reasons"]


def test_separate_cold_and_warm_profile():
    destination = Path(os.environ["OPENBOOST_NORMAL_ARTIFACTS"]) / "checkpoint" / "profile.json"
    result = child("normal", 100_000, "cuda", destination, 60, profile=True)
    assert result["status"] == "complete"
    assert result["profile"] and len(result["profile_calls"]) == 2
    assert all(row["calls"] for row in result["profile_calls"])
    assert all(m["final_metrics"]["live_bytes"] == 0 for m in result["measurements"])
