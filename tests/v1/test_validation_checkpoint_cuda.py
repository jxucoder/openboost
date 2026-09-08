"""105 frozen same-input GPU comparison; all child failures retain their evidence."""

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from benchmarks.v1.build_validation_baseline import build
from benchmarks.v1.performance_checkpoint import workload
from benchmarks.v1.performance_evidence import input_record, write_json
from benchmarks.v1.validation_judge import judge

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "v1-sprints/105-validation-run11.json"


def configuration():
    return json.loads(PROTOCOL.read_text())


def artifacts():
    return Path(os.environ["OPENBOOST_NORMAL_ARTIFACTS"]) / "validation"


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    try:
        result = build(ROOT, tmp_path_factory.mktemp("baseline"), configuration())
    except Exception as error:
        write_json(artifacts() / "build.json", dict(passed=False, error=str(error)))
        raise
    write_json(artifacts() / "build.json", result)
    return result["python"]


def child(python, inputs, output, seconds, cache, *, cpu=False, profile=False):
    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        CUPY_CACHE_DIR=str(cache / "cupy"),
        CUDA_CACHE_PATH=str(cache / "driver"),
    )
    record = json.loads(inputs.read_text())
    command = [
        str(python),
        "-m",
        "benchmarks.v1.validation_checkpoint",
        "--inputs",
        str(inputs),
        "--input-sha",
        record["sha256"],
        "--output",
        str(output),
        "--backend",
        "cpu" if cpu else "cuda",
    ] + (["--profile"] if profile else [])
    start = perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cache.parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=seconds,
        )
        status, code = ("complete" if completed.returncode == 0 else "error"), completed.returncode
        log = completed.stdout + completed.stderr
    except subprocess.TimeoutExpired as error:
        status, code = "timeout", None

        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

        log = decode(error.stdout) + decode(error.stderr)
    result = json.loads(output.read_text()) if output.exists() else {}
    result.update(
        status=status,
        child_exit_code=code,
        child_argv=command,
        child_timeout_seconds=seconds,
        child_wall_seconds=perf_counter() - start,
        child_log=log[-32768:],
    )
    write_json(output, result)
    return result


@pytest.mark.parametrize("case", ["squared-10000", "squared-100000", "normal-10000"])
def test_paired_fits_preserve_quality_and_meet_cost_gate(case, baseline, tmp_path):
    protocol = configuration()
    bounds = protocol["measurement_cases"][case]
    recipe, rows = case.split("-")
    inputs = input_record(*workload(int(rows), recipe), recipe)
    destination = artifacts() / case
    path = destination / "inputs.json"
    write_json(path, inputs)
    cpu = None
    if "cpu_seconds" in bounds:
        cpu = child(
            os.environ["OPENBOOST_FRESH_CPU_PYTHON"],
            path,
            destination / "cpu.json",
            bounds["cpu_seconds"],
            tmp_path / "cpu",
            cpu=True,
        )
    original = child(
        baseline,
        path,
        destination / "baseline.json",
        bounds["gpu_seconds_per_arm"],
        tmp_path / "baseline",
    )
    candidate = child(
        sys.executable,
        path,
        destination / "candidate.json",
        bounds["gpu_seconds_per_arm"],
        tmp_path / "candidate",
    )
    sources = {
        p: h for p, h in protocol["frozen_sources"].items() if p.startswith("src/openboost/")
    }
    result = judge(
        original,
        candidate,
        inputs,
        baseline_sources=protocol["baseline_sources"],
        candidate_sources=sources,
        limit=bounds["candidate_over_baseline_limit"],
        cpu=cpu,
    )
    write_json(destination / "judgment.json", result)
    assert result["measurement_complete"] and result["quality_passed"], result
    assert result["cost_passed"], result


def test_separate_validation_operation_profile(baseline, tmp_path):
    protocol = configuration()
    path = artifacts() / "squared-100000/inputs.json"
    # The profile is independent of preceding timing assertions or case order.
    if not path.exists():
        write_json(path, input_record(*workload(100000, "squared"), "squared"))
    results = {}
    for arm, python in (("baseline", baseline), ("candidate", sys.executable)):
        results[arm] = child(
            python,
            path,
            artifacts() / f"profile-{arm}.json",
            protocol["profile_seconds_per_arm"],
            tmp_path / arm,
            profile=True,
        )
    expected = {
        p: h for p, h in protocol["frozen_sources"].items() if p.startswith("src/openboost/")
    }
    for arm, result in results.items():
        assert result["status"] == "complete", result
        assert result["profile"] and result["shape"] == [100000, 2]
        assert result["input_sha256"] == json.loads(path.read_text())["sha256"]
        assert result["environment"]["core_sources"] == (
            expected if arm == "candidate" else protocol["baseline_sources"]
        )
        assert result["live_bytes_after_release"] == result["final_metrics"]["live_bytes"] == 0
        assert set(result["launches_by_name"]) == {"validate_fields"}
        assert result["launches_by_name"]["validate_fields"]["calls"] == 11
        assert len(result["samples"]) == 11
        assert all(
            np.isfinite(s["stream_interval_ms"]) and s["stream_interval_ms"] >= 0
            for s in result["samples"]
        )
