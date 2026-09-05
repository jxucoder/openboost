"""The evidence gate must reject apparently successful but incomplete runs."""

import json
import subprocess
import sys

import pytest
from benchmarks.foundation.runner import validate_result


@pytest.fixture
def evidence():
    manifest = {"wheel_sha256": "abc", "source_sha": "def", "source_dirty": False}
    result = {
        "returncode": 0,
        "timed_out": False,
        "junit": '<testsuites><testsuite><testcase name="test_device_interop" />'
        '<testcase name="test_normal_gpu_fit" /></testsuite></testsuites>',
        "checks": {
            "interop": True,
            "native_tree_calls": 4,
            "device_objective_calls": 2,
            "installed_files_verified": 10,
            "dataset_sha256": "123",
        },
        "environment": {"cuda_available": True, "gpu_name": "Tesla T4"},
        "wheel_sha256": "abc",
        "source_sha": "def",
    }
    return manifest, result


def test_complete_result(evidence):
    validate_result(*evidence)


@pytest.mark.parametrize(
    "fault",
    [
        "exit",
        "timeout",
        "missing",
        "skip",
        "failure",
        "duplicate",
        "cuda",
        "wheel",
        "source",
        "fallback",
    ],
)
def test_incomplete_result_rejected(evidence, fault):
    manifest, result = evidence
    if fault == "exit":
        result["returncode"] = 1
    elif fault == "timeout":
        result["timed_out"] = True
    elif fault == "missing":
        result["junit"] = "<testsuites/>"
    elif fault in ("skip", "failure"):
        tag = "skipped" if fault == "skip" else "failure"
        result["junit"] = result["junit"].replace(
            'name="test_device_interop" />', f'name="test_device_interop"><{tag}/></testcase>'
        )
    elif fault == "duplicate":
        result["junit"] = result["junit"].replace("test_normal_gpu_fit", "test_device_interop")
    elif fault == "cuda":
        result["environment"]["cuda_available"] = False
    elif fault == "wheel":
        result["wheel_sha256"] = "wrong"
    elif fault == "source":
        result["source_sha"] = "wrong"
    else:
        result["checks"]["device_objective_calls"] = 0
    with pytest.raises(ValueError):
        validate_result(manifest, result)


def test_cli_returns_nonzero_for_missing_report(tmp_path, evidence):
    manifest, _ = evidence
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    run = subprocess.run(
        [sys.executable, "-m", "benchmarks.foundation.runner", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert "results.json" in run.stderr
