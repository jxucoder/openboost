"""Offline result integrity and numerical-audit counterexamples; no CUDA work."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from benchmarks.v1.replay_glm_run12 import snapshot

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "benchmarks/v1/evidence/cuda-glm-108/analyze.py"
SPEC = importlib.util.spec_from_file_location("glm_evidence_108", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_frozen_audit_and_all_original_counterexamples(tmp_path):
    archive = snapshot(tmp_path / "run12")
    # Run all eleven original audit controls with the actual executed core/oracles.
    # The old source-equality guard and every negative control remain unchanged.
    source = subprocess.check_output(
        ["git", "show", "31303e3:tests/v1/test_glm_evidence.py"], cwd=ROOT
    )
    (archive / "tests/v1/test_glm_evidence.py").write_bytes(source)
    script = """
import pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(root / 'src'), str(root)]
import openboost.objectives
assert pathlib.Path(openboost.objectives.__file__).resolve().is_relative_to(root)
import pytest
raise SystemExit(pytest.main(['-c', '/dev/null', '-n', '0', '-q',
    str(root / 'tests/v1/test_glm_evidence.py')]))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(archive)],
        cwd=archive,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "11 passed" in result.stdout


def test_archive_rejects_changed_executed_source(tmp_path, monkeypatch):
    from benchmarks.v1 import replay_glm_run12

    check_output = subprocess.check_output

    def changed(command, **kwargs):
        result = check_output(command, **kwargs)
        if command[:2] == ["git", "show"] and command[2].endswith(":src/openboost/objectives.py"):
            return result + b"\n# changed\n"
        return result

    monkeypatch.setattr(replay_glm_run12.subprocess, "check_output", changed)
    with pytest.raises(ValueError, match="dispatch source differs"):
        snapshot(tmp_path / "run12")


@pytest.mark.parametrize("fault", [None, "bounds", "identity", "method"])
def test_independent_difference_rejects_invalid_comparison_claims(fault):
    arrays = ([0], [0.01], [1], [0], [1], [1])
    exact = str(audit.direct_difference("binary", *arrays))
    record = dict(
        lower=-1,
        upper=0,
        unchanged=False,
        method="binary-convex-taylor18-interval-v1",
        reason="test",
    )
    if fault == "bounds":
        record.update(lower=1, upper=2)
    elif fault == "identity":
        record.update(lower=0, upper=0, unchanged=True)
    elif fault == "method":
        record["method"] = "unverified"
    result = audit.comparison("binary", arrays, record, exact)
    assert all(result[k] for k in ("encloses", "identity_matches", "method_matches")) is (
        fault is None
    )
    with pytest.raises(ValueError, match="direct likelihood differs"):
        audit.comparison("binary", arrays, record, "1")
