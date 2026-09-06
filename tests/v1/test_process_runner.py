import sys

import pytest
from benchmarks.v1.process_runner import execute


def test_success_needs_artifacts(tmp_path):
    r = execute([sys.executable, "-c", 'print("done")'], tmp_path, timeout_s=2)
    assert r["status"] == "error" and r["exit_code"] == 0
    assert "missing" in r["reason"]


def test_failure_and_timeout_are_not_passes(tmp_path):
    fail = execute(
        [sys.executable, "-c", 'raise RuntimeError("broken")'], tmp_path / "fail", timeout_s=2
    )
    assert fail["status"] == "error" and fail["exit_code"] != 0
    timed = execute(
        [sys.executable, "-c", "import time; time.sleep(10)"], tmp_path / "timeout", timeout_s=0.1
    )
    assert timed["status"] == "timeout"
    assert "broken" in (tmp_path / "fail/worker.log").read_text()


def test_fresh_output_and_hashes(tmp_path):
    code = 'from pathlib import Path; Path("predictions.npz").write_bytes(b"data"); Path("model.bin").write_bytes(b"model")'
    r = execute([sys.executable, "-c", code], tmp_path, timeout_s=2)
    assert r["status"] == "pass" and len(r["artifacts"]["model.bin"]) == 64
    with pytest.raises(ValueError, match="empty"):
        execute([sys.executable, "-c", code], tmp_path, timeout_s=2)
