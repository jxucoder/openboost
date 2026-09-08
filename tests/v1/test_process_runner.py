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


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_address_ceiling_rejected_before_output(tmp_path, limit):
    with pytest.raises(ValueError, match="address"):
        execute(
            [sys.executable, "-c", "pass"],
            tmp_path / "worker",
            timeout_s=1,
            address_limit_bytes=limit,
        )
    assert not (tmp_path / "worker").exists()


def test_privilege_mode_rejects_unsupported_host_before_output(tmp_path, monkeypatch):
    from benchmarks.v1 import process_runner

    monkeypatch.setattr(process_runner.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="Linux"):
        execute(
            [sys.executable, "-c", "pass"],
            tmp_path / "worker",
            timeout_s=1,
            unprivileged=True,
            address_limit_bytes=8 * 1024**3,
        )
    assert not (tmp_path / "worker").exists()
