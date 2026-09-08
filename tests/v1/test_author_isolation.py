"""Local supervisor failure/timeout semantics; native policy needs actual smoke."""

import os
import sys

from benchmarks.v1.authoring.isolation import invoke


def test_supervisor_retains_timeout_log_and_killed_status(tmp_path):
    result = invoke(
        [sys.executable, "-c", "import time; print('started',flush=True); time.sleep(30)"],
        tmp_path,
        os.environ.copy(),
        0.25,
    )
    assert result["status"] == "timeout" and result["exit_code"] == -9
    assert "started" in result["stdout"] and result["wall_limit_s"] == 0.25


def test_nonzero_command_is_not_reclassified_as_timeout(tmp_path):
    result = invoke(
        [sys.executable, "-c", "raise ValueError('deliberate')"], tmp_path, os.environ.copy(), 5
    )
    assert result["status"] == "complete" and result["exit_code"] != 0
    assert "deliberate" in result["stderr"]
