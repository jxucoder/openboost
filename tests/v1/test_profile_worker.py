"""Diagnostic profiles survive a bounded soft interruption."""

import json
import signal
import time

import pytest
from benchmarks.v1.profile_worker import ProfileDeadline, profile_call


def test_profile_preserves_return_and_timer(tmp_path):
    before = signal.getsignal(signal.SIGALRM)
    assert profile_call(lambda: sum(range(100)), tmp_path, 1) == 4950
    assert signal.getsignal(signal.SIGALRM) == before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert json.loads((tmp_path / "profile.json").read_text())["status"] == "complete"


def test_deadline_retains_profile(tmp_path):
    with pytest.raises(ProfileDeadline):
        profile_call(lambda: time.sleep(1), tmp_path, 0.02)
    report = json.loads((tmp_path / "profile.json").read_text())
    assert report["status"] == "deadline"
    assert report["functions"] and (tmp_path / "profile.pstats").stat().st_size
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
