import pytest
from benchmarks.v1.a6_resource_preflight import profile_complete


def test_retained_soft_deadline_is_not_a_fit_success():
    execution = dict(status="error", exit_code=124)
    record = dict(status="deadline", soft_limit_s=60, functions=[{"function": "fit"}])
    assert profile_complete(execution, record)
    assert execution["status"] != "pass"


@pytest.mark.parametrize(
    "execution,record",
    [
        (
            {"status": "timeout", "exit_code": -9},
            {"status": "deadline", "soft_limit_s": 60, "functions": [1]},
        ),
        (
            {"status": "error", "exit_code": 1},
            {"status": "deadline", "soft_limit_s": 60, "functions": [1]},
        ),
        ({"status": "error", "exit_code": 124}, {}),
        (
            {"status": "error", "exit_code": 124},
            {"status": "deadline", "soft_limit_s": 60, "functions": []},
        ),
    ],
)
def test_missing_or_hard_failed_profile_is_rejected(execution, record):
    assert not profile_complete(execution, record)
