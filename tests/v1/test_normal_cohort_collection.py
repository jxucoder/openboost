"""Current comparison coverage replaces every setting in the frozen CPU cohort."""

import subprocess
import sys
from pathlib import Path


def test_current_collection_replaces_all_ninety_historical_settings():
    root = Path(__file__).resolve().parents[2]
    old = "tests/v1/test_device_normal_reference.py::test_three_round_public_transactions"
    new = "tests/v1/test_compared_normal_transactions.py::test_three_round_compared_public_transactions"
    command = [
        sys.executable, "-m", "pytest", old.split("::")[0], new.split("::")[0],
        "--collect-only", "-o", "addopts=", "-q",
    ]
    current = subprocess.check_output(command, cwd=root, text=True).splitlines()
    included = subprocess.check_output(
        [*command, "--include-historical-normal"], cwd=root, text=True,
    ).splitlines()
    original = {line.removeprefix(old) for line in included if line.startswith(old)}
    revised = {line.removeprefix(new) for line in current if line.startswith(new)}
    assert len(original) == len(revised) == 90
    assert original == revised
    assert not any(line.startswith(old) for line in current)
    assert {line for line in included if line.startswith(new)} == {
        line for line in current if line.startswith(new)
    }
