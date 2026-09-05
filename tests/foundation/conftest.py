"""Persist measured checks even if another required smoke case fails."""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def checks():
    values = {}
    yield values
    Path("checks.json").write_text(json.dumps(values, indent=2) + "\n")
