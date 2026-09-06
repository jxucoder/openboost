"""Prove that importing and running reference code does not load production."""

import subprocess
import sys
from pathlib import Path


def test_reference_runs_with_production_imports_blocked():
    script = """
import importlib.abc
import sys

class BlockProduction(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openboost" or fullname.startswith("openboost."):
            raise AssertionError("reference tried to import production: " + fullname)

sys.meta_path.insert(0, BlockProduction())
sys.path.insert(0, sys.argv[1])
from tests.v1.reference.tree import boost_squared
for policy in ("depthwise", "best_first", "symmetric"):
    result = boost_squared([[0], [0], [1], [1]], [-2, -2, 2, 2], policy=policy)
    assert abs(result.predict([[0]])[0] + 58/225) < 1e-12
assert not any(n == "openboost" or n.startswith("openboost.") for n in sys.modules)
print("independent-reference-ok")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(__file__).resolve().parents[2])],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stdout.strip() == "independent-reference-ok"
