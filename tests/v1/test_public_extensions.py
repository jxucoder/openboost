"""Source development checks; installed-wheel proof uses the separate verifier."""

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "examples/v1_extensions"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_development_extension_oracles(tmp_path):
    cohort = load(ROOT / "cohort_splits/src/ob_cohort_splits/__init__.py", "cohort")
    leaves = load(ROOT / "penalized_leaves/src/ob_penalized_leaves/__init__.py", "leaves")
    checks = load(ROOT / "checks.py", "checks")
    checks.run_checks(cohort, leaves, tmp_path)


def test_extensions_use_only_public_openboost_imports():
    for path in ROOT.glob("*/src/*/*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("openboost"):
                assert not any(p.startswith("_") for p in node.module.split("."))
                assert not any(a.name.startswith("_") for a in node.names)
