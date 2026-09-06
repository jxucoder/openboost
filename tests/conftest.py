"""Collect the new v1 suite; historical tests remain as evidence, not current CI.

The original conftest and production fixtures are available at revision
50acfc6. Run historical tests in that checkout, not against the new namespace.
"""

from pathlib import Path

_ROOT = Path(__file__).parent
collect_ignore = [
    path.name
    for path in _ROOT.iterdir()
    if path.name != "v1"
    and (path.is_dir() or (path.name.startswith("test_") and path.suffix == ".py"))
]
