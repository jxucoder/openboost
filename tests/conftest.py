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


def pytest_addoption(parser):
    parser.addoption(
        "--include-historical-normal",
        action="store_true",
        help="Include the frozen 090 full-loss Normal cohort (not current conformance).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--include-historical-normal"):
        return
    prefix = "tests/v1/test_device_normal_reference.py::test_three_round_public_transactions["
    historical = [item for item in items if item.nodeid.startswith(prefix)]
    if historical:
        items[:] = [item for item in items if item not in historical]
        config.hook.pytest_deselected(items=historical)
