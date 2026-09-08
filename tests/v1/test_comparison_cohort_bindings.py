"""Every historical requirement has a collected revised counterpart, never a claimed pass."""

import hashlib
import json
import subprocess
import sys

from benchmarks.v1.comparison_cohorts import (
    BINDINGS,
    HISTORICAL,
    REFERENCE,
    ROOT,
    bindings,
    trajectories,
)


def test_binding_file_covers_all_historical_cases_and_actual_collection():
    saved = json.loads((ROOT / BINDINGS).read_text())
    assert saved == bindings()
    assert (
        saved["historical_mapping_sha256"]
        == hashlib.sha256((ROOT / HISTORICAL).read_bytes()).hexdigest()
    )
    nodes = [c["revised_node"] for c in saved["cases"]]
    assert len(nodes) == len(set(nodes)) == 383
    files = list(dict.fromkeys(n.split("::")[0] for n in nodes))
    result = subprocess.check_output(
        [sys.executable, "-m", "pytest", *files, "--collect-only", "-o", "addopts=", "-q"],
        cwd=ROOT,
        text=True,
    )
    collected = [line for line in result.splitlines() if line.startswith("tests/")]
    assert collected == nodes
    assert all(c["revised_status"] == "collected_not_run" for c in saved["cases"])
    assert [c["historical_outcome"] for c in saved["cases"]].count("failed") == 2


def test_reference_freeze_reproduces_all_ninety_trajectories_and_preserves_old_sources():
    saved = json.loads((ROOT / REFERENCE).read_text())
    assert saved == trajectories()
    assert len(saved["cases"]) == 90 and saved["device_execution"] is False
    old = json.loads((ROOT / HISTORICAL).read_text())
    for case in old["cases"]:
        file = case["historical_node"].split("::")[0]
        assert (
            hashlib.sha256((ROOT / file).read_bytes()).hexdigest()
            == case["historical_source_sha256"]
        )
    sources = json.loads(
        (ROOT / "benchmarks/v1/evidence/normal-comparison-092/study.json").read_text()
    )["sources"]
    for file, digest in sources.items():
        if file.startswith("tests/v1/reference/"):
            assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest
