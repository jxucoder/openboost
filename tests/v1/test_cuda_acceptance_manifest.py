"""Run-7 freeze and permissions; diagnostic cases cannot erase conformance failures."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from benchmarks.v1.cuda_acceptance_preflight import PROTOCOL
from benchmarks.v1.cuda_aggregation_preflight import (
    check_dispatch,
    check_frozen_sources,
    judge_junit,
    snapshot_hashes,
    snapshot_paths,
)

ROOT = Path(__file__).resolve().parents[2]


def config():
    return json.loads((ROOT / PROTOCOL).read_text())


def test_exact_source_freeze_preserves_production_and_all_old_verifiers():
    p = config()
    sources = snapshot_hashes(ROOT, snapshot_paths(ROOT, PROTOCOL, p))
    check_frozen_sources(PROTOCOL, p, sources)
    old = json.loads((ROOT / "v1-sprints/090-normal-run6.json").read_text())
    assert len(sources) == p["upload_file_count"] == 70
    for name, digest in old["frozen_sources"].items():
        if name.startswith(("src/openboost/", "tests/")):
            assert sources[name] == digest
    assert p["packages"] == old["packages"]
    assert p["limits"] == old["limits"]
    assert p["normal_float32"] == old["normal_float32"] and p["float32"] == old["float32"]
    assert (
        sources["benchmarks/v1/cuda_aggregation_preflight.py"]
        == old["frozen_sources"]["benchmarks/v1/cuda_aggregation_preflight.py"]
    )
    assert p["budget"] == dict(old["budget"], run=7, previous_runs_consumed=6)
    assert p["retained_artifacts"] == old["retained_artifacts"] + [
        "normal/acceptance/forward.json",
        "normal/acceptance/reverse.json",
    ]
    assert len(set(p["retained_artifacts"])) == 78 and p["retained_artifact_bytes"] == 2 * 1024**2
    assert not any(
        name.startswith((".git/", ".codex/", ".agents/")) or "sealed" in name for name in sources
    )
    sources["tests/v1/test_device_normal_acceptance_cuda.py"] = "changed"
    with pytest.raises(ValueError, match="source freeze"):
        check_frozen_sources(PROTOCOL, p, sources)


def test_collects_all_383_original_cases_and_two_separate_diagnostics():
    p = config()
    result = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "pytest",
            *p["test_files"],
            "--collect-only",
            "-o",
            "addopts=",
            "-q",
        ],
        cwd=ROOT,
        text=True,
    )
    cases = [line for line in result.splitlines() if line.startswith("tests/")]
    old = json.loads((ROOT / "v1-sprints/090-normal-run6.json").read_text())
    assert cases == p["expected_cases"] and cases[:383] == old["expected_cases"]
    assert len(cases) == len(set(cases)) == 385
    assert all("test_capture_original_acceptance_failure" in name for name in cases[-2:])


@pytest.mark.parametrize(
    "compute,upload", [("pending", "pending"), ("approved", "pending"), ("consumed", "approved")]
)
def test_requires_new_upload_and_compute_allowances(compute, upload, tmp_path):
    p = dict(config(), authorization=compute, upload_authorization=upload)
    with pytest.raises(ValueError, match="allowance"):
        check_dispatch(tmp_path, tmp_path / p["output"], p)


def test_pending_cli_rejects_before_modal_import(monkeypatch, tmp_path):
    from benchmarks.v1.cuda_acceptance_preflight import main

    p = dict(config(), authorization="pending")
    monkeypatch.setitem(sys.modules, "modal", None)
    monkeypatch.setattr("benchmarks.v1.cuda_aggregation_preflight.json.loads", lambda _: p)
    with pytest.raises(ValueError, match="allowance"):
        main(tmp_path / "not-created", protocol_path=PROTOCOL)
    assert not (tmp_path / "not-created").exists()


def test_two_complete_diagnostics_cannot_turn_original_failures_into_pass():
    p = config()
    root = ET.Element("testsuite")
    failures = set(p["diagnostics"]["expected_unresolved_cases"])
    for node in p["expected_cases"]:
        path, name = node.split("::")
        case = ET.SubElement(root, "testcase", classname=path[:-3].replace("/", "."), name=name)
        if node in failures:
            ET.SubElement(case, "failure", message="Original failure remains a failure")
    verdict = judge_junit(ET.tostring(root), p["expected_cases"])
    assert verdict["passed"] is False
    assert len(verdict["cases"]) == 385
    assert sum(c["status"] == "fail" for c in verdict["cases"]) == 2
    assert all(c["status"] == "pass" for c in verdict["cases"][-2:])
