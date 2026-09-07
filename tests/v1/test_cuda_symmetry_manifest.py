"""Run-5 upload freeze and dispatch guards; CPU checks are not CUDA evidence."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from benchmarks.v1.cuda_aggregation_preflight import (
    check_dispatch,
    check_frozen_sources,
    snapshot_hashes,
    snapshot_paths,
)
from benchmarks.v1.cuda_symmetry_preflight import PROTOCOL

REPO = Path(__file__).resolve().parents[2]


def protocol():
    return json.loads((REPO / PROTOCOL).read_text())


def test_exact_upload_and_unchanged_original_verifiers():
    config = protocol()
    assert PROTOCOL == "v1-sprints/089-symmetry-run5.json"
    assert config["authorization"] in ("pending", "approved", "consumed")
    assert config["upload_authorization"] in ("pending", "approved")
    assert config["require_upload_authorization"] is True
    assert config["upload_destination"] == "Modal"
    if config["authorization"] == "consumed":
        sources = json.loads((REPO / config["output"] / "manifest.json").read_text())["sources"]
    else:
        sources = snapshot_hashes(REPO, snapshot_paths(REPO, PROTOCOL, config))
    assert len(sources) == config["upload_file_count"] == 49
    assert sum(p.startswith("src/openboost/") for p in sources) == 27
    assert all(not p.startswith((".git/", ".codex/", ".agents/")) for p in sources)
    check_frozen_sources(PROTOCOL, config, sources)
    prior = json.loads((REPO / "v1-sprints/078-resident-run4.json").read_text())
    assert config["packages"] == prior["packages"]
    assert config["limits"] == prior["limits"]
    assert config["float32"] == prior["float32"]
    assert config["metric_relative_scale"] == prior["metric_relative_scale"]
    for path, digest in prior["frozen_sources"].items():
        if path.startswith("tests/"):
            assert sources[path] == digest
    assert (
        sources["src/openboost/_device_kernels.py"]
        != prior["frozen_sources"]["src/openboost/_device_kernels.py"]
    )
    assert config["budget"] == dict(
        run=5,
        previous_runs_consumed=4,
        additional_runs=1,
        gpu="T4",
        function_seconds=900,
        test_seconds=600,
        retries=0,
    )
    sources["tests/v1/run4_score_kernel.py"] = "changed"
    with pytest.raises(ValueError, match="source freeze"):
        check_frozen_sources(PROTOCOL, config, sources)


def test_exact_matrix_retains_every_original_case():
    config = protocol()
    if config["authorization"] == "consumed":
        root = ET.parse(REPO / config["output"] / "junit.xml").getroot()
        cases = [
            case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
            for case in root.iter("testcase")
        ]
    else:
        collected = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "pytest",
                *config["test_files"],
                "--collect-only",
                "-o",
                "addopts=",
                "-q",
            ],
            cwd=REPO,
            text=True,
        )
        cases = [line for line in collected.splitlines() if line.startswith("tests/")]
    assert cases == config["expected_cases"]
    assert len(cases) == len(set(cases)) == 212
    prior = json.loads((REPO / "v1-sprints/078-resident-run4.json").read_text())
    assert set(prior["expected_cases"]) <= set(cases)
    assert (
        len(set(cases) - set(prior["expected_cases"])) == config["diagnostics"]["new_cases"] == 10
    )
    assert cases[0].endswith("::test_weighted_root_scores_and_archived_kernel_diagnostics")


def test_new_dispatch_requires_both_approvals_and_fresh_output(tmp_path):
    config = protocol()
    output = tmp_path / config["output"]
    for compute, upload in (
        ("pending", "pending"),
        ("consumed", "approved"),
        ("approved", "pending"),
    ):
        with pytest.raises(ValueError, match="allowance"):
            check_dispatch(
                tmp_path, output, dict(config, authorization=compute, upload_authorization=upload)
            )
        assert not output.exists()
    approved = dict(config, authorization="approved", upload_authorization="approved")
    check_dispatch(tmp_path, output, approved)
    with pytest.raises(ValueError, match="location"):
        check_dispatch(tmp_path, tmp_path / "other", approved)
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot be reused"):
        check_dispatch(tmp_path, output, approved)


def test_pending_cli_does_not_import_modal_or_create_output(monkeypatch, tmp_path):
    from benchmarks.v1 import cuda_symmetry_preflight

    config = protocol()
    if config["authorization"] == "approved" and config["upload_authorization"] == "approved":
        # Approved dispatch is never exercised by a local CPU test.
        config = dict(config, authorization="pending")
    monkeypatch.setattr("benchmarks.v1.cuda_aggregation_preflight.json.loads", lambda _: config)
    monkeypatch.setitem(sys.modules, "modal", None)
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="allowance"):
        cuda_symmetry_preflight.main(output, protocol_path=PROTOCOL)
    assert not output.exists()
