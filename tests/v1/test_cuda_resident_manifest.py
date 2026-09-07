"""Local run-4 freeze and judging checks; synthetic JUnit is never device evidence."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from benchmarks.v1.cuda_aggregation_preflight import (
    check_dispatch,
    check_frozen_sources,
    judge_run,
    snapshot_hashes,
    snapshot_paths,
)
from benchmarks.v1.cuda_resident_preflight import PROTOCOL

REPO = Path(__file__).resolve().parents[2]


def protocol():
    return json.loads((REPO / PROTOCOL).read_text())


def sources_for(config):
    if config["authorization"] == "consumed":
        return json.loads((REPO / config["output"] / "manifest.json").read_text())["sources"]
    return snapshot_hashes(REPO, snapshot_paths(REPO, PROTOCOL, config))


def test_exact_sources_upload_closure_and_run_limits():
    config = protocol()
    assert config["authorization"] in ("pending", "approved", "consumed")
    assert config["upload_authorization"] in ("pending", "approved")
    assert config["require_upload_authorization"] is True
    assert config["upload_destination"] == "Modal"
    assert config["upload_file_count"] == 47
    assert config["budget"] == dict(
        run=4,
        previous_runs_consumed=3,
        additional_runs=1,
        gpu="T4",
        function_seconds=900,
        test_seconds=600,
        retries=0,
    )
    assert config["limits"] == dict(
        rows=8192,
        features=32,
        bins=32,
        pool_bytes=16 * 1024**2,
        training_rows=8,
        training_depth=2,
        training_rounds=24,
    )
    assert config["float32"] == dict(rtol=1e-4, atol=1e-5)
    assert config["metric_relative_scale"] == 1e-3
    sources = sources_for(config)
    assert len(sources) == 47 and sum(p.startswith("src/openboost/") for p in sources) == 27
    assert all(not p.startswith((".git/", ".codex/", ".agents/")) for p in sources)
    check_frozen_sources(PROTOCOL, config, sources)
    sources["src/openboost/device_runtime.py"] = "changed"
    with pytest.raises(ValueError, match="source freeze"):
        check_frozen_sources(PROTOCOL, config, sources)


def test_exact_case_matrix_retains_all_previous_cases_and_dependencies():
    config = protocol()
    if config["authorization"] == "consumed":
        root = ET.parse(REPO / config["output"] / "junit.xml").getroot()
        nodes = [
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
        nodes = [line for line in collected.splitlines() if line.startswith("tests/")]
    assert nodes == config["expected_cases"]
    assert len(nodes) == len(set(nodes)) == 202
    prior = json.loads((REPO / "v1-sprints/078-splits-run3.json").read_text())
    assert len(set(nodes) - set(prior["expected_cases"])) == 114
    assert set(prior["expected_cases"]) <= set(nodes)
    assert config["packages"] == prior["packages"]


@pytest.mark.parametrize("authorization", ["pending", "consumed"])
def test_missing_or_consumed_compute_allowance_blocks_without_output(tmp_path, authorization):
    config = dict(protocol(), authorization=authorization, upload_authorization="approved")
    output = tmp_path / config["output"]
    with pytest.raises(ValueError, match="allowance"):
        check_dispatch(tmp_path, output, config)
    assert not output.exists()


def test_compute_approval_does_not_approve_private_upload(tmp_path):
    config = dict(protocol(), authorization="approved", upload_authorization="pending")
    output = tmp_path / config["output"]
    with pytest.raises(ValueError, match="private source upload"):
        check_dispatch(tmp_path, output, config)
    assert not output.exists()


def test_fixed_output_and_no_retry(tmp_path):
    config = dict(protocol(), authorization="approved", upload_authorization="approved")
    output = tmp_path / config["output"]
    check_dispatch(tmp_path, output, config)
    with pytest.raises(ValueError, match="location"):
        check_dispatch(tmp_path, tmp_path / "other", config)
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot be reused"):
        check_dispatch(tmp_path, output, config)


@pytest.mark.parametrize(
    "broken",
    [
        None,
        "installed_sources",
        "snapshot_sources",
        "packages",
        "exit_code",
        "missing",
        "skipped",
        "failure",
        "duplicate",
    ],
)
def test_judge_requires_every_case_and_installed_metadata(broken):
    config = protocol()
    sources = sources_for(config)
    result = dict(
        exit_code=0,
        installed_sources={k: v for k, v in sources.items() if k.startswith("src/openboost/")},
        snapshot_sources=sources,
        packages={p.split("==")[0]: p.split("==")[1] for p in config["packages"]},
    )
    root = ET.Element("testsuite")
    for node in config["expected_cases"]:
        module, name = node.split("::")
        ET.SubElement(root, "testcase", classname=module[:-3].replace("/", "."), name=name)
    if broken in ("missing", "skipped", "failure", "duplicate"):
        if broken == "missing":
            root.remove(root[0])
        elif broken == "duplicate":
            root.append(ET.fromstring(ET.tostring(root[0])))
        else:
            ET.SubElement(root[0], broken)
    elif broken is not None:
        result.pop(broken)
    assert judge_run(result, ET.tostring(root), config, sources)["passed"] == (broken is None)
