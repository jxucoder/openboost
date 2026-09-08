"""Local freeze/allowance/integrity checks; never run or simulate CUDA kernels."""

import hashlib
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
from benchmarks.v1.cuda_splits_preflight import PROTOCOL

REPO = Path(__file__).resolve().parents[2]


def protocol():
    return json.loads((REPO / PROTOCOL).read_text())


def sources_for(config):
    if config["authorization"] == "consumed":
        # A completed freeze describes its recorded execution, not later README/code.
        return json.loads((REPO / config["output"] / "manifest.json").read_text())["sources"]
    return snapshot_hashes(REPO, snapshot_paths(REPO, PROTOCOL, config))


def test_exact_frozen_sources_and_new_run_bounds():
    config = protocol()
    assert config["authorization"] in ("pending", "approved", "consumed")
    assert config["budget"] == dict(
        run=3,
        previous_runs_consumed=2,
        additional_runs=1,
        gpu="T4",
        function_seconds=900,
        test_seconds=600,
        retries=0,
    )
    assert config["limits"] == dict(rows=8192, features=32, bins=32, pool_bytes=16 * 1024**2)
    assert config["float32"] == dict(rtol=1e-4, atol=1e-5)
    sources = sources_for(config)
    assert len(sources) == 39
    check_frozen_sources(PROTOCOL, config, sources)
    sources["src/openboost/device.py"] = "changed"
    with pytest.raises(ValueError, match="source freeze"):
        check_frozen_sources(PROTOCOL, config, sources)


def test_frozen_cases_match_collection_or_completed_evidence_and_keep_regressions():
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
    assert len(nodes) == len(set(nodes)) == 88
    assert nodes == config["expected_cases"]
    prior = json.loads((REPO / "v1-sprints/078-aggregation-run2.json").read_text())
    assert set(prior["expected_cases"]) <= set(nodes)
    assert config["packages"] == prior["packages"]


def test_pending_or_consumed_allowance_rejects_without_output(tmp_path):
    config = protocol()
    output = tmp_path / config["output"]
    for state in ("pending", "consumed"):
        with pytest.raises(ValueError, match="allowance"):
            check_dispatch(tmp_path, output, dict(config, authorization=state))
        assert not output.exists()


def test_single_output_is_required_and_cannot_be_reused(tmp_path):
    config = dict(protocol(), authorization="approved")
    output = tmp_path / config["output"]
    check_dispatch(tmp_path, output, config)
    with pytest.raises(ValueError, match="location"):
        check_dispatch(tmp_path, tmp_path / "elsewhere", config)
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot be reused"):
        check_dispatch(tmp_path, output, config)


@pytest.mark.parametrize(
    "broken", [None, "installed_sources", "snapshot_sources", "packages", "exit_code"]
)
def test_judge_requires_installed_sources_snapshots_versions_and_exit(broken):
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
    if broken is not None:
        result.pop(broken)
    assert judge_run(result, ET.tostring(root), config, sources)["passed"] == (broken is None)


def test_completed_run_artifacts_and_original_protocol_are_intact():
    config = protocol()
    assert config["authorization"] == "consumed"
    output = REPO / config["output"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["protocol"] == dict(config, authorization="approved")
    assert manifest["status"] == "pass" and manifest["dirty"] is False
    assert manifest["revision"] == "9ce790ef766b8572cad22b96f127d858b4a2e766"
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    observed = judge_run(
        manifest["result"], (output / "junit.xml").read_text(), config, manifest["sources"]
    )
    assert observed == json.loads((output / "verdict.json").read_text())
    assert observed["passed"]
