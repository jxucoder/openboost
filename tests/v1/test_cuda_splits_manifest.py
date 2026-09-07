"""Local freeze/allowance/integrity checks; never run or simulate CUDA kernels."""

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
    sources = snapshot_hashes(REPO, snapshot_paths(REPO, PROTOCOL, config))
    assert len(sources) == 39
    check_frozen_sources(PROTOCOL, config, sources)
    sources["src/openboost/device.py"] = "changed"
    with pytest.raises(ValueError, match="source freeze"):
        check_frozen_sources(PROTOCOL, config, sources)


def test_collection_matches_all_frozen_cases_and_keeps_prior_regressions():
    config = protocol()
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
    sources = snapshot_hashes(REPO, snapshot_paths(REPO, PROTOCOL, config))
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
