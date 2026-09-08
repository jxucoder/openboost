"""Run-6 exact-source and dispatch checks; no device execution in these tests."""

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from benchmarks.v1.cuda_aggregation_preflight import (
    check_dispatch,
    check_frozen_sources,
    judge_run,
    retain_artifacts,
    snapshot_hashes,
    snapshot_paths,
)
from benchmarks.v1.cuda_normal_preflight import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]


def config():
    return json.loads((ROOT / PROTOCOL).read_text())


def source_hashes(protocol):
    if protocol["authorization"] == "consumed":
        return json.loads((ROOT / protocol["output"] / "manifest.json").read_text())["sources"]
    return snapshot_hashes(ROOT, snapshot_paths(ROOT, PROTOCOL, protocol))


def test_source_freeze_and_retained_regressions():
    protocol = config()
    sources = source_hashes(protocol)
    check_frozen_sources(PROTOCOL, protocol, sources)
    assert len(sources) == protocol["upload_file_count"] == 67
    assert sum(p.startswith("src/openboost/") for p in sources) == 28
    prior = json.loads((ROOT / "v1-sprints/089-symmetry-run5.json").read_text())
    assert protocol["packages"] == prior["packages"] + ["uv==0.12.1"]
    assert protocol["limits"] == prior["limits"]
    assert protocol["float32"] == prior["float32"]
    for path, digest in prior["frozen_sources"].items():
        if path.startswith("tests/"):
            assert sources[path] == digest
    assert protocol["budget"] == dict(
        run=6,
        previous_runs_consumed=5,
        additional_runs=1,
        gpu="T4",
        function_seconds=900,
        test_seconds=600,
        retries=0,
    )
    assert protocol["normal_float32"]["rtol"] == 2e-4
    assert protocol["normal_float32"]["loss_rtol"] == 2e-5
    assert protocol["diagnostics"]["near_tie_is_structural_conformance"] is False
    assert protocol["normal_cpu_environment"] is True
    assert protocol["retained_artifact_bytes"] == 2 * 1024**2
    assert len(protocol["retained_artifacts"]) == 76
    assert len(set(protocol["retained_artifacts"])) == 76
    assert not any(p.startswith((".git/", ".codex/", ".agents/")) or "sealed" in p for p in sources)
    sources["src/openboost/device_normal.py"] = "changed"
    with pytest.raises(ValueError, match="source freeze"):
        check_frozen_sources(PROTOCOL, protocol, sources)


def test_exact_case_matrix():
    protocol = config()
    if protocol["authorization"] == "consumed":
        root = ET.parse(ROOT / protocol["output"] / "junit.xml").getroot()
        cases = [
            c.attrib["classname"].replace(".", "/") + ".py::" + c.attrib["name"]
            for c in root.iter("testcase")
        ]
    else:
        collected = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "pytest",
                *protocol["test_files"],
                "--collect-only",
                "-o",
                "addopts=",
                "-q",
            ],
            cwd=ROOT,
            text=True,
        )
        cases = [line for line in collected.splitlines() if line.startswith("tests/")]
    assert cases == protocol["expected_cases"]
    assert len(cases) == len(set(cases)) == 383
    prior = json.loads((ROOT / "v1-sprints/089-symmetry-run5.json").read_text())
    assert cases[:212] == prior["expected_cases"]
    assert sum("normal_extension" in c for c in cases) == 20
    assert sum("normal_ties" in c for c in cases) == 1


def test_both_allowances_output_and_pending_cli(monkeypatch, tmp_path):
    from benchmarks.v1 import cuda_normal_preflight

    protocol = config()
    output = tmp_path / protocol["output"]
    for compute, upload in (
        ("pending", "pending"),
        ("approved", "pending"),
        ("consumed", "approved"),
    ):
        current = dict(protocol, authorization=compute, upload_authorization=upload)
        with pytest.raises(ValueError, match="allowance"):
            check_dispatch(tmp_path, output, current)
    approved = dict(protocol, authorization="approved", upload_authorization="approved")
    check_dispatch(tmp_path, output, approved)
    with pytest.raises(ValueError, match="location"):
        check_dispatch(tmp_path, tmp_path / "different", approved)
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot be reused"):
        check_dispatch(tmp_path, output, approved)
    monkeypatch.setitem(sys.modules, "modal", None)
    monkeypatch.setattr(
        "benchmarks.v1.cuda_aggregation_preflight.json.loads",
        lambda _: dict(protocol, authorization="pending"),
    )
    with pytest.raises(ValueError, match="allowance"):
        cuda_normal_preflight.main(tmp_path / "never-created", protocol_path=PROTOCOL)
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize(
    "broken", [None, "extension_source", "extension_version", "cpu_environment", "artifact", "exit"]
)
def test_judge_rejects_incomplete_installed_or_replay_evidence(broken):
    protocol = config()
    sources = source_hashes(protocol)
    (project,) = protocol["install_projects"]
    root = ET.Element("testsuite")
    for node in protocol["expected_cases"]:
        path, name = node.split("::")
        ET.SubElement(root, "testcase", classname=path[:-3].replace("/", "."), name=name)
    extension = dict(
        version=project["version"],
        sources={p: h for p, h in sources.items() if p.startswith(project["source_root"] + "/")},
    )
    result = dict(
        exit_code=0,
        installed_sources={p: h for p, h in sources.items() if p.startswith("src/openboost/")},
        snapshot_sources=sources,
        packages=dict(p.split("==") for p in protocol["packages"]),
        installed_extensions={project["distribution"]: extension},
        cpu_environment={"passed": True},
        retained_artifact_hashes={p: "a" * 64 for p in protocol["retained_artifacts"]},
    )
    if broken == "extension_source":
        extension["sources"] = {}
    if broken == "extension_version":
        extension["version"] = "wrong"
    if broken == "cpu_environment":
        result["cpu_environment"] = {}
    if broken == "artifact":
        result["retained_artifact_hashes"].popitem()
    if broken == "exit":
        result["exit_code"] = 1
    assert judge_run(result, ET.tostring(root), protocol, sources)["passed"] == (broken is None)


def test_bounded_artifact_retention_and_partial_failure(tmp_path):
    protocol = dict(
        retained_artifacts=["normal/example/model.json", "normal/example/inputs.json"],
        retained_artifact_bytes=64,
    )
    payload = {"normal/example/model.json": "{}\n"}
    assert retain_artifacts(tmp_path, payload, protocol) == {
        "normal/example/model.json": hashlib.sha256(b"{}\n").hexdigest()
    }
    with pytest.raises(ValueError, match="undeclared"):
        retain_artifacts(tmp_path, {"not-declared.json": "{}"}, protocol)
    with pytest.raises(ValueError, match="byte limit"):
        retain_artifacts(tmp_path, {"normal/example/inputs.json": " " * 65}, protocol)
    with pytest.raises(ValueError, match="relative JSON"):
        retain_artifacts(
            tmp_path,
            {"../outside.json": "{}"},
            dict(protocol, retained_artifacts=["../outside.json"]),
        )
    assert not (tmp_path.parent / "outside.json").exists()


def test_completed_run_preserves_failure_verdict_and_all_raw_artifacts():
    protocol = config()
    output = ROOT / protocol["output"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert protocol["authorization"] == "consumed"
    assert manifest["protocol"] == dict(protocol, authorization="approved")
    assert manifest["revision"] == "4143d188b9635749308ef52382a1562928ae2612"
    assert manifest["dirty"] is False and manifest["status"] == "fail"
    assert manifest["result"]["exit_code"] == 1
    assert len(manifest["artifacts"]) == 79
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    verdict = judge_run(
        manifest["result"], (output / "junit.xml").read_text(), protocol, manifest["sources"]
    )
    assert verdict == json.loads((output / "verdict.json").read_text())
    assert verdict["passed"] is False
    assert all(
        verdict[name]
        for name in (
            "installed_sources_match",
            "versions_match",
            "snapshot_sources_match",
            "installed_extensions_match",
            "cpu_environment_built",
            "retained_artifacts_complete",
        )
    )
    failed = [c["case"] for c in verdict["cases"] if c["status"] != "pass"]
    assert failed == [
        f"tests.v1.test_device_normal_runtime_cuda::test_frozen_three_round_transactions[False-8.0-{order}-ordinary-0-conflict-0-None]"
        for order in ("forward", "reverse")
    ]
    assert (
        len(verdict["cases"]) == 383 and sum(c["status"] == "pass" for c in verdict["cases"]) == 381
    )
    assert all(c["status"] == "pass" for c in verdict["cases"][:212])


def test_nineteen_archived_models_match_fresh_cpu_replays():
    from openboost.artifacts import Model
    from openboost.data import NumericData

    output = ROOT / config()["output"]
    manifest = json.loads((output / "manifest.json").read_text())
    paths = sorted((output / "normal").glob("*/model.json"))
    assert len(paths) == 19
    for path in paths:
        inputs = json.loads(path.with_name("inputs.json").read_text())
        replay = json.loads(path.with_name("cpu-replay.json").read_text())
        assert replay["absent"] == ["ob_cohort_splits", "cupy", "numba"]
        assert replay["sources"] == manifest["result"]["installed_sources"]
        assert replay["model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        model = Model.load(path)
        data = NumericData(inputs["values"], inputs["row_ids"], tuple(inputs["feature_names"]))
        prediction = model.predict(data)
        np.testing.assert_array_equal(prediction, replay["raw"])
        np.testing.assert_allclose(prediction, inputs["expected_raw"], rtol=2e-4, atol=2e-5)
        if path.parent.name == "missing-normal":
            assert np.isnan(data.values).any()
        else:
            assert model.terms[0].learner.feature[0] == 0
            assert model.terms[0].learner.threshold[0] == 1
