"""Run-12 source closure, artifact retention and fail-closed local preflight controls."""

import base64
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from benchmarks.v1 import freeze_glm_run12 as freezer
from benchmarks.v1.cuda_aggregation_preflight import (
    check_dispatch,
    check_frozen_sources,
    judge_run,
    main,
    retain_artifacts,
    snapshot_hashes,
    snapshot_paths,
)
from benchmarks.v1.cuda_glm_preflight import PROTOCOL

from .glm_artifacts import directory, input_snapshot
from .test_device_glm_reference import paired_fixture

ROOT = Path(__file__).resolve().parents[2]


def packet():
    return json.loads((ROOT / PROTOCOL).read_text())


def sources(protocol):
    if protocol["authorization"] != "consumed":
        return snapshot_hashes(ROOT, snapshot_paths(ROOT, PROTOCOL, protocol))
    manifest = json.loads((ROOT / protocol["output"] / "manifest.json").read_text())
    assert manifest["protocol"]["frozen_sources"] == protocol["frozen_sources"]
    result = {
        p: hashlib.sha256(
            subprocess.check_output(["git", "show", manifest["revision"] + ":" + p], cwd=ROOT)
        ).hexdigest()
        for p in manifest["sources"]
    }
    assert result == manifest["sources"]
    return result


def test_exact_freeze_and_installed_collection():
    p = packet()
    s = sources(p)
    check_frozen_sources(PROTOCOL, p, s)
    assert len(s) == p["upload_file_count"] == 85
    report = json.loads((ROOT / "v1-sprints/108-isolated-collection.json").read_text())
    assert report["cohorts"]["candidate"] == dict(
        cases=p["expected_cases"], collected=571, executed=0
    )
    assert len(p["expected_cases"]) == len(set(p["expected_cases"])) == 571
    assert report["installed_sources"] == {
        name: sha for name, sha in s.items() if name.startswith("src/openboost/")
    }
    assert {name: sha for name, sha in report["snapshot_sources"].items() if name != PROTOCOL} == p[
        "frozen_sources"
    ]
    assert p["group_counts"] == dict(scalar=212, normal=167, validation=39, glm=153)
    assert p["budget"] == dict(
        run=12,
        previous_runs_consumed=11,
        additional_runs=1,
        gpu="T4",
        function_seconds=900,
        test_seconds=600,
        retries=0,
    )
    assert p["resources"] == dict(
        cpu=2, memory_mib=8192, gpu="T4", timeout_seconds=900, retries=0, max_containers=1
    )
    assert p["retained_artifacts"] == freezer.artifact_names()
    assert len(p["retained_artifacts"]) == 77


def test_pending_entrypoint_stops_before_hardware(tmp_path, monkeypatch):
    p = packet()
    p.update(authorization="pending", upload_authorization="pending")
    path = tmp_path / "pending.json"
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="GPU allowance is pending"):
        main(ROOT / p["output"], protocol_path=str(path))


@pytest.mark.parametrize("kind", ["upload", "reuse", "destination", "source"])
def test_dispatch_guards(kind, tmp_path):
    p = packet()
    changed = sources(p) if kind == "source" else None
    p.update(authorization="approved", upload_authorization="approved")
    output = tmp_path / p["output"]
    if kind == "source":
        changed["src/openboost/device_glm.py"] = "changed"
        with pytest.raises(ValueError, match="source freeze changed"):
            check_frozen_sources(PROTOCOL, p, changed)
        return
    if kind == "upload":
        p["upload_authorization"] = "pending"
        reason = "source upload allowance is pending"
    elif kind == "reuse":
        output.mkdir(parents=True)
        reason = "cannot be reused"
    else:
        output = tmp_path / "other"
        reason = "location is fixed"
    with pytest.raises(ValueError, match=reason):
        check_dispatch(tmp_path, output, p)


@pytest.mark.parametrize("state", ["approved", "consumed", "attempted"])
def test_freezer_cannot_regenerate_authorized_or_attempted_packet(state, tmp_path, monkeypatch):
    p = packet()
    p.update(authorization="pending", upload_authorization="pending")
    if state == "attempted":
        (tmp_path / p["output"]).mkdir(parents=True)
    else:
        p["authorization"] = state
    path = tmp_path / PROTOCOL
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(p))
    monkeypatch.setattr(freezer, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="regenerated"):
        freezer.build_protocol()


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "case",
        "duplicate",
        "skip",
        "failure",
        "artifact",
        "core",
        "snapshot",
        "package",
        "cpu",
        "exit",
    ],
)
def test_verdict_requires_complete_matrix_identity_and_artifacts(fault):
    p = packet()
    s = {**p["frozen_sources"], PROTOCOL: "test-only-protocol-hash"}
    result = dict(
        exit_code=0,
        installed_sources={k: v for k, v in s.items() if k.startswith("src/openboost/")},
        snapshot_sources=dict(s),
        packages=dict(value.split("==") for value in p["packages"]),
        cpu_environment=dict(passed=True),
        retained_artifact_hashes={
            name: "test-only-artifact-hash" for name in p["retained_artifacts"]
        },
    )
    xml = ET.Element("testsuite")
    for node in p["expected_cases"]:
        file, name = node.split("::")
        ET.SubElement(xml, "testcase", classname=file[:-3].replace("/", "."), name=name)
    if fault == "case":
        xml.remove(xml[0])
    elif fault == "duplicate":
        ET.SubElement(xml, "testcase", **xml[0].attrib)
    elif fault in ("skip", "failure"):
        ET.SubElement(xml[0], "skipped" if fault == "skip" else "failure")
    elif fault == "artifact":
        result["retained_artifact_hashes"].pop(p["retained_artifacts"][0])
    elif fault in ("core", "snapshot", "package"):
        field = dict(core="installed_sources", snapshot="snapshot_sources", package="packages")[
            fault
        ]
        result[field].pop(next(iter(result[field])))
    elif fault == "cpu":
        result["cpu_environment"]["passed"] = False
    elif fault == "exit":
        result["exit_code"] = "timeout"
    assert judge_run(result, ET.tostring(xml), p, s)["passed"] is (fault is None)


@pytest.mark.parametrize("kind", ["comparisons", "recipes"])
def test_artifacts_use_collector_root_and_preserve_explicit_override(kind, tmp_path, monkeypatch):
    env = "OPENBOOST_GLM_" + dict(comparisons="COMPARISON", recipes="RECIPE")[kind] + "_ARTIFACTS"
    monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("OPENBOOST_NORMAL_ARTIFACTS", raising=False)
    assert directory(kind, tmp_path) == tmp_path
    monkeypatch.setenv("OPENBOOST_NORMAL_ARTIFACTS", str(tmp_path / "normal"))
    assert directory(kind, tmp_path) == tmp_path / "normal" / ("glm-" + kind)
    monkeypatch.setenv(env, str(tmp_path / "explicit"))
    assert directory(kind, tmp_path) == tmp_path / "explicit"


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_retained_fixture_bytes_round_trip_including_missing_values(family):
    for p in paired_fixture(family):
        report = json.loads(json.dumps(input_snapshot(p), allow_nan=False))
        originals = dict(
            features=p.data.values,
            row_ids=p.data.row_ids,
            target=p.target,
            offset=p.offset,
            weight=p.weight,
            **p.structure,
        )
        assert report.keys() == originals.keys()
        for name, r in report.items():
            value = np.frombuffer(base64.b64decode(r["data_base64"]), dtype=r["dtype"]).reshape(
                r["shape"]
            )
            assert value.tobytes() == np.ascontiguousarray(originals[name]).tobytes()
            np.testing.assert_array_equal(value, originals[name])


def test_ptx_json_retains_literal_text_and_rejects_undeclared_output(tmp_path):
    p = packet()
    name = "normal/glm-comparisons/binary-ptx.json"
    value = json.dumps(dict(ptx="// literal PTX\nadd.rm.f64 %fd1, %fd2, %fd3;\n"))
    hashes = retain_artifacts(tmp_path, {name: value}, p)
    assert hashes[name] == hashlib.sha256(value.encode()).hexdigest()
    assert json.loads((tmp_path / name).read_text()) == json.loads(value)
    with pytest.raises(ValueError, match="undeclared"):
        retain_artifacts(tmp_path, {"normal/unlisted.json": value}, p)
