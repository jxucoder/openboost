"""Run-7 freeze and permissions; diagnostic cases cannot erase conformance failures."""

import hashlib
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
    judge_run,
    snapshot_hashes,
    snapshot_paths,
)

ROOT = Path(__file__).resolve().parents[2]


def config():
    return json.loads((ROOT / PROTOCOL).read_text())


def source_hashes(protocol):
    if protocol["authorization"] == "consumed":
        return json.loads((ROOT / protocol["output"] / "manifest.json").read_text())["sources"]
    return snapshot_hashes(ROOT, snapshot_paths(ROOT, PROTOCOL, protocol))


def test_exact_source_freeze_preserves_production_and_all_old_verifiers():
    p = config()
    sources = source_hashes(p)
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
    if p["authorization"] == "consumed":
        root = ET.parse(ROOT / p["output"] / "junit.xml").getroot()
        cases = [
            c.attrib["classname"].replace(".", "/") + ".py::" + c.attrib["name"]
            for c in root.iter("testcase")
        ]
    else:
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


def test_completed_run_preserves_original_failures_and_all_raw_hashes():
    p = config()
    output = ROOT / p["output"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert p["authorization"] == "consumed"
    assert manifest["protocol"] == dict(p, authorization="approved")
    assert manifest["revision"] == "80740f27c07c9d676531edf2467bd82a7ef97da6"
    assert manifest["dirty"] is False and manifest["status"] == "fail"
    assert manifest["result"]["exit_code"] == 1
    assert len(manifest["artifacts"]) == 81
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    verdict = judge_run(
        manifest["result"], (output / "junit.xml").read_text(), p, manifest["sources"]
    )
    assert verdict == json.loads((output / "verdict.json").read_text())
    assert verdict["passed"] is False
    assert all(
        verdict[k]
        for k in (
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
        name.split("::")[0][:-3].replace("/", ".") + "::" + name.split("::")[1]
        for name in p["diagnostics"]["expected_unresolved_cases"]
    ]
    assert len(verdict["cases"]) == 385
    assert sum(c["status"] == "pass" for c in verdict["cases"]) == 383
    old = json.loads((ROOT / "benchmarks/v1/evidence/cuda-normal-090/verdict.json").read_text())
    assert verdict["cases"][:383] == old["cases"]
    assert all(c["status"] == "pass" for c in verdict["cases"][-2:])


@pytest.mark.parametrize("order,steps", [("forward", 1), ("reverse", 2)])
def test_actual_failing_traces_preserve_ownership_and_false_improvement(order, steps):
    import numpy as np
    from benchmarks.v1.normal_acceptance_trace import analyze, unpack

    from .reference.device_normal import geometry

    output = ROOT / config()["output"]
    trace = json.loads((output / "normal/acceptance" / f"{order}.json").read_text())
    assert trace["ownership_checks_complete"] is True and trace["final_live_bytes"] == 0
    assert trace["conformance"]["status"] == "failure"
    assert trace["conformance"]["kind"] == "known_coefficient_mismatch"
    assert (
        trace["original_test_source_sha256"]
        == source_hashes(config())["tests/v1/test_device_normal_runtime_cuda.py"]
    )
    report = analyze(trace)
    stored_analysis = json.loads((output / f"analysis-{order}.json").read_text())
    assert {key: stored_analysis[key] for key in report} == json.loads(json.dumps(report))
    assert stored_analysis["trace_sha256"] == hashlib.sha256(
        (output / "normal/acceptance" / f"{order}.json").read_bytes()
    ).hexdigest()
    assert stored_analysis["analysis_sources"] == {
        name: source_hashes(config())[name] for name in stored_analysis["analysis_sources"]
    }
    data = trace["inputs"]["training"]
    target, offset, weight = (unpack(data[k]) for k in ("target", "offset", "weight"))
    for observed in trace["steps"]:
        _, gradient, fisher = geometry(
            unpack(observed["before"]["training_raw"]), target[:, 0], offset, weight
        )
        np.testing.assert_array_equal(unpack(observed["gradient"]), gradient.astype(np.float32))
        np.testing.assert_array_equal(unpack(observed["fisher"]), fisher.astype(np.float32))
    assert len(trace["steps"]) == steps
    step, results = trace["steps"][-1], report["steps"][-1]
    assert step["round"] == 0 and step["channels"] == [0]
    assert [t["coefficient"] for t in step["trials"]] == [8, 4]
    assert [t["accepted"] for t in step["trials"]] == [False, True]
    assert results["stored_field_total"] == [0, 6]
    assert results["measured_total_minus_stored_field_total"] == [4.470348358154297e-08, 0]
    before = unpack(step["before"]["training_raw"])
    after = unpack(step["trials"][-1]["proposal"]["training_raw"])
    np.testing.assert_array_equal(after[:, 0], np.nextafter(before[:, 0], np.float32(-np.inf)))
    np.testing.assert_array_equal(after[:, 1], before[:, 1])
    result = results["trials"][-1]["comparison"]
    assert result["training"]["measured_full_loss_difference"] == -8.881784197001252e-16
    assert result["training"]["original_row_float64_difference"] == -4.440892098500626e-16
    assert result["training"]["high_precision"]["signs"] == [1, 1]
    assert result["validation"]["high_precision"]["signs"] == [-1, -1]
    assert all(
        t["comparison"][name]["high_precision"]["estimates_agree"]
        for s in report["steps"]
        for t in s["trials"]
        for name in ("training", "validation")
    )
    assert step["trials"][-1]["resolved"]["version"] == 1
    assert step["trials"][-1]["resolved"]["best_n_terms"] == 1


def test_nineteen_run7_models_preserve_inference_and_original_model_bytes():
    import numpy as np

    from openboost.artifacts import Model
    from openboost.data import NumericData

    output = ROOT / config()["output"]
    old = ROOT / "benchmarks/v1/evidence/cuda-normal-090"
    manifest = json.loads((output / "manifest.json").read_text())
    paths = sorted((output / "normal").glob("*/model.json"))
    assert len(paths) == 19
    for path in paths:
        assert path.read_bytes() == (old / path.relative_to(output)).read_bytes()
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
