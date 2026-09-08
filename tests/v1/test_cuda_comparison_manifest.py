"""Separate run-8 cohort accounting and failure-closed dispatch; CPU only."""

import copy
import xml.etree.ElementTree as ET

import pytest
from benchmarks.v1.cuda_comparison_preflight import judge_cohorts


def junit(nodes, failures=None):
    root = ET.Element("testsuite")
    for node in nodes:
        path, name = node.split("::")
        case = ET.SubElement(root, "testcase", classname=path[:-3].replace("/", "."), name=name)
        if node in (failures or {}):
            kind, content = failures[node]
            ET.SubElement(case, kind, message=content).text = (
                ">   " + content + "\n/snapshot/" + path + ":10: AssertionError"
            )
    return ET.tostring(root, encoding="unicode")


def specimen():
    old = ["tests/old.py::known", "tests/old.py::good"]
    new = ["tests/new.py::first", "tests/new.py::second"]
    protocol = dict(
        cohorts={
            "historical": dict(
                expected_cases=old, expected_failures={old[0]: "assert coefficients"}
            ),
            "revised": dict(expected_cases=new),
        },
        packages=["numpy==2.3.5"],
        frozen_sources={"src/openboost/a.py": "a"},
        install_projects=[dict(distribution="extension", version="1", source_root="extension/src")],
        normal_cpu_environment=True,
        retained_artifacts=["revised/a.json"],
    )
    sources = {"src/openboost/a.py": "a", "extension/src/a.py": "b"}
    result = dict(
        installed_sources={"src/openboost/a.py": "a"},
        snapshot_sources=sources,
        installed_extensions={"extension": dict(version="1", sources={"extension/src/a.py": "b"})},
        packages={"numpy": "2.3.5"},
        cpu_environment={"passed": True},
        retained_artifact_hashes={"revised/a.json": "c"},
        cohorts={
            "historical": dict(exit_code=1),
            "revised": dict(exit_code=0),
        },
    )
    xmls = dict(
        historical=junit(old, {old[0]: ("failure", "assert coefficients")}), revised=junit(new)
    )
    return protocol, sources, result, xmls


def test_expected_disagreement_never_turns_the_historical_verdict_green():
    protocol, sources, result, xmls = specimen()
    verdict = judge_cohorts(result, xmls, protocol, sources)
    assert verdict["passed"] is True
    assert verdict["historical"]["passed"] is False
    assert verdict["historical"]["expected_disagreements_match"] is True
    assert verdict["revised"]["passed"] is True


@pytest.mark.parametrize("cohort", ["historical", "revised"])
@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "extra", "skip", "error", "invalid", "exit"]
)
def test_incomplete_or_unexpected_results_fail(cohort, fault):
    protocol, sources, result, xmls = specimen()
    root = ET.fromstring(xmls[cohort])
    if fault == "missing":
        root.remove(root[-1])
    elif fault == "duplicate":
        root.append(copy.deepcopy(root[-1]))
    elif fault == "extra":
        ET.SubElement(root, "testcase", classname="tests.extra", name="new")
    elif fault in ("skip", "error"):
        ET.SubElement(root[-1], "skipped" if fault == "skip" else "error")
    elif fault == "exit":
        result["cohorts"][cohort]["exit_code"] = "timeout"
    xmls[cohort] = "<broken" if fault == "invalid" else ET.tostring(root, encoding="unicode")
    assert judge_cohorts(result, xmls, protocol, sources)["passed"] is False


@pytest.mark.parametrize(
    "fault", ["unexpected_pass", "wrong_assertion", "wrong_type", "extra_failure"]
)
def test_historical_disagreements_must_match_frozen_assertion(fault):
    protocol, sources, result, xmls = specimen()
    root = ET.fromstring(xmls["historical"])
    if fault == "unexpected_pass":
        root[0].remove(root[0][0])
    elif fault == "wrong_assertion":
        root[0][0].text = root[0][0].attrib["message"] = "assert topology"
    elif fault == "wrong_type":
        root[0][0].attrib["type"] = "RuntimeError"
    else:
        ET.SubElement(root[1], "failure", type="AssertionError").text = "assert other"
    xmls["historical"] = ET.tostring(root, encoding="unicode")
    assert judge_cohorts(result, xmls, protocol, sources)["passed"] is False


@pytest.mark.parametrize(
    "field",
    [
        "installed_sources",
        "snapshot_sources",
        "installed_extensions",
        "packages",
        "cpu_environment",
        "retained_artifact_hashes",
    ],
)
def test_missing_provenance_or_artifacts_fail(field):
    protocol, sources, result, xmls = specimen()
    result[field] = {}
    assert judge_cohorts(result, xmls, protocol, sources)["passed"] is False


def test_an_assertion_appearing_in_context_is_not_the_failing_assertion():
    protocol, sources, result, xmls = specimen()
    root = ET.fromstring(xmls["historical"])
    root[0][
        0
    ].text = "    assert coefficients\n>   assert wrong\n/snapshot/tests/old.py:10: AssertionError"
    xmls["historical"] = ET.tostring(root, encoding="unicode")
    assert judge_cohorts(result, xmls, protocol, sources)["passed"] is False


def test_actual_archived_pytest_xml_recognizes_only_the_original_two_failures():
    import json
    from pathlib import Path

    from benchmarks.v1.cuda_comparison_preflight import judge_history

    root = Path(__file__).resolve().parents[2]
    previous = json.loads((root / "v1-sprints/091-acceptance-run7.json").read_text())
    cohort = dict(
        expected_cases=previous["expected_cases"],
        expected_failures={
            node: 'assert tuple(coefficients) == tuple(a[0] for a in ref["attempts"])'
            for node in previous["diagnostics"]["expected_unresolved_cases"]
        },
    )
    verdict = judge_history((root / previous["output"] / "junit.xml").read_text(), cohort, 1)
    assert verdict["passed"] is False
    assert verdict["expected_disagreements_match"] is True


def configuration():
    import json
    from pathlib import Path

    from benchmarks.v1.cuda_comparison_preflight import PROTOCOL

    root = Path(__file__).resolve().parents[2]
    return root, json.loads((root / PROTOCOL).read_text())


def test_freeze_preserves_originals_and_exact_isolated_collection():
    import hashlib
    import json

    from benchmarks.v1.cuda_aggregation_preflight import (
        check_frozen_sources,
        snapshot_hashes,
        snapshot_paths,
    )
    from benchmarks.v1.cuda_comparison_preflight import PROTOCOL

    root, protocol = configuration()
    sources = (
        json.loads((root / protocol["output"] / "manifest.json").read_text())["sources"]
        if protocol["authorization"] == "consumed"
        else snapshot_hashes(root, snapshot_paths(root, PROTOCOL, protocol))
    )
    check_frozen_sources(PROTOCOL, protocol, sources)
    assert len(sources) == protocol["upload_file_count"]
    previous = json.loads((root / "v1-sprints/091-acceptance-run7.json").read_text())
    for name, digest in previous["frozen_sources"].items():
        if name.startswith("tests/"):
            assert sources[name] == digest
    collected = json.loads((root / "v1-sprints/092-isolated-collection.json").read_text())
    assert {p: h for p, h in collected["snapshot_sources"].items() if p != PROTOCOL} == {
        p: h for p, h in sources.items() if p != PROTOCOL
    }
    assert collected["installed_sources"] == {
        p: h for p, h in sources.items() if p.startswith("src/openboost/")
    }
    for name, cohort in protocol["cohorts"].items():
        assert collected["cohorts"][name]["cases"] == cohort["expected_cases"]
        assert collected["cohorts"][name]["executed"] == 0
    bindings_file = root / "v1-sprints/092-collected-case-bindings.json"
    assert protocol["bindings_sha256"] == hashlib.sha256(bindings_file.read_bytes()).hexdigest()
    bindings = json.loads(bindings_file.read_text())
    assert protocol["cohorts"]["revised"]["expected_cases"][:383] == [
        c["revised_node"] for c in bindings["cases"]
    ]
    assert len(protocol["cohorts"]["historical"]["expected_cases"]) == 385
    assert len(protocol["cohorts"]["historical"]["expected_failures"]) == 26
    assert len(protocol["cohorts"]["revised"]["expected_cases"]) == 529
    assert protocol["packages"] == previous["packages"]
    assert protocol["limits"] == previous["limits"]
    assert protocol["normal_float32"] == previous["normal_float32"]
    assert protocol["float32"] == previous["float32"]
    assert len(set(protocol["retained_artifacts"])) == len(protocol["retained_artifacts"]) == 409
    assert protocol["retained_artifact_bytes"] == 32 * 1024**2
    assert protocol["budget"] == dict(
        run=8,
        previous_runs_consumed=7,
        additional_runs=1,
        gpu="T4",
        function_seconds=900,
        test_seconds=600,
        retries=0,
    )
    assert protocol["resources"] == dict(
        cpu=2, memory_mib=8192, gpu="T4", timeout_seconds=900, retries=0, max_containers=1
    )
    assert not any(
        p.startswith((".git/", ".codex/", ".agents/")) or "sealed" in p.lower() for p in sources
    )
    sources["src/openboost/device_normal.py"] = "changed"
    with pytest.raises(ValueError, match="source freeze"):
        check_frozen_sources(PROTOCOL, protocol, sources)


def test_no_allowance_reuse_and_pending_dispatch_stops_before_modal(monkeypatch, tmp_path):
    import sys

    from benchmarks.v1 import cuda_comparison_preflight as harness
    from benchmarks.v1.cuda_aggregation_preflight import check_dispatch

    _, protocol = configuration()
    output = tmp_path / protocol["output"]
    for compute, upload in [
        ("pending", "pending"),
        ("approved", "pending"),
        ("consumed", "approved"),
    ]:
        with pytest.raises(ValueError, match="allowance"):
            check_dispatch(
                tmp_path, output, dict(protocol, authorization=compute, upload_authorization=upload)
            )
    approved = dict(protocol, authorization="approved", upload_authorization="approved")
    check_dispatch(tmp_path, output, approved)
    with pytest.raises(ValueError, match="location"):
        check_dispatch(tmp_path, tmp_path / "wrong", approved)
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot be reused"):
        check_dispatch(tmp_path, output, approved)
    monkeypatch.setitem(sys.modules, "modal", None)
    # A pending protocol must stop before import or output creation.
    monkeypatch.setattr(harness.json, "loads", lambda _: dict(protocol, authorization="pending"))
    with pytest.raises(ValueError, match="allowance"):
        harness.main(tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


def test_authorized_dispatch_still_rejects_dirty_or_changed_sources(monkeypatch, tmp_path):
    from benchmarks.v1 import cuda_comparison_preflight as harness

    root, protocol = configuration()
    protocol = dict(
        protocol,
        authorization="approved",
        upload_authorization="approved",
        output=str(tmp_path / "unused-run"),
    )
    monkeypatch.setattr(harness.json, "loads", lambda _: protocol)
    monkeypatch.setattr(harness.subprocess, "check_output", lambda *a, **kw: b"dirty")
    with pytest.raises(ValueError, match="clean source"):
        harness.main(root / protocol["output"])
    monkeypatch.setattr(harness.subprocess, "check_output", lambda *a, **kw: b"")
    monkeypatch.setattr(harness, "snapshot_hashes", lambda *a: {"changed.py": "wrong"})
    with pytest.raises(ValueError, match="source freeze"):
        harness.main(root / protocol["output"])
    assert not (root / protocol["output"]).exists()


@pytest.mark.parametrize(
    "field", ["budget", "resources", "image", "cohorts", "retained_artifact_bytes"]
)
def test_executable_budget_cannot_drift_from_request(field):
    from benchmarks.v1.cuda_comparison_preflight import check_protocol

    _, protocol = configuration()
    check_protocol(protocol)
    protocol[field] = 0 if field == "retained_artifact_bytes" else {}
    with pytest.raises(ValueError, match="changed"):
        check_protocol(protocol)


@pytest.mark.parametrize("status", ["approved", "consumed"])
def test_regeneration_cannot_reset_authorized_or_consumed_allowance(monkeypatch, status):
    from benchmarks.v1 import freeze_comparison_run8 as freeze

    _, protocol = configuration()
    monkeypatch.setattr(freeze.json, "loads", lambda _: dict(protocol, authorization=status))
    with pytest.raises(ValueError, match="pending freeze"):
        freeze.build_protocol()
