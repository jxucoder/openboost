"""The device preflight judge must reject incomplete or skipped real case matrices."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from benchmarks.v1.cuda_aggregation_preflight import judge_junit


def result(names, tag=None):
    root = ET.Element("testsuite")
    for name in names:
        case = ET.SubElement(root, "testcase", classname="tests.v1.example", name=name)
        if tag:
            ET.SubElement(case, tag)
    return ET.tostring(root)


@pytest.mark.parametrize("tag", ["skipped", "failure", "error"])
def test_nonpass_case_is_not_device_acceptance(tag):
    assert not judge_junit(result(["test_a"], tag), ["tests/v1/example.py::test_a"])["passed"]


@pytest.mark.parametrize("names", [[], ["test_a", "test_a"], ["test_b"]])
def test_exact_case_set_required(names):
    assert not judge_junit(result(names), ["tests/v1/example.py::test_a"])["passed"]


def test_valid_case_and_invalid_report():
    expected = ["tests/v1/example.py::test_a"]
    assert judge_junit(result(["test_a"]), expected)["passed"]
    assert not judge_junit("", expected)["passed"]
    assert not judge_junit(result(["test_a"]), expected * 2)["passed"]


def test_preregistered_manifest_is_run_two_and_cases_are_unique():
    protocol = json.loads(Path("v1-sprints/078-aggregation-run2.json").read_text())
    assert protocol["budget"] == dict(
        run=2, allowed_runs=2, gpu="T4", function_seconds=900, test_seconds=600, retries=0
    )
    assert len(protocol["expected_cases"]) == len(set(protocol["expected_cases"])) == 33
    assert protocol["limits"] == dict(rows=8192, features=32, bins=32, pool_bytes=16 * 1024**2)
