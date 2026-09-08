"""Standalone D1/D2 numerical gates, failure controls and closure integrity."""

import ast
import json

import pytest
from benchmarks.v1.authoring.development import collect
from benchmarks.v1.authoring.export import REFERENCE_MODULES, export
from benchmarks.v1.authoring.export import ROOT as REPO
from benchmarks.v1.authoring.judge import judge, read_json

from tests.v1.test_public_extensions import ROOT, load


@pytest.fixture
def prepared(tmp_path):
    bundle, observations = tmp_path / "bundle", tmp_path / "observations"
    export(bundle)
    for task, directory, module in (
        ("D1", "expectile", "ob_expectile"),
        ("D2", "cohort_splits", "ob_cohort_splits"),
    ):
        plugin = load(ROOT / f"{directory}/src/{module}/__init__.py", f"author_test_{module}")
        collect(bundle / "inputs.json", observations, task, plugin)
    return bundle, observations


@pytest.mark.parametrize("task,count", [("D1", 2), ("D2", 9)])
def test_independent_task_judging(prepared, task, count):
    bundle, observations = prepared
    result = judge(bundle, observations, task)
    assert result["passed"] and result["cases"] == count
    assert not result["dispatch_ready"] and result["attempts"] == []


@pytest.mark.parametrize(
    "failure", ["gradient", "round", "information", "nonfinite", "shape", "boolean"]
)
def test_wrong_observations_cannot_pass(prepared, failure):
    bundle, observations = prepared
    task = "D2" if failure == "information" else "D1"
    path = observations / f"{task}.json"
    data = json.loads(path.read_text())
    case = next(iter(data.values()))
    if failure == "gradient":
        case["trace"][0]["gradient"][0] += 1
    elif failure == "round":
        case["trace"].pop()
    elif failure == "information":
        data["d2-depthwise-zero-weight"]["probe"]["information"][0] = [0, 0]
    elif failure == "nonfinite":
        case["raw"][0] = float("nan")
    elif failure == "shape":
        case["raw"] = [case["raw"]]
    else:
        case["base"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        judge(bundle, observations, task)


@pytest.mark.parametrize("filename", ["inputs.json", "expected.json", "judge.py"])
def test_changed_evaluator_input_fails(prepared, filename):
    bundle, observations = prepared
    path = bundle / filename
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="verifier input changed"):
        judge(bundle, observations, "D1")


def test_wrong_model_cannot_hide_behind_correct_observations(prepared):
    bundle, observations = prepared
    path = observations / "d1-0.8.model.json"
    model = json.loads(path.read_text())
    model["base"][0] += 1
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="saved_model"):
        judge(bundle, observations, "D1")


def test_missing_model_fails(prepared):
    bundle, observations = prepared
    (observations / "d2-depthwise-ordinary.model.json").unlink()
    with pytest.raises(ValueError, match="regular file"):
        judge(bundle, observations, "D2")


def test_model_symlink_is_rejected(prepared, tmp_path):
    bundle, observations = prepared
    path = observations / "d1-0.8.model.json"
    other = tmp_path / "outside-model.json"
    path.rename(other)
    path.symlink_to(other)
    with pytest.raises(ValueError, match="regular file"):
        judge(bundle, observations, "D1")


def test_incomplete_closure_fails(prepared):
    bundle, observations = prepared
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["runtime_files"]["expected.json"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete verifier closure"):
        judge(bundle, observations, "D1")


def test_extra_model_output_is_not_ignored(prepared):
    bundle, observations = prepared
    path = observations / "d1-0.8.model.json"
    model = json.loads(path.read_text())
    model["base"].append(123)
    for term in model["terms"]:
        term["mapping"][0].append(0)
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="output shape"):
        judge(bundle, observations, "D1")


def test_duplicate_observation_keys_fail(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"base": 1, "base": 2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_json(path)


def test_oracle_provenance_covers_relative_import_closure():
    for name in REFERENCE_MODULES:
        source = REPO / f"tests/v1/reference/{name}.py"
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.level == 1 and node.module in REFERENCE_MODULES
