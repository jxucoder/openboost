"""Exact benchmark inputs and recoverable real CPU fits; never CUDA emulation."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from benchmarks.v1.performance_checkpoint import CONFIG, workload
from benchmarks.v1.performance_evidence import (
    array_record,
    identity,
    input_record,
    load_inputs,
    run,
    verify_report,
    write_json,
)


@pytest.fixture
def inputs(monkeypatch):
    for key, value in dict(rounds=2, bins=4, max_depth=1).items():
        monkeypatch.setitem(CONFIG, key, value)
    return input_record(*workload(32, "squared"), "squared")


@pytest.mark.parametrize("recipe", ["squared", "normal"])
def test_exact_input_bytes_survive_json_without_generator(recipe):
    original = workload(33, recipe)
    record = input_record(*original, recipe)
    restored = load_inputs(json.loads(json.dumps(record)), record["sha256"])
    for expected, actual in zip(original, restored, strict=True):
        assert actual.identity == expected.identity
        assert actual.data.identity == expected.data.identity
        for left, right in (
            (expected.data.values, actual.data.values),
            (expected.row_ids, actual.row_ids),
            (expected.target, actual.target),
            (expected.weight, actual.weight),
            (expected.offset, actual.offset),
        ):
            assert left.dtype.str == right.dtype.str and left.tobytes() == right.tobytes()
            assert not right.flags.writeable


def test_changed_target_cannot_reuse_original_input_binding(inputs, tmp_path):
    report = run(inputs, "cpu", tmp_path / "result.json", repetitions=1)
    train, validation = load_inputs(inputs, inputs["sha256"])
    from dataclasses import replace

    changed = input_record(train, replace(validation, target=validation.target + 1), "squared")
    # The unchanged model has identical predictions; its target/quality contract changed.
    with pytest.raises(ValueError, match="input"):
        verify_report(report, changed, inputs["sha256"])


@pytest.mark.parametrize("fault", ["payload", "identity", "checksum", "shape"])
def test_input_corruption_rejected(inputs, fault):
    broken = copy.deepcopy(inputs)
    if fault == "identity":
        broken["train"]["identity"] = "changed"
    elif fault == "checksum":
        broken["sha256"] = "changed"
    else:
        field = broken["train"]["arrays"]["target"]
        field["data" if fault == "payload" else "shape"] = (
            "broken" if fault == "payload" else [99, 1]
        )
    with pytest.raises(ValueError, match="input"):
        load_inputs(broken, inputs["sha256"])


def test_valid_outer_hash_cannot_hide_wrong_array_or_problem_identity(inputs):
    broken = copy.deepcopy(inputs)
    broken["train"]["arrays"]["target"]["sha256"] = "changed"
    broken["sha256"] = identity({k: v for k, v in broken.items() if k != "sha256"})
    with pytest.raises(ValueError, match="array bytes/checksum"):
        load_inputs(broken, broken["sha256"])
    train, _ = load_inputs(inputs, inputs["sha256"])
    broken["train"]["arrays"]["target"] = array_record(train.target + 1)
    broken["sha256"] = identity({k: v for k, v in broken.items() if k != "sha256"})
    with pytest.raises(ValueError, match="problem identity"):
        load_inputs(broken, broken["sha256"])


def test_complete_fit_records_replay_and_detect_wrong_scores(inputs, tmp_path):
    report = run(inputs, "cpu", tmp_path / "result.json", repetitions=2)
    result = verify_report(report, inputs, inputs["sha256"])
    assert result == dict(complete=True, verified_fits=2)
    assert json.loads((tmp_path / "result.json").read_text()) == report
    bad = copy.deepcopy(report)
    bad["fits"][0]["quality"]["half_mse"] += 1
    with pytest.raises(ValueError, match="quality"):
        verify_report(bad, inputs, inputs["sha256"])
    bad = copy.deepcopy(report)
    bad["status"] = "timeout"
    assert verify_report(bad, inputs, inputs["sha256"]) == dict(complete=False, verified_fits=2)
    bad.update(expected_repetitions=0, fits=[], status="complete")
    with pytest.raises(ValueError, match="contract"):
        verify_report(bad, inputs, inputs["sha256"])


def test_real_child_interruption_keeps_first_complete_fit(inputs, tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "result.json"
    write_json(source, inputs)
    script = tmp_path / "child.py"
    script.write_text(
        "import json, pathlib, sys, time\n"
        "import benchmarks.v1.performance_evidence as evidence\n"
        "original = evidence.fit\n"
        "calls = 0\n"
        "def delayed(*args, **kwargs):\n"
        "    global calls\n"
        "    calls += 1\n"
        "    if calls == 2: time.sleep(20)\n"
        "    return original(*args, **kwargs)\n"
        "evidence.fit = delayed\n"
        "evidence.run(json.loads(pathlib.Path(sys.argv[1]).read_text()), 'cpu', "
        "pathlib.Path(sys.argv[2]), repetitions=2)\n"
    )
    import os

    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    child = subprocess.Popen(
        [sys.executable, str(script), str(source), str(output)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        from time import monotonic, sleep

        deadline = monotonic() + 8
        while monotonic() < deadline:
            if output.exists() and len(json.loads(output.read_text()).get("fits", [])) == 1:
                break
            if child.poll() is not None:
                pytest.fail(child.communicate()[1].decode())
            sleep(0.02)
        else:
            pytest.fail("first fit was not checkpointed")
        child.kill()
        child.communicate(timeout=3)
        report = json.loads(output.read_text())
        assert report["status"] == "running"
        assert verify_report(report, inputs, inputs["sha256"]) == dict(
            complete=False, verified_fits=1
        )
        assert report["fits"][0]["model"] and report["fits"][0]["validation_raw"]
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=3)
