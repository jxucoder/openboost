"""Constructed optimization judge controls; these are not GPU execution evidence."""

import copy

import pytest
from benchmarks.v1.performance_checkpoint import CONFIG, workload
from benchmarks.v1.performance_evidence import input_record, run
from benchmarks.v1.validation_judge import judge


@pytest.fixture
def pair(tmp_path, monkeypatch):
    for key, value in dict(rounds=2, bins=4, max_depth=1).items():
        monkeypatch.setitem(CONFIG, key, value)
    inputs = input_record(*workload(32, "squared"), "squared")
    base = run(inputs, "cpu", tmp_path / "base.json", repetitions=4)
    base["backend"] = "cuda"
    for f in base["fits"]:
        f["measurement"].update(
            fit_seconds=2.0,
            live_bytes_after_run_close=0,
            final_metrics=dict(live_bytes=0, kernel_launches=1, validation_export_bytes=4),
        )
    candidate = copy.deepcopy(base)
    for f in candidate["fits"]:
        f["measurement"]["fit_seconds"] = 1.0
    return base, candidate, inputs


def evaluate(pair):
    base, candidate, inputs = pair
    sources = base["environment"]["core_sources"]
    return judge(
        base, candidate, inputs, baseline_sources=sources, candidate_sources=sources, limit=0.8
    )


def test_complete_exact_candidate_qualifies(pair):
    result = evaluate(pair)
    assert result["measurement_complete"] and result["quality_passed"] and result["cost_passed"]
    assert result["candidate_over_baseline_warm_ratio"] == 0.5


@pytest.mark.parametrize(
    "fault",
    ["timeout", "missing_repeat", "sources", "quality", "model", "cleanup", "launches", "profile"],
)
def test_incomplete_or_incorrect_candidate_cannot_be_fast_win(pair, fault):
    _, candidate, _ = pair
    fit = candidate["fits"][0]
    if fault == "timeout":
        candidate["status"] = "timeout"
    elif fault == "missing_repeat":
        candidate["fits"].pop()
    elif fault == "sources":
        candidate["environment"]["core_sources"] = {}
    elif fault == "quality":
        fit["quality"]["half_mse"] += 1
    elif fault == "model":
        fit["model_sha256"] = "changed"
    elif fault == "cleanup":
        fit["measurement"]["live_bytes_after_run_close"] = 4
    elif fault == "profile":
        candidate["profile"] = True
    else:
        fit["measurement"]["final_metrics"]["kernel_launches"] += 1
    result = evaluate(pair)
    assert not result["cost_passed"] and result["candidate_over_baseline_warm_ratio"] is None


def test_correct_but_slow_candidate_retains_failed_cost_gate(pair):
    for fit in pair[1]["fits"]:
        fit["measurement"]["fit_seconds"] = 2
    result = evaluate(pair)
    assert result["measurement_complete"] and result["quality_passed"]
    assert not result["cost_passed"] and result["candidate_over_baseline_warm_ratio"] == 1


@pytest.mark.parametrize("fault", [None, "timeout", "sources", "quality"])
def test_cpu_control_is_required_to_complete_and_match_quality(pair, tmp_path, fault):
    base, candidate, inputs = pair
    cpu_inputs = copy.deepcopy(inputs)
    if fault == "quality":
        from benchmarks.v1.performance_evidence import identity

        cpu_inputs["config"]["learning_rate"] = 0.9
        cpu_inputs["sha256"] = identity({k: v for k, v in cpu_inputs.items() if k != "sha256"})
    cpu = run(cpu_inputs, "cpu", tmp_path / "cpu.json", repetitions=2)
    # A real different model with internally valid replay/quality, bound to the
    # original workload for this constructed judge control only.
    cpu.update(config=inputs["config"], input_sha256=inputs["sha256"])
    if fault == "timeout":
        cpu["status"] = "timeout"
    elif fault == "sources":
        cpu["environment"]["core_sources"] = {}
    sources = base["environment"]["core_sources"]
    result = judge(
        base,
        candidate,
        inputs,
        baseline_sources=sources,
        candidate_sources=sources,
        limit=0.8,
        cpu=cpu,
    )
    assert result["cost_passed"] == (fault is None)
    if fault is not None:
        assert result["candidate_over_baseline_warm_ratio"] is None


def test_actual_supervisor_timeout_preserves_child_evidence(tmp_path, monkeypatch):
    import json
    import sys

    from . import test_validation_checkpoint_cuda as supervisor

    script = tmp_path / "benchmarks/v1/validation_checkpoint.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json, pathlib, sys, time\n"
        "p=pathlib.Path(sys.argv[sys.argv.index('--output')+1])\n"
        "p.write_text(json.dumps({'status':'running','fits':[{'completed':True}]}))\n"
        "time.sleep(20)\n"
    )
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(dict(sha256="supervisor-control")))
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    output = tmp_path / "result.json"
    result = supervisor.child(sys.executable, inputs, output, 0.3, tmp_path / "cache", cpu=True)
    assert result["status"] == "timeout" and result["fits"] == [{"completed": True}]
    assert result["child_wall_seconds"] < 5
    assert json.loads(output.read_text()) == result
