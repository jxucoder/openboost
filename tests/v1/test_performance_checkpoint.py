"""104 timing eligibility rejects quality/provenance failures; no CUDA emulation."""

import copy

import numpy as np
import pytest
from benchmarks.v1.performance_checkpoint import CONFIG, quality, run, workload
from benchmarks.v1.performance_judge import judge


@pytest.fixture
def pair(monkeypatch):
    for key, value in dict(rounds=2, bins=4, max_depth=1, repetitions=2, cpu_repetitions=2).items():
        monkeypatch.setitem(CONFIG, key, value)
    cpu = run(32, "squared", "cpu")
    cuda = copy.deepcopy(cpu)
    # Construct judge input only; this is never recorded as a CUDA execution.
    cuda["backend"] = "cuda"
    for row in cuda["measurements"]:
        row["fit_seconds"] /= 2
        row["final_metrics"] = dict(live_bytes=0)
        row["live_bytes_after_run_close"] = 0
    return cpu, cuda


def test_actual_cpu_export_and_complete_judge_control(pair):
    cpu, cuda = pair
    result = judge(cpu, cuda)
    assert result["measurement_complete"] and result["quality_comparable"]
    assert result["cpu_over_cuda_warm_fit_ratio"] == pytest.approx(2)
    assert cpu["replay_exact"] and len(set(cpu["model_identities"])) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "timeout",
        "missing_repeat",
        "profile",
        "inputs",
        "settings",
        "cleanup",
        "rounds",
        "predictions",
        "quality",
        "replay",
    ],
)
def test_fast_invalid_measurement_cannot_receive_speed_ratio(pair, fault):
    cpu, cuda = pair
    if fault == "timeout":
        cuda["status"] = "timeout"
    elif fault == "missing_repeat":
        cuda["measurements"].pop()
    elif fault == "profile":
        cuda["profile"] = True
    elif fault == "inputs":
        cuda["train_identity"] = "other"
    elif fault == "settings":
        cuda["config"]["rounds"] += 1
    elif fault == "cleanup":
        cuda["measurements"][0]["final_metrics"]["live_bytes"] = 4
    elif fault == "rounds":
        cuda["measurements"][0]["state"]["version"] = 1
    elif fault == "predictions":
        cuda["validation_raw"][0][0] = float("nan")
    elif fault == "quality":
        cuda["quality"]["half_mse"] *= 0.5
    else:
        cuda["replay_exact"] = False
    result = judge(cpu, cuda)
    assert not result["quality_comparable"] and result["cpu_over_cuda_warm_fit_ratio"] is None


def test_faster_complete_fit_with_genuinely_worse_quality_is_not_a_win(pair):
    cpu, cuda = pair
    _, validation = workload(32, "squared")
    cuda["validation_raw"] = (np.asarray(cuda["validation_raw"]) + 3).tolist()
    cuda["quality"] = quality(validation, cuda["validation_raw"], "squared")
    result = judge(cpu, cuda)
    assert result["measurement_complete"] and not result["quality_comparable"]
    assert result["cpu_over_cuda_warm_fit_ratio"] is None


def test_direct_metrics_match_independent_normal_oracle():
    from .reference.coupled import normal_scores

    _, validation = workload(32, "normal")
    raw = np.zeros_like(validation.offset)
    actual = quality(validation, raw, "normal")
    nll, crps = normal_scores(
        raw + validation.offset, validation.target[:, 0], weight=validation.weight
    )
    assert actual == pytest.approx(dict(nll=nll, crps=crps), abs=1e-12)


def test_workload_repeatability_and_disjoint_holdout():
    first, other = workload(32, "squared"), workload(32, "squared")
    assert [p.identity for p in first] == [p.identity for p in other]
    assert not set(first[0].row_ids) & set(first[1].row_ids)
    assert np.any(first[0].weight == 0) and np.isnan(first[0].data.values).any()


def test_timeout_without_child_provenance_withholds_ratio():
    result = judge(dict(status="timeout"), dict(status="timeout"), expected_sources={"core": "sha"})
    assert not result["measurement_complete"]
    assert result["cpu_over_cuda_warm_fit_ratio"] is None


def test_actual_child_deadline_preserves_completed_partial_work(tmp_path, monkeypatch):
    import json
    import sys

    from . import test_early_performance_cuda as supervisor

    script = tmp_path / "benchmarks/v1/performance_checkpoint.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json, pathlib, sys, time\n"
        "p = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "p.write_text(json.dumps({'status': 'running', 'measurements': [1]}))\n"
        "time.sleep(20)\n"
    )
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setenv("OPENBOOST_FRESH_CPU_PYTHON", sys.executable)
    output = tmp_path / "result.json"
    result = supervisor.child("normal", 1000, "cpu", output, 0.2)
    assert result["status"] == "timeout" and result["measurements"] == [1]
    assert result["child_wall_seconds"] < 5
    assert json.loads(output.read_text()) == result
