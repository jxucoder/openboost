import copy
import json
from pathlib import Path

import pytest
from benchmarks.v1.a6_preflight_plan import compile_plan


def design():
    return json.loads(
        (Path(__file__).resolve().parents[2] / "benchmarks/v1/search-design.json").read_text()
    )


def test_full_frozen_openboost_matrix_preserves_every_configuration():
    frozen = design()
    original = copy.deepcopy(frozen)
    plan = compile_plan(frozen)
    assert frozen == original
    assert plan["job_count"] == 160
    assert plan["maximum_sequential_worker_seconds"] == 288000
    assert plan["maximum_reserved_cpu_seconds"] == 576000
    assert len({j["id"] for j in plan["jobs"]}) == 160
    for mode in ("shared", "independent"):
        for fold in range(5):
            jobs = [j for j in plan["jobs"] if j["fold"] == fold and j["config"]["mode"] == mode]
            assert len(jobs) == 16
            for job, cfg in zip(jobs, frozen["families"]["xgboost"], strict=True):
                assert job["config"] == {**cfg, "mode": mode, "bins": 255}
                assert job["early_stopping_rounds"] == 50
                assert job["seed"] == fold and job["threads"] == 1
                assert "input_npz" not in job


@pytest.mark.parametrize("fault", ["omit", "duplicate", "short", "unknown", "budget", "fold"])
def test_preflight_rejects_incomplete_or_reinterpreted_design(fault):
    frozen = design()
    configs = frozen["families"]["xgboost"]
    if fault == "omit":
        configs.pop()
    elif fault == "duplicate":
        configs[1] = configs[0]
    elif fault == "short":
        configs[0]["rounds"] = 4
    elif fault == "unknown":
        configs[0]["unhandled_sampling"] = 0.5
    elif fault == "budget":
        frozen["budgets"]["trial_wall_s"] = 60
    else:
        frozen["selection"]["folds"] = 1
    with pytest.raises(ValueError):
        compile_plan(frozen)
