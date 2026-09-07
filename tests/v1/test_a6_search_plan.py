"""The A6 CPU plan must retain every candidate and comparator trial."""

import copy
import json
from pathlib import Path

import pytest
from benchmarks.v1.a6_search_plan import compile_plan, validate_plan


def design():
    return json.loads(Path("benchmarks/v1/search-design.json").read_text())


def test_complete_cpu_methods():
    frozen = design()
    before = copy.deepcopy(frozen)
    plan = compile_plan(frozen)
    assert frozen == before
    assert plan["job_count"] == 400
    assert len({j["id"] for j in plan["jobs"]}) == 400
    assert plan["maximum_sequential_worker_seconds"] == 720000
    assert plan["maximum_reserved_cpu_seconds"] == 1440000
    assert not plan["dispatch_ready"] and plan["outcome"] == "not_run"
    for method in ("openboost-shared", "openboost-independent", "xgboost", "lightgbm", "catboost"):
        library = method.split("-")[0]
        family = "xgboost" if library == "openboost" else library
        for fold in range(5):
            jobs = [j for j in plan["jobs"] if j["id"].startswith(f"{method}:{fold}:")]
            assert len(jobs) == 16
            for job, cfg in zip(jobs, frozen["families"][family], strict=True):
                assert {k: v for k, v in job["config"].items() if k not in ("bins", "mode")} == cfg
                assert job["config"]["bins"] == 255
                assert job["threads"] == 1 and job["seed"] == fold
                assert job["early_stopping_rounds"] == 50
    validate_plan(plan, frozen)


@pytest.mark.parametrize(
    "fault", ["method", "fold", "trial", "duplicate", "config", "budget", "ready"]
)
def test_rehashed_producer_plan_cannot_shrink_scope(fault):
    frozen = design()
    plan = compile_plan(frozen)
    if fault == "method":
        plan["jobs"] = [j for j in plan["jobs"] if j["library"] != "catboost"]
        del plan["methods"]["catboost"]
    elif fault == "fold":
        plan["jobs"] = [j for j in plan["jobs"] if j["fold"] != 4]
    elif fault == "trial":
        plan["jobs"].pop()
    elif fault == "duplicate":
        plan["jobs"][-1] = plan["jobs"][0]
    elif fault == "config":
        plan["jobs"][-1]["config"]["rounds"] = 4
    elif fault == "budget":
        plan["policy"]["timeout_s"] = 60
    else:
        plan["dispatch_ready"] = True
    plan["job_count"] = len(plan["jobs"])
    with pytest.raises(ValueError, match="evaluator-owned"):
        validate_plan(plan, frozen)


@pytest.mark.parametrize("library", ["xgboost", "lightgbm", "catboost"])
def test_missing_frozen_configuration_is_rejected(library):
    frozen = design()
    frozen["families"][library].pop()
    with pytest.raises(ValueError):
        compile_plan(frozen)
