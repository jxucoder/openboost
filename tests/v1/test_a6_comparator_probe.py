"""A bounded comparator preflight cannot silently omit a required method."""

import json
from pathlib import Path

import pytest
from benchmarks.v1.a6_resource_preflight import comparator_jobs, main


def frozen():
    return json.loads(Path("v1-sprints/070-a6-cpu-search-plan-bins.json").read_text())["plan"]


def test_three_frozen_probes():
    jobs = comparator_jobs(frozen())
    assert [j["library"] for j in jobs] == ["xgboost", "lightgbm", "catboost"]
    assert all(j["fold"] == 0 and j["config"]["rounds"] == 300 for j in jobs)
    assert all(j["config"]["bins"] == 255 and j["early_stopping_rounds"] == 50 for j in jobs)


@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_omitted_or_duplicate_probe_rejected(fault):
    plan = frozen()
    job = comparator_jobs(plan)[1]
    if fault == "missing":
        plan["jobs"].remove(job)
    else:
        plan["jobs"].append(job)
    with pytest.raises(ValueError, match="unique"):
        comparator_jobs(plan)


@pytest.mark.parametrize("mode", ["profile", "paired"])
def test_comparator_mode_exclusive(tmp_path, mode):
    with pytest.raises(ValueError, match="separate"):
        main(tmp_path / "out", tmp_path, comparators=True, **{mode: True})
    assert not (tmp_path / "out").exists()
