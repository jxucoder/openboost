"""CPU controls for frozen 120 schedules; no production device execution."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import device_recipes, recipes
from openboost import tree as cpu_trees
from openboost.device_runs import RunSpec
from openboost.runtime import RunContext

from .reference.device_runs import case, run
from .test_device_round_reference import prepared_fixture


def jobs(count):
    train, validation, binned, _ = prepared_fixture("weighted")
    result = []
    for i in range(count):
        f, options = case(i)
        t = replace(train, target=f["target"][:, None], weight=f["weight"])
        v = replace(validation, target=f["validation_target"][:, None])
        result.append(RunSpec(f"job-{i}", 17, t, v, device_recipes.squared,
                              options=options, binning=binned.binning))
    return tuple(result)


def cpu(job):
    options = dict(job.options)
    depth = options.pop("max_depth")

    def learner(binned, fields):
        return cpu_trees.depthwise(job.binning.transform(binned.data), fields, max_depth=depth)

    return recipes.squared(job.train, job.validation, context=RunContext(job.run_id, job.seed),
                           learner=learner, bins=4, **options)


@pytest.mark.parametrize("index", range(16))
def test_independent_original_row_outcomes_match_public_cpu(index):
    job = jobs(index + 1)[-1]
    actual, expected = cpu(job), run(index)
    state = actual.state
    for actual_raw, expected_raw in (
        (state.train_raw[:, 0], expected["raw"]),
        (state.validation_raw[:, 0], expected["validation_raw"]),
        (state.best_model.predict(job.validation.data)[:, 0], expected["best_validation_raw"]),
    ):
        np.testing.assert_allclose(actual_raw, expected_raw, rtol=1e-12, atol=1e-12)
    assert state.version == expected["version"]
    assert len(state.best_model.terms) == expected["best_prefix"]
    assert actual.stop.completed_rounds == expected["completed"]
    assert actual.stop.reason == expected["reason"]
    assert actual.stop.stale_rounds == expected["stale"]
    assert [s.accepted for s in actual.steps] == [s["accepted"] for s in expected["steps"]]


@pytest.mark.parametrize("count", [1, 8, 32])
def test_frozen_schedule_has_stable_ids_shared_cuts_and_independent_streams(count):
    specs = jobs(count)
    assert len({s.run_id for s in specs}) == count
    assert all(s.binning is specs[0].binning for s in specs)
    streams = {s.run_id: tuple(RunContext(s.run_id, s.seed).rng(0, "learner", "rows").integers(0, 2**32, 8))
               for s in specs}
    assert len(set(streams.values())) == count
    for s in reversed(specs):
        assert tuple(RunContext(s.run_id, s.seed).rng(0, "learner", "rows").integers(0, 2**32, 8)) == streams[s.run_id]
    if count > 1:
        assert {run(i)["reason"] for i in range(count)} == {"budget", "patience"}
        assert any(not row["accepted"] for i in range(count) for row in run(i)["steps"])
