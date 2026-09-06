"""Explicit training preparation reuse without sharing run state."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import MixedData, Problem, RunContext
from openboost.binning import Binning, PreparedData, prepare_training
from openboost.recipes import multi_squared, normal, poisson, squared
from openboost.runs import RunSpec, run_many


def fixture():
    data = MixedData(
        [[0, "a"], [1, "b"], [2, None], [3, "a"], [4, "c"], [None, "b"]],
        [0, 1, 2, 3, 4, 5],
        ("x", "c"),
        ("numeric", "categorical"),
    )
    y = np.array([1, 2, 4, 3, 8, 2])[:, None]
    scalar = Problem(data, y, data.row_ids)
    distribution = replace(scalar, raw_width=2, offset=np.zeros((6, 2)))
    count = replace(scalar, structure={"exposure": np.ones((6, 1))})
    multi = replace(
        scalar, target=np.column_stack((y, 2 * y)), raw_width=2, offset=np.zeros((6, 2))
    )
    return data, [
        (squared, scalar),
        (normal, distribution),
        (poisson, count),
        (multi_squared, multi),
    ]


@pytest.mark.parametrize("count", [1, 8, 32])
def test_shared_independent_reordered_and_regrouped_runs(count, monkeypatch):
    data, jobs = fixture()
    prepared = PreparedData(data, bins=4)
    specs = tuple(
        RunSpec(
            RunContext(f"job-{i}", i),
            jobs[i % 4][1],
            jobs[i % 4][1],
            jobs[i % 4][0],
            {"bins": 4, "rounds": 1 + i % 3},
            prepared=prepared,
        )
        for i in range(count)
    )
    independent = {
        s.context.run_id: s.recipe(s.train, s.validation, context=s.context, **s.options)
        for s in specs
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("supplied preparation must not refit")

    monkeypatch.setattr(Binning, "fit", forbidden)
    outcomes = run_many(specs)
    reversed_results = {r.run_id: r for r in run_many(reversed(specs))}
    regrouped = {r.run_id: r for group in (specs[::2], specs[1::2]) for r in run_many(group)}
    for i, outcome in enumerate(outcomes):
        assert outcome.error_type is None
        state = outcome.result.state
        assert state.identity == independent[outcome.run_id].state.identity
        assert state.identity == reversed_results[outcome.run_id].result.state.identity
        assert state.identity == regrouped[outcome.run_id].result.state.identity
        for other in outcomes[:i]:
            assert not np.shares_memory(state.train_raw, other.result.state.train_raw)
    assert all(s.prepared is prepared for s in specs)


def test_preparation_identity_config_and_owned_codes():
    data, jobs = fixture()
    prepared = PreparedData(data, 4)
    assert prepare_training(data, bins=4, prepared=prepared) is prepared.binned
    assert prepared.identity == PreparedData(data, 4).identity
    assert prepared.identity != PreparedData(data, 3).identity
    with pytest.raises(ValueError):
        prepared.binned.codes.setflags(write=True)
    changed = MixedData(data.values, data.row_ids + 10, data.feature_names, data.feature_kinds)
    for d, bins in [(data, 3), (changed, 4), (data, True)]:
        with pytest.raises(ValueError, match="identity"):
            prepare_training(d, bins=bins, prepared=prepared)
    # Target/weight changes may share exactly the same feature preparation.
    p = replace(jobs[0][1], target=jobs[0][1].target + 3, weight=[1, 2, 3, 4, 5, 6])
    result = squared(p, p, context=RunContext("different-target", 1), bins=4, prepared=prepared)
    expected = squared(p, p, context=RunContext("different-target", 1), bins=4)
    assert result.state.identity == expected.state.identity


def test_failed_preparation_does_not_contaminate_other_run():
    data, jobs = fixture()
    p = jobs[0][1]
    prepared = PreparedData(data, 4)
    specs = [
        RunSpec(RunContext("bad", 1), p, p, squared, {"bins": 3}, prepared),
        RunSpec(RunContext("good", 2), p, p, squared, {"bins": 4}, prepared),
    ]
    results = run_many(specs)
    assert results[0].error_type == "ValueError" and results[0].result is None
    assert results[1].result.state.version == 2
    with pytest.raises(ValueError):
        RunSpec(RunContext("ambiguous", 1), p, p, squared, {"prepared": None}, prepared)

